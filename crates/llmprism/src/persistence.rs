//! [`ConversationStore`] and [`PersistenceMiddleware`] -- save a
//! conversation's message history after each call and reload it before the
//! next one, keyed by an opaque id, so a caller only has to send the newest
//! turn instead of replaying the whole conversation every time.
//!
//! Attach [`PersistenceMiddleware`] the same way any other
//! [`ProviderMiddleware`] gets attached, via
//! [`Registry::wrap`](crate::Registry::wrap), then opt individual requests
//! into it with
//! [`PendingTextRequest::with_conversation_id`](crate::text::PendingTextRequest::with_conversation_id):
//!
//! ```
//! use llmprism::persistence::{InMemoryConversationStore, PersistenceMiddleware};
//! use llmprism::testing::{FakeProvider, FakeTextResponse};
//! use llmprism::Registry;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let fake = FakeProvider::new("openai")
//!     .respond_with(FakeTextResponse::new("Nice to meet you, Alex."))
//!     .respond_with(FakeTextResponse::new("Your name is Alex."));
//!
//! let mut registry = Registry::new();
//! registry.register("openai", fake);
//! registry
//!     .wrap("openai", PersistenceMiddleware::new(InMemoryConversationStore::new()))
//!     .unwrap();
//!
//! registry
//!     .text("openai", "gpt-4o-mini")
//!     .unwrap()
//!     .with_conversation_id("session-42")
//!     .with_prompt("My name is Alex.")
//!     .generate()
//!     .await
//!     .unwrap();
//!
//! // A later call under the same id sees the first turn automatically --
//! // the caller only has to send the newest message.
//! let response = registry
//!     .text("openai", "gpt-4o-mini")
//!     .unwrap()
//!     .with_conversation_id("session-42")
//!     .with_prompt("What's my name?")
//!     .generate()
//!     .await
//!     .unwrap();
//!
//! assert_eq!(response.text.as_deref(), Some("Your name is Alex."));
//! # }
//! ```
//!
//! A request with no [`conversation_id`](crate::text::TextRequest::conversation_id)
//! set passes straight through, untouched -- persistence is opt-in per
//! request, not forced on every call through a wrapped provider.
//!
//! `ConversationStore`'s `load`/`save` failing propagates as
//! [`Error::Store`], failing the whole `generate()`/`stream()` call even
//! though the model itself may have already replied successfully. That's a
//! deliberate trade-off, not an oversight: silently swallowing a failed
//! save would let a conversation quietly drift out of sync with what's
//! actually stored, which is worse than a caller finding out immediately
//! that persistence broke.
//!
//! Only `llmprism` core's own in-memory reference implementation
//! ([`InMemoryConversationStore`]) ships here -- real backends (Postgres,
//! SQLite, Redis) are meant to ship as separate, independent crates, each a
//! thin `ConversationStore` impl, so using one doesn't force that
//! database's client library onto every other consumer of this crate.

use std::collections::HashMap;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use tokio::sync::RwLock;

use crate::error::Error;
use crate::middleware::ProviderMiddleware;
use crate::provider::Provider;
use crate::stream_event::StreamEvent;
use crate::text::{Step, TextRequest};
use crate::value_objects::{AssistantMessage, FinishReason, Message, ToolCall};

/// Saves and loads a conversation's message history by an opaque id.
///
/// Implement this against whatever storage your application already uses
/// (a database table, a cache, a file) and attach it via
/// [`PersistenceMiddleware`] -- see the [module docs](self) for a full
/// example.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// Loads the stored history for `id`, oldest message first. An id
    /// that's never been saved returns an empty history (a fresh
    /// conversation), not an error.
    async fn load(&self, id: &str) -> Result<Vec<Message>, Error>;

    /// Overwrites the stored history for `id` with `messages`. This
    /// replaces the previous history in full -- it isn't an append -- since
    /// [`PersistenceMiddleware`] always calls it with the complete
    /// conversation so far.
    async fn save(&self, id: &str, messages: &[Message]) -> Result<(), Error>;
}

/// A [`ConversationStore`] backed by an in-process `HashMap`.
///
/// Useful on its own for tests and single-process applications --
/// conversations don't survive a restart and aren't visible to any other
/// process. For anything that needs to, implement [`ConversationStore`]
/// against a real backend instead; see the [module docs](self) for why
/// that's meant to live in its own crate rather than here.
#[derive(Debug, Default)]
pub struct InMemoryConversationStore {
    conversations: RwLock<HashMap<String, Vec<Message>>>,
}

impl InMemoryConversationStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Lets an already-shared `Arc<impl ConversationStore>` be passed directly
/// to [`PersistenceMiddleware::new`] -- handy when a test or application
/// also wants to keep its own handle to the same store (to inspect it, or
/// to reuse it outside this middleware).
#[async_trait]
impl<T: ConversationStore + ?Sized> ConversationStore for Arc<T> {
    async fn load(&self, id: &str) -> Result<Vec<Message>, Error> {
        T::load(self, id).await
    }

    async fn save(&self, id: &str, messages: &[Message]) -> Result<(), Error> {
        T::save(self, id, messages).await
    }
}

#[async_trait]
impl ConversationStore for InMemoryConversationStore {
    async fn load(&self, id: &str) -> Result<Vec<Message>, Error> {
        Ok(self
            .conversations
            .read()
            .await
            .get(id)
            .cloned()
            .unwrap_or_default())
    }

    async fn save(&self, id: &str, messages: &[Message]) -> Result<(), Error> {
        self.conversations
            .write()
            .await
            .insert(id.to_string(), messages.to_vec());
        Ok(())
    }
}

/// A round trip is the *final* one for a call (the one whose reply should
/// actually be persisted) once the model isn't asking for another tool
/// call -- the same condition [`crate::tool_loop`]/[`crate::stream_loop`]
/// themselves use to decide whether to keep looping.
fn is_final_step(finish_reason: FinishReason, tool_calls: &[ToolCall]) -> bool {
    finish_reason != FinishReason::ToolCalls || tool_calls.is_empty()
}

/// A [`ProviderMiddleware`] that loads a request's conversation history
/// before sending it and saves the updated history back after, via a
/// [`ConversationStore`]. See the [module docs](self) for how to attach and
/// use one.
///
/// Only intercepts [`Provider::text_step`]/[`Provider::stream_text_once`]
/// (every other capability has no notion of an ongoing conversation to
/// persist). [`Provider::text_step`]/`stream_text_once` are called once per
/// round trip of a multi-step tool-calling loop, not once per
/// `generate()`/`stream()` call -- this middleware still only loads/merges
/// history fresh on every round trip (safe, since a middleware's merge
/// never mutates the tool loop's own accumulating request) and only saves
/// once, on the round trip that actually ends the call, rather than once
/// per round trip.
pub struct PersistenceMiddleware<S: ConversationStore> {
    store: Arc<S>,
}

impl<S: ConversationStore> PersistenceMiddleware<S> {
    /// Wraps `store` as a [`ProviderMiddleware`], ready to attach with
    /// [`Registry::wrap`](crate::Registry::wrap).
    pub fn new(store: S) -> Self {
        Self {
            store: Arc::new(store),
        }
    }
}

#[async_trait]
impl<S: ConversationStore + 'static> ProviderMiddleware for PersistenceMiddleware<S> {
    async fn text_step(
        &self,
        mut request: TextRequest,
        next: &dyn Provider,
    ) -> Result<Step, Error> {
        let Some(id) = request.conversation_id.clone() else {
            return next.text_step(&request).await;
        };

        let history = self.store.load(&id).await?;
        let new_turn = std::mem::take(&mut request.messages);
        request.messages = history
            .iter()
            .cloned()
            .chain(new_turn.iter().cloned())
            .collect();

        let step = next.text_step(&request).await?;

        if is_final_step(step.finish_reason, &step.tool_calls) {
            let mut to_save = history;
            to_save.extend(new_turn);
            to_save.push(Message::Assistant(AssistantMessage {
                content: step.text.clone(),
                tool_calls: step.tool_calls.clone(),
            }));
            self.store.save(&id, &to_save).await?;
        }

        Ok(step)
    }

    async fn stream_text_once(
        &self,
        mut request: TextRequest,
        next: &dyn Provider,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let Some(id) = request.conversation_id.clone() else {
            return next.stream_text_once(&request).await;
        };

        let history = self.store.load(&id).await?;
        let new_turn = std::mem::take(&mut request.messages);
        request.messages = history
            .iter()
            .cloned()
            .chain(new_turn.iter().cloned())
            .collect();

        let mut inner = next.stream_text_once(&request).await?;
        let store = Arc::clone(&self.store);

        let stream = try_stream! {
            let mut text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut finish_reason = FinishReason::Stop;

            while let Some(event) = inner.next().await {
                let event = event?;

                match &event {
                    StreamEvent::TextDelta { text: delta } => text.push_str(delta),
                    StreamEvent::ToolCall(call) => tool_calls.push(call.clone()),
                    StreamEvent::StepFinish { finish_reason: step_finish_reason, .. } => {
                        finish_reason = *step_finish_reason;
                    }
                    StreamEvent::StreamStart { .. }
                    | StreamEvent::ToolCallDelta { .. }
                    | StreamEvent::ToolResult(_)
                    | StreamEvent::StreamEnd { .. } => {}
                }

                yield event;
            }

            if is_final_step(finish_reason, &tool_calls) {
                let mut to_save = history;
                to_save.extend(new_turn);
                to_save.push(Message::Assistant(AssistantMessage {
                    content: if text.is_empty() { None } else { Some(text) },
                    tool_calls,
                }));
                store.save(&id, &to_save).await?;
            }
        };

        Ok(stream.boxed())
    }
}
