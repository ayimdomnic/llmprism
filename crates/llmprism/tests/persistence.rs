//! Exercises `ConversationStore`/`PersistenceMiddleware` against
//! `FakeProvider` -- no network access, no feature flags required.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use llmprism::error::ToolError;
use llmprism::persistence::{ConversationStore, InMemoryConversationStore, PersistenceMiddleware};
use llmprism::schema::{ObjectSchema, Schema, StringSchema};
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::tool::Tool;
use llmprism::value_objects::{Message, ToolOutput};
use llmprism::{Error, Registry};
use serde_json::Value;

/// Wraps an [`InMemoryConversationStore`] and counts how many times
/// `load`/`save` were actually called, so a test can assert on *whether*
/// the store was touched at all, not just on the resulting content.
#[derive(Default)]
struct CountingConversationStore {
    inner: InMemoryConversationStore,
    loads: AtomicU32,
    saves: AtomicU32,
}

#[async_trait]
impl ConversationStore for CountingConversationStore {
    async fn load(&self, id: &str) -> Result<Vec<Message>, Error> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.inner.load(id).await
    }

    async fn save(&self, id: &str, messages: &[Message]) -> Result<(), Error> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        self.inner.save(id, messages).await
    }
}

struct EchoTool {
    parameters: ObjectSchema,
}

impl EchoTool {
    fn new() -> Self {
        Self {
            parameters: ObjectSchema::new("parameters")
                .with_property(Schema::String(StringSchema::new("message")), true),
        }
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes the given message back"
    }

    fn parameters(&self) -> &ObjectSchema {
        &self.parameters
    }

    async fn call(&self, _args: Value) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::from("echoed"))
    }
}

#[tokio::test]
async fn a_second_call_under_the_same_conversation_id_sees_the_first_turns_history() {
    let fake = Arc::new(
        FakeProvider::new("fake")
            .respond_with(FakeTextResponse::new("Nice to meet you, Alex."))
            .respond_with(FakeTextResponse::new("Your name is Alex.")),
    );

    let mut registry = Registry::new();
    registry.register_arc("fake", fake.clone());
    registry
        .wrap(
            "fake",
            PersistenceMiddleware::new(InMemoryConversationStore::new()),
        )
        .unwrap();

    registry
        .text("fake", "test-model")
        .unwrap()
        .with_conversation_id("session-42")
        .with_prompt("My name is Alex.")
        .generate()
        .await
        .unwrap();

    let response = registry
        .text("fake", "test-model")
        .unwrap()
        .with_conversation_id("session-42")
        .with_prompt("What's my name?")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.text.as_deref(), Some("Your name is Alex."));

    let recorded = fake.recorded_requests();
    assert_eq!(recorded.len(), 2);
    // The second call's provider-bound request carries the full history --
    // the caller only ever sent the newest message.
    assert_eq!(recorded[1].messages.len(), 3);
}

#[tokio::test]
async fn a_request_with_no_conversation_id_never_touches_the_store() {
    let fake = FakeProvider::new("fake").respond_with(FakeTextResponse::new("hi"));

    let mut registry = Registry::new();
    registry.register("fake", fake);
    let store = Arc::new(CountingConversationStore::default());
    registry
        .wrap("fake", PersistenceMiddleware::new(Arc::clone(&store)))
        .unwrap();

    let response = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("hi")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.text.as_deref(), Some("hi"));
    assert_eq!(store.loads.load(Ordering::SeqCst), 0);
    assert_eq!(store.saves.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn streaming_requests_persist_conversation_history_too() {
    let fake = Arc::new(
        FakeProvider::new("fake")
            .respond_with(FakeTextResponse::new("Nice to meet you, Alex."))
            .respond_with(FakeTextResponse::new("Your name is Alex.")),
    );

    let mut registry = Registry::new();
    registry.register_arc("fake", fake.clone());
    registry
        .wrap(
            "fake",
            PersistenceMiddleware::new(InMemoryConversationStore::new()),
        )
        .unwrap();

    let mut first = registry
        .text("fake", "test-model")
        .unwrap()
        .with_conversation_id("session-7")
        .with_prompt("My name is Alex.")
        .stream();
    while first.next().await.is_some() {}

    let mut second = registry
        .text("fake", "test-model")
        .unwrap()
        .with_conversation_id("session-7")
        .with_prompt("What's my name?")
        .stream();
    while second.next().await.is_some() {}

    let recorded = fake.recorded_requests();
    assert_eq!(recorded.len(), 2);
    assert_eq!(recorded[1].messages.len(), 3);
}

#[tokio::test]
async fn a_multi_step_tool_calling_conversation_saves_exactly_once() {
    let fake = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call(
            "call-1",
            "echo",
            serde_json::json!({"message": "hi"}),
        ))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", fake);
    let store = Arc::new(CountingConversationStore::default());
    registry
        .wrap("fake", PersistenceMiddleware::new(Arc::clone(&store)))
        .unwrap();

    registry
        .text("fake", "test-model")
        .unwrap()
        .with_conversation_id("session-tools")
        .with_prompt("use the tool")
        .with_tool(EchoTool::new())
        .with_max_steps(3)
        .generate()
        .await
        .unwrap();

    assert_eq!(store.saves.load(Ordering::SeqCst), 1);
    // Both round trips still load fresh (see the module docs for why that's
    // safe): the tool loop's second step needs the merged history again,
    // since its own accumulating `request.messages` never saw the first
    // step's merge.
    assert_eq!(store.loads.load(Ordering::SeqCst), 2);
}
