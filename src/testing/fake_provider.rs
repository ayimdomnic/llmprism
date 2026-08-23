use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

use crate::error::Error;
use crate::moderation::{ModerationRequest, ModerationResponse};
use crate::provider::Provider;
use crate::stream_event::StreamEvent;
use crate::structured::{StructuredRequest, StructuredResponse};
use crate::text::{Step, TextRequest};

const DEFAULT_STREAM_CHUNK_WORDS: usize = 1;

/// A stand-in for a real provider, for use in tests.
///
/// Register it into a [`Registry`](crate::Registry) under a normal provider name
/// (e.g. `"openai"`) and it behaves like the real thing as far as your
/// application code can tell -- except instead of making a network call, it
/// hands back canned responses you scripted ahead of time with
/// [`respond_with`](Self::respond_with), and records every request it received
/// so you can assert on what your code actually sent.
///
/// # Example
///
/// ```
/// # #[tokio::main]
/// # async fn main() {
/// use llmprism::testing::{FakeProvider, FakeTextResponse};
/// use llmprism::Registry;
///
/// let fake = FakeProvider::new("openai").respond_with(FakeTextResponse::new("Hello!"));
///
/// let mut registry = Registry::new();
/// registry.register("openai", fake);
///
/// // Code under test just calls `registry.text("openai", ...)` as normal, with
/// // no idea it's talking to a fake.
/// let response = registry.text("openai", "gpt-4o-mini").unwrap()
///     .with_prompt("hi")
///     .generate()
///     .await
///     .unwrap();
///
/// assert_eq!(response.text.as_deref(), Some("Hello!"));
/// # }
/// ```
///
/// The same scripted responses work for [`stream`](crate::text::PendingTextRequest::stream)
/// too -- `FakeProvider` synthesizes [`StreamEvent`]s from the canned [`Step`]
/// instead of needing a second, stream-shaped fixture format. By default it
/// yields one word per [`StreamEvent::TextDelta`]; change that with
/// [`with_stream_chunk_words`](Self::with_stream_chunk_words).
///
/// Every capability keeps its own independent queue (text, structured output,
/// moderation, and so on) -- a test exercising more than one capability
/// against the same fake scripts each separately, e.g.
/// [`respond_with_structured`](Self::respond_with_structured) alongside
/// [`respond_with`](Self::respond_with).
pub struct FakeProvider {
    name: String,
    text: ResponseQueue<TextRequest, Step>,
    structured: ResponseQueue<StructuredRequest, StructuredResponse>,
    moderation: ResponseQueue<ModerationRequest, ModerationResponse>,
    stream_chunk_words: usize,
}

impl FakeProvider {
    /// Creates a fake provider that will report `name` from [`Provider::name`],
    /// with no canned responses queued yet.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            text: ResponseQueue::new(),
            structured: ResponseQueue::new(),
            moderation: ResponseQueue::new(),
            stream_chunk_words: DEFAULT_STREAM_CHUNK_WORDS,
        }
    }

    /// Queues one canned response, to be returned the next time this provider is
    /// called. Call this once per request/response round trip you expect your
    /// test to trigger -- for example, twice if you're testing a single tool
    /// call followed by a final reply.
    pub fn respond_with(self, step: impl Into<Step>) -> Self {
        self.text.push(step.into());
        self
    }

    /// Queues one canned structured-output response, to be returned the next
    /// time this provider receives a structured request. Kept as a separate
    /// queue from [`respond_with`](Self::respond_with) since a structured
    /// response is a different shape ([`StructuredResponse`], not [`Step`]) --
    /// a test that exercises both capabilities scripts each independently.
    pub fn respond_with_structured(self, response: impl Into<StructuredResponse>) -> Self {
        self.structured.push(response.into());
        self
    }

    /// Queues one canned moderation response, to be returned the next time
    /// this provider receives a moderation request.
    pub fn respond_with_moderation(self, response: impl Into<ModerationResponse>) -> Self {
        self.moderation.push(response.into());
        self
    }

    /// Controls how many words [`stream_text_once`](Provider::stream_text_once)
    /// groups into each [`StreamEvent::TextDelta`] it synthesizes from a canned
    /// response's text. Defaults to one word per delta; raise it if a test wants
    /// fewer, larger chunks.
    pub fn with_stream_chunk_words(mut self, words: usize) -> Self {
        self.stream_chunk_words = words.max(1);
        self
    }

    /// Returns every request this provider actually received, in order --
    /// useful for asserting on exactly what your code sent (which model, which
    /// messages, which tools, and so on).
    pub fn recorded_requests(&self) -> Vec<TextRequest> {
        self.text.recorded()
    }

    /// Returns every structured request this provider actually received, in
    /// order -- the structured-output counterpart to
    /// [`recorded_requests`](Self::recorded_requests).
    pub fn recorded_structured_requests(&self) -> Vec<StructuredRequest> {
        self.structured.recorded()
    }

    /// Returns every moderation request this provider actually received, in
    /// order.
    pub fn recorded_moderation_requests(&self) -> Vec<ModerationRequest> {
        self.moderation.recorded()
    }
}

#[async_trait]
impl Provider for FakeProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
        Ok(self.text.next(request, &self.name, "text", "respond_with"))
    }

    async fn stream_text_once(
        &self,
        request: TextRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let step = self.text.next(request, &self.name, "text", "respond_with");

        let mut events = vec![StreamEvent::StreamStart {
            meta: step.meta.clone(),
        }];

        if let Some(text) = &step.text {
            for chunk in chunk_words(text, self.stream_chunk_words) {
                events.push(StreamEvent::TextDelta { text: chunk });
            }
        }
        for call in &step.tool_calls {
            events.push(StreamEvent::ToolCall(call.clone()));
        }
        events.push(StreamEvent::StepFinish {
            usage: step.usage,
            finish_reason: step.finish_reason,
        });

        Ok(stream::iter(events.into_iter().map(Ok)).boxed())
    }

    async fn structured(&self, request: StructuredRequest) -> Result<StructuredResponse, Error> {
        Ok(self
            .structured
            .next(request, &self.name, "structured", "respond_with_structured"))
    }

    async fn moderation(&self, request: ModerationRequest) -> Result<ModerationResponse, Error> {
        Ok(self
            .moderation
            .next(request, &self.name, "moderation", "respond_with_moderation"))
    }
}

/// One capability's queue of canned responses plus a log of the requests it
/// actually received. Every `Fake*` queue on [`FakeProvider`] is one of these
/// -- factored out so the "record the request, pop the next canned response,
/// panic with a clear message if the queue runs out" behavior is written once
/// instead of once per capability.
struct ResponseQueue<Req, Resp> {
    responses: Mutex<Vec<Resp>>,
    recorded: Mutex<Vec<Req>>,
}

impl<Req: Clone, Resp> ResponseQueue<Req, Resp> {
    fn new() -> Self {
        Self {
            responses: Mutex::new(Vec::new()),
            recorded: Mutex::new(Vec::new()),
        }
    }

    fn push(&self, response: Resp) {
        self.responses
            .lock()
            .expect("fake provider mutex poisoned")
            .push(response);
    }

    fn recorded(&self) -> Vec<Req> {
        self.recorded
            .lock()
            .expect("fake provider mutex poisoned")
            .clone()
    }

    /// Records `request` and pops the next canned response off the queue,
    /// panicking with a message that names the right `respond_with_*` method
    /// to call if the test under-scripted this provider.
    fn next(&self, request: Req, provider_name: &str, capability: &str, builder: &str) -> Resp {
        self.recorded
            .lock()
            .expect("fake provider mutex poisoned")
            .push(request);

        let mut responses = self.responses.lock().expect("fake provider mutex poisoned");
        if responses.is_empty() {
            panic!(
                "FakeProvider '{provider_name}' received a {capability} request but has no \
                 more canned {capability} responses queued -- call .{builder}(...) for every \
                 expected call",
            );
        }
        responses.remove(0)
    }
}

/// Splits `text` into `words_per_chunk`-word pieces, keeping a single trailing
/// space on every piece but the last so re-joining the pieces reproduces `text`
/// exactly -- the same property a real provider's word-by-word streaming has.
fn chunk_words(text: &str, words_per_chunk: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut count = 0;

    for word in text.split_inclusive(' ') {
        current.push_str(word);
        count += 1;
        if count >= words_per_chunk {
            chunks.push(std::mem::take(&mut current));
            count = 0;
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}
