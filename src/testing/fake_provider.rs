use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

use crate::error::Error;
use crate::provider::Provider;
use crate::stream_event::StreamEvent;
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
pub struct FakeProvider {
    name: String,
    responses: Mutex<Vec<Step>>,
    recorded: Mutex<Vec<TextRequest>>,
    stream_chunk_words: usize,
}

impl FakeProvider {
    /// Creates a fake provider that will report `name` from [`Provider::name`],
    /// with no canned responses queued yet.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            responses: Mutex::new(Vec::new()),
            recorded: Mutex::new(Vec::new()),
            stream_chunk_words: DEFAULT_STREAM_CHUNK_WORDS,
        }
    }

    /// Queues one canned response, to be returned the next time this provider is
    /// called. Call this once per request/response round trip you expect your
    /// test to trigger -- for example, twice if you're testing a single tool
    /// call followed by a final reply.
    pub fn respond_with(self, step: impl Into<Step>) -> Self {
        self.responses
            .lock()
            .expect("fake provider mutex poisoned")
            .push(step.into());
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
        self.recorded
            .lock()
            .expect("fake provider mutex poisoned")
            .clone()
    }

    /// Records `request` and pops the next canned `Step` off the queue --
    /// the behavior `text_step` and `stream_text_once` both build on.
    fn next_step(&self, request: TextRequest) -> Result<Step, Error> {
        self.recorded
            .lock()
            .expect("fake provider mutex poisoned")
            .push(request);

        let mut responses = self.responses.lock().expect("fake provider mutex poisoned");
        if responses.is_empty() {
            panic!(
                "FakeProvider '{}' received a request but has no more canned responses queued -- \
                 call .respond_with(...) for every expected round trip",
                self.name
            );
        }
        Ok(responses.remove(0))
    }
}

#[async_trait]
impl Provider for FakeProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
        self.next_step(request)
    }

    async fn stream_text_once(
        &self,
        request: TextRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let step = self.next_step(request)?;

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
