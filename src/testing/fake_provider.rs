use std::sync::Mutex;

use async_trait::async_trait;

use crate::error::Error;
use crate::provider::Provider;
use crate::text::{Step, TextRequest};

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
pub struct FakeProvider {
    name: String,
    responses: Mutex<Vec<Step>>,
    recorded: Mutex<Vec<TextRequest>>,
}

impl FakeProvider {
    /// Creates a fake provider that will report `name` from [`Provider::name`],
    /// with no canned responses queued yet.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            responses: Mutex::new(Vec::new()),
            recorded: Mutex::new(Vec::new()),
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

    /// Returns every request this provider actually received, in order --
    /// useful for asserting on exactly what your code sent (which model, which
    /// messages, which tools, and so on).
    pub fn recorded_requests(&self) -> Vec<TextRequest> {
        self.recorded
            .lock()
            .expect("fake provider mutex poisoned")
            .clone()
    }
}

#[async_trait]
impl Provider for FakeProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
        self.recorded
            .lock()
            .expect("fake provider mutex poisoned")
            .push(request.clone());

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
