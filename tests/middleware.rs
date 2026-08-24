//! Exercises `ProviderMiddleware`/`Registry::wrap` against `FakeProvider` --
//! no network access, no feature flags required.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use llmprism::middleware::ProviderMiddleware;
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::text::{Step, TextRequest};
use llmprism::{Error, Provider, Registry};

/// Rewrites the request before it reaches the wrapped provider -- confirms a
/// middleware can actually transform what gets sent, not just observe it.
struct AddSystemPrompt(&'static str);

#[async_trait]
impl ProviderMiddleware for AddSystemPrompt {
    async fn text_step(
        &self,
        mut request: TextRequest,
        next: &dyn Provider,
    ) -> Result<Step, Error> {
        request.system_prompts.insert(0, self.0.to_string());
        next.text_step(request).await
    }
}

#[tokio::test]
async fn a_middleware_can_rewrite_the_request_before_it_reaches_the_provider() {
    // Held onto directly (as well as registered) so its `recorded_requests`
    // can be inspected afterward -- `Registry::wrap` replaces the
    // registration with a `MiddlewareProvider`, so `registry.provider("fake")`
    // alone wouldn't give access to the wrapped `FakeProvider` anymore.
    let fake = Arc::new(FakeProvider::new("fake").respond_with(FakeTextResponse::new("hi")));

    let mut registry = Registry::new();
    registry.register_arc("fake", fake.clone());
    registry
        .wrap("fake", AddSystemPrompt("Be concise."))
        .unwrap();

    registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("hello")
        .generate()
        .await
        .unwrap();

    let recorded = fake.recorded_requests();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].system_prompts, vec!["Be concise."]);
}

/// A caching middleware that never calls `next` on a hit -- confirms a
/// middleware can short-circuit the call entirely, not just wrap it.
struct CountCallsAndCacheFirstResult {
    calls: Arc<AtomicU32>,
    cached: std::sync::Mutex<Option<Step>>,
}

#[async_trait]
impl ProviderMiddleware for CountCallsAndCacheFirstResult {
    async fn text_step(&self, request: TextRequest, next: &dyn Provider) -> Result<Step, Error> {
        if let Some(step) = self.cached.lock().unwrap().clone() {
            return Ok(step);
        }

        self.calls.fetch_add(1, Ordering::SeqCst);
        let step = next.text_step(request).await?;
        *self.cached.lock().unwrap() = Some(step.clone());
        Ok(step)
    }
}

#[tokio::test]
async fn a_middleware_can_short_circuit_and_never_call_the_wrapped_provider() {
    let provider = FakeProvider::new("fake").respond_with(FakeTextResponse::new("cached"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let calls = Arc::new(AtomicU32::new(0));
    registry
        .wrap(
            "fake",
            CountCallsAndCacheFirstResult {
                calls: calls.clone(),
                cached: std::sync::Mutex::new(None),
            },
        )
        .unwrap();

    for _ in 0..3 {
        let response = registry
            .text("fake", "test-model")
            .unwrap()
            .with_prompt("hello")
            .generate()
            .await
            .unwrap();
        assert_eq!(response.text.as_deref(), Some("cached"));
    }

    // `FakeProvider` only had one response queued -- if the middleware had
    // called `next` more than once, the second and third calls would have
    // panicked on an empty queue instead of returning the cached value.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn wrap_returns_an_error_for_an_unregistered_name() {
    let mut registry = Registry::new();
    let result = registry.wrap("does-not-exist", AddSystemPrompt("x"));
    assert!(matches!(result, Err(Error::UnknownProvider(_))));
}
