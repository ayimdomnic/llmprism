//! Integration tests for the retry behavior documented on
//! `client::build_http_client`/`build_http_client_with_max_retries` -- confirms
//! actual request counts against a mock server rather than just checking a
//! retry *decision* in isolation. Uses a stateful responder so the sequence of
//! responses (fail, fail, then succeed) is deterministic rather than depending
//! on wiremock's mock-priority rules between two overlapping mocks.

#![cfg(feature = "openai")]

use std::sync::atomic::{AtomicU32, Ordering};

use llmprism::client::build_http_client_with_max_retries;
use llmprism::providers::openai::OpenAiProvider;
use llmprism::text::TextRequest;
use llmprism::Provider;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Returns `429` for the first `fail_count` requests, then a well-formed
/// success response for every request after that.
struct FailNTimesThenSucceed {
    remaining_failures: AtomicU32,
}

impl Respond for FailNTimesThenSucceed {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        if self.remaining_failures.fetch_sub(1, Ordering::SeqCst) > 0 {
            ResponseTemplate::new(429).insert_header("retry-after", "0")
        } else {
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-test",
                "model": "gpt-4o-mini",
                "choices": [{
                    "message": {"role": "assistant", "content": "pong"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1}
            }))
        }
    }
}

#[tokio::test]
async fn a_429_is_retried_and_eventually_succeeds() {
    // Covers the default client too, not just a configured one: `429` is
    // retried by `reqwest-retry`'s `DefaultRetryableStrategy`, which backs
    // both `build_http_client` and `build_http_client_with_max_retries`.
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(FailNTimesThenSucceed {
            remaining_failures: AtomicU32::new(2),
        })
        .expect(3)
        .mount(&mock_server)
        .await;

    let provider = OpenAiProvider::with_base_url("sk-test", mock_server.uri())
        .with_client(build_http_client_with_max_retries(5));

    let step = provider
        .text_step(TextRequest::new("gpt-4o-mini"))
        .await
        .expect("should succeed after retrying past the two 429 responses");

    assert_eq!(step.text.as_deref(), Some("pong"));

    // `.expect(3)` above already asserts the exact call count on drop, but
    // spelling it out here makes the intent obvious without needing to go
    // find that line.
    mock_server.verify().await;
}

#[tokio::test]
async fn the_default_client_retries_a_429_before_giving_up() {
    // The default client (`build_http_client`, what every provider uses
    // unless you opt into a different one) retries a persistent 429 up to
    // its default of 2 times -- 3 requests total -- before finally
    // surfacing `Error::RateLimited` rather than retrying forever.
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429))
        .expect(3)
        .mount(&mock_server)
        .await;

    let provider = OpenAiProvider::with_base_url("sk-test", mock_server.uri());

    let result = provider.text_step(TextRequest::new("gpt-4o-mini")).await;

    assert!(matches!(result, Err(llmprism::Error::RateLimited { .. })));
    mock_server.verify().await;
}

#[tokio::test]
async fn zero_max_retries_surfaces_a_429_on_the_first_attempt() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = OpenAiProvider::with_base_url("sk-test", mock_server.uri())
        .with_client(build_http_client_with_max_retries(0));

    let result = provider.text_step(TextRequest::new("gpt-4o-mini")).await;

    assert!(matches!(result, Err(llmprism::Error::RateLimited { .. })));
    mock_server.verify().await;
}
