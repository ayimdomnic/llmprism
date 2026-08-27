//! Integration tests for the HTTP error-mapping path (`client::ErrorMapper`)
//! against a real (mocked) HTTP server, rather than a real provider. These
//! need no API key -- `wiremock` stands in for the provider, the same way
//! `FakeProvider` stands in for this crate's own `Provider` trait -- so they
//! run in CI and locally with no network access to a real provider at all.
//!
//! Exercised through `OpenAiProvider` specifically (any provider would do,
//! since `ErrorMapper` is shared code -- OpenAI's just the simplest to point
//! at an arbitrary base URL). What matters here is confirming the full
//! path -- an HTTP response with a given status/headers/body actually
//! becomes the right typed `Error` -- not just that `ErrorMapper` alone
//! behaves correctly in isolation.

#![cfg(feature = "openai")]

use llmprism::providers::openai::OpenAiProvider;
use llmprism::text::TextRequest;
use llmprism::{Error, Provider};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(base_url: &str) -> OpenAiProvider {
    OpenAiProvider::with_base_url("sk-test", base_url)
}

#[tokio::test]
async fn a_429_response_becomes_rate_limited_with_headers_parsed() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "30")
                .insert_header("x-ratelimit-limit-requests", "60")
                .insert_header("x-ratelimit-remaining-requests", "0")
                .insert_header("x-ratelimit-reset-requests", "1m0s")
                .set_body_json(serde_json::json!({
                    "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
                })),
        )
        .mount(&mock_server)
        .await;

    let result = provider(&mock_server.uri())
        .text_step(&TextRequest::new("gpt-4o-mini"))
        .await;

    match result {
        Err(Error::RateLimited {
            retry_after,
            limits,
            ..
        }) => {
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(30)));
            assert_eq!(limits.len(), 1);
            assert_eq!(limits[0].name, "requests");
            assert_eq!(limits[0].limit, 60);
            assert_eq!(limits[0].remaining, 0);
        }
        other => panic!("expected Error::RateLimited, got: {other:?}"),
    }
}

#[tokio::test]
async fn a_413_response_becomes_request_too_large() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(413).set_body_json(serde_json::json!({
            "error": {"message": "Request body too large"}
        })))
        .mount(&mock_server)
        .await;

    let result = provider(&mock_server.uri())
        .text_step(&TextRequest::new("gpt-4o-mini"))
        .await;

    match result {
        Err(Error::RequestTooLarge { details, .. }) => {
            assert_eq!(details, "Request body too large");
        }
        other => panic!("expected Error::RequestTooLarge, got: {other:?}"),
    }
}

#[tokio::test]
async fn a_529_response_becomes_overloaded() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(529))
        .mount(&mock_server)
        .await;

    let result = provider(&mock_server.uri())
        .text_step(&TextRequest::new("gpt-4o-mini"))
        .await;

    assert!(matches!(result, Err(Error::Overloaded { .. })));
}

#[tokio::test]
async fn an_unrecognized_error_status_becomes_a_generic_provider_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
            "error": {"message": "Internal server error", "type": "server_error"}
        })))
        .mount(&mock_server)
        .await;

    let result = provider(&mock_server.uri())
        .text_step(&TextRequest::new("gpt-4o-mini"))
        .await;

    match result {
        Err(Error::Provider {
            status,
            kind,
            message,
            ..
        }) => {
            assert_eq!(status, 500);
            assert_eq!(kind.as_deref(), Some("server_error"));
            assert_eq!(message, "Internal server error");
        }
        other => panic!("expected Error::Provider, got: {other:?}"),
    }
}

#[tokio::test]
async fn an_error_body_that_isnt_json_still_produces_a_readable_message() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(502)
                .insert_header("content-type", "text/plain")
                .set_body_string("upstream connect error"),
        )
        .mount(&mock_server)
        .await;

    let result = provider(&mock_server.uri())
        .text_step(&TextRequest::new("gpt-4o-mini"))
        .await;

    match result {
        Err(Error::Provider { message, kind, .. }) => {
            assert_eq!(message, "upstream connect error");
            assert!(kind.is_none());
        }
        other => panic!("expected Error::Provider, got: {other:?}"),
    }
}

#[tokio::test]
async fn a_malformed_success_body_becomes_a_decode_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&mock_server)
        .await;

    let result = provider(&mock_server.uri())
        .text_step(&TextRequest::new("gpt-4o-mini"))
        .await;

    assert!(matches!(result, Err(Error::Decode { .. })));
}

#[tokio::test]
async fn a_well_formed_success_response_still_works_through_the_mock() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-test",
            "model": "gpt-4o-mini",
            "choices": [{
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        })))
        .mount(&mock_server)
        .await;

    let step = provider(&mock_server.uri())
        .text_step(&TextRequest::new("gpt-4o-mini"))
        .await
        .unwrap();

    assert_eq!(step.text.as_deref(), Some("pong"));
}
