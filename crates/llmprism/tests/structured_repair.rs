//! Integration tests for `RepairStrategy` (the structured-output decode-repair
//! hook) against a real (mocked) HTTP server -- confirms the full path from a
//! malformed provider response through `Error::StructuredDecode` to a
//! salvaged `StructuredResponse`, not just the trait in isolation. Needs no
//! API key.

#![cfg(feature = "openai")]

use async_trait::async_trait;
use llmprism::error::Error;
use llmprism::providers::openai::OpenAiProvider;
use llmprism::schema::{NumberSchema, ObjectSchema, Schema, StringSchema};
use llmprism::structured::RepairStrategy;
use llmprism::Registry;
use serde_json::Value;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn schema() -> ObjectSchema {
    ObjectSchema::new("recipe")
        .with_property(Schema::String(StringSchema::new("title")), true)
        .with_property(Schema::Number(NumberSchema::new("minutes")), true)
}

/// Some models wrap JSON in a Markdown code fence even when explicitly asked
/// not to; a repair strategy that strips it and reparses is a realistic,
/// minimal example of what this hook is for.
struct StripCodeFence;

#[async_trait]
impl RepairStrategy for StripCodeFence {
    async fn repair(&self, raw: &str, _error: &Error) -> Option<Value> {
        let stripped = raw
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```");
        serde_json::from_str(stripped.trim()).ok()
    }
}

async fn mount_chat_completion_with_content(mock_server: &MockServer, content: &str) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-test",
            "model": "gpt-4o-mini",
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        })))
        .mount(mock_server)
        .await;
}

#[tokio::test]
async fn with_repair_salvages_a_reply_wrapped_in_a_code_fence() {
    let mock_server = MockServer::start().await;
    mount_chat_completion_with_content(
        &mock_server,
        "```json\n{\"title\": \"Pasta\", \"minutes\": 10}\n```",
    )
    .await;

    let mut registry = Registry::new();
    registry.register(
        "openai",
        OpenAiProvider::with_base_url("sk-test", mock_server.uri()),
    );

    let response = registry
        .structured("openai", "gpt-4o-mini", schema())
        .unwrap()
        .with_prompt("A quick pasta recipe.")
        .with_repair(StripCodeFence)
        .generate()
        .await
        .expect("the repair strategy should salvage the fenced JSON");

    assert_eq!(response.data["title"], "Pasta");
    assert_eq!(response.data["minutes"], 10);
}

#[tokio::test]
async fn without_repair_a_code_fenced_reply_fails_with_structured_decode() {
    let mock_server = MockServer::start().await;
    mount_chat_completion_with_content(
        &mock_server,
        "```json\n{\"title\": \"Pasta\", \"minutes\": 10}\n```",
    )
    .await;

    let mut registry = Registry::new();
    registry.register(
        "openai",
        OpenAiProvider::with_base_url("sk-test", mock_server.uri()),
    );

    let result = registry
        .structured("openai", "gpt-4o-mini", schema())
        .unwrap()
        .with_prompt("A quick pasta recipe.")
        .generate()
        .await;

    assert!(matches!(result, Err(Error::StructuredDecode { .. })));
}

#[tokio::test]
async fn a_repair_that_gives_up_still_returns_the_original_error() {
    struct NeverRepairs;

    #[async_trait]
    impl RepairStrategy for NeverRepairs {
        async fn repair(&self, _raw: &str, _error: &Error) -> Option<Value> {
            None
        }
    }

    let mock_server = MockServer::start().await;
    mount_chat_completion_with_content(&mock_server, "not json at all").await;

    let mut registry = Registry::new();
    registry.register(
        "openai",
        OpenAiProvider::with_base_url("sk-test", mock_server.uri()),
    );

    let result = registry
        .structured("openai", "gpt-4o-mini", schema())
        .unwrap()
        .with_prompt("A quick pasta recipe.")
        .with_repair(NeverRepairs)
        .generate()
        .await;

    match result {
        Err(Error::StructuredDecode { provider, .. }) => assert_eq!(provider, "openai"),
        other => panic!("expected Error::StructuredDecode, got {other:?}"),
    }
}
