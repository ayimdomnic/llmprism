//! Integration test for `PendingStructuredRequest::stream` against a mocked
//! SSE server (via `wiremock`, no real API key needed) -- confirms the full
//! path end to end: OpenAI's `stream_structured_once` accumulating raw
//! content deltas, repairing them into partial JSON via `partial-json-fixer`
//! after every chunk, and finishing with the fully-parsed final object.

#![cfg(feature = "openai")]

use futures::StreamExt;
use llmprism::providers::openai::OpenAiProvider;
use llmprism::schema::{NumberSchema, ObjectSchema, Schema, StringSchema};
use llmprism::structured::StructuredStreamEvent;
use llmprism::Registry;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Three SSE chunks that, concatenated, spell out
/// `{"title": "Pasta", "minutes": 15}` -- split mid-string and mid-object on
/// purpose, so every intermediate chunk is genuinely incomplete JSON.
const SSE_BODY: &str = concat!(
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"{\\\"title\\\": \\\"Pas\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"ta\\\", \\\"minutes\\\": 1\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"5}\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":8}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn streams_increasingly_complete_partial_objects_then_ends() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(SSE_BODY, "text/event-stream"))
        .mount(&mock_server)
        .await;

    let mut registry = Registry::new();
    registry.register(
        "openai",
        OpenAiProvider::with_base_url("sk-test", mock_server.uri()),
    );

    let schema = ObjectSchema::new("recipe")
        .with_property(Schema::String(StringSchema::new("title")), true)
        .with_property(Schema::Number(NumberSchema::new("minutes")), false);

    let mut stream = registry
        .structured("openai", "gpt-4o-mini", schema)
        .unwrap()
        .with_prompt("A quick pasta recipe.")
        .stream();

    let mut partials = Vec::new();
    let mut end_response = None;

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StructuredStreamEvent::PartialObject { data } => partials.push(data),
            StructuredStreamEvent::End { response } => end_response = Some(response),
        }
    }

    assert!(
        !partials.is_empty(),
        "expected at least one partial object before the stream ended"
    );
    // The very first partial (after just `{"title": "Pas`) should already
    // show a (truncated) title -- proves the repair-and-parse happens on
    // genuinely incomplete JSON, not just once the object closes.
    assert!(partials[0]["title"].as_str().unwrap().starts_with("Pas"));

    let response = end_response.expect("stream should end with a StructuredStreamEvent::End");
    assert_eq!(response.data["title"], "Pasta");
    assert_eq!(response.data["minutes"], 15);
    assert_eq!(response.usage.prompt_tokens, 10);
    assert_eq!(response.usage.completion_tokens, 8);
}
