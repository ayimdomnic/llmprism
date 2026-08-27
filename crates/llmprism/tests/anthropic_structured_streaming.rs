//! Integration test for `PendingStructuredRequest::stream` against a mocked
//! SSE server (via `wiremock`, no real API key needed) -- confirms the full
//! path end to end: Anthropic's `stream_structured_once` reading the forced
//! tool call's `input_json_delta` events, repairing them into partial JSON
//! via `partial-json-fixer` after every chunk, and finishing with the fully
//! parsed final object.

#![cfg(feature = "anthropic")]

use futures::StreamExt;
use llmprism::providers::anthropic::AnthropicProvider;
use llmprism::schema::{NumberSchema, ObjectSchema, Schema, StringSchema};
use llmprism::structured::StructuredStreamEvent;
use llmprism::Registry;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A forced tool call's `input_json_delta` events, split mid-string and
/// mid-object on purpose so every intermediate chunk is genuinely incomplete
/// JSON, concatenating to `{"title": "Pasta", "minutes": 15}`.
const SSE_BODY: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-haiku-20241022\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"recipe\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"title\\\": \\\"Pas\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"ta\\\", \\\"minutes\\\": 1\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"5}\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":8}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[tokio::test]
async fn streams_increasingly_complete_partial_objects_then_ends() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(SSE_BODY, "text/event-stream"))
        .mount(&mock_server)
        .await;

    let mut registry = Registry::new();
    registry.register(
        "anthropic",
        AnthropicProvider::with_base_url("sk-ant-test", mock_server.uri()),
    );

    let schema = ObjectSchema::new("recipe")
        .with_property(Schema::String(StringSchema::new("title")), true)
        .with_property(Schema::Number(NumberSchema::new("minutes")), false);

    let mut stream = registry
        .structured("anthropic", "claude-3-5-haiku-20241022", schema)
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
    // "tool_use" is the *success* stop reason for a forced call -- reported
    // as a normal Stop, not ToolCalls, matching the non-streaming path.
    assert_eq!(
        response.finish_reason,
        llmprism::value_objects::FinishReason::Stop
    );
}

#[tokio::test]
async fn a_reply_with_no_tool_use_block_reports_the_text_as_the_error_context() {
    let mock_server = MockServer::start().await;

    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-haiku-20241022\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"I can't help with that.\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":6}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream"))
        .mount(&mock_server)
        .await;

    let mut registry = Registry::new();
    registry.register(
        "anthropic",
        AnthropicProvider::with_base_url("sk-ant-test", mock_server.uri()),
    );

    let schema =
        ObjectSchema::new("recipe").with_property(Schema::String(StringSchema::new("title")), true);

    let mut stream = registry
        .structured("anthropic", "claude-3-5-haiku-20241022", schema)
        .unwrap()
        .with_prompt("A quick pasta recipe.")
        .stream();

    let mut last_error = None;
    while let Some(event) = stream.next().await {
        if let Err(err) = event {
            last_error = Some(err);
        }
    }

    match last_error.expect("stream should end with an error") {
        llmprism::Error::StructuredDecode { context, .. } => {
            assert_eq!(context.raw, "I can't help with that.");
        }
        other => panic!("expected Error::StructuredDecode, got {other:?}"),
    }
}
