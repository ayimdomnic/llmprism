//! Exercises structured-output requests against `FakeProvider` -- no network
//! access, no feature flags required.

use std::sync::Arc;

use futures::StreamExt;
use llmprism::schema::{BooleanSchema, ObjectSchema, Schema, StringSchema};
use llmprism::structured::StructuredStreamEvent;
use llmprism::testing::{FakeProvider, FakeStructuredResponse};
use llmprism::value_objects::FinishReason;
use llmprism::Registry;
use serde_json::json;

fn review_schema() -> ObjectSchema {
    ObjectSchema::new("review")
        .with_property(Schema::String(StringSchema::new("summary")), true)
        .with_property(Schema::Boolean(BooleanSchema::new("recommended")), true)
}

#[tokio::test]
async fn structured_request_returns_the_canned_data() {
    let provider = FakeProvider::new("fake").respond_with_structured(FakeStructuredResponse::new(
        json!({"summary": "Great crate.", "recommended": true}),
    ));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .structured("fake", "test-model", review_schema())
        .unwrap()
        .with_prompt("Review this crate.")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.data["summary"], "Great crate.");
    assert_eq!(response.data["recommended"], true);
    assert_eq!(response.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn structured_request_records_what_was_sent() {
    // Keep our own `Arc` handle (via `register_arc`) so we can inspect the
    // provider's recorded requests after the registry has used it.
    let provider = Arc::new(FakeProvider::new("fake").respond_with_structured(
        FakeStructuredResponse::new(json!({"summary": "ok", "recommended": false})),
    ));

    let mut registry = Registry::new();
    registry.register_arc("fake", provider.clone());

    registry
        .structured("fake", "test-model", review_schema())
        .unwrap()
        .with_system_prompt("Be concise.")
        .with_prompt("Review this crate.")
        .generate()
        .await
        .unwrap();

    let recorded = provider.recorded_structured_requests();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].model, "test-model");
    assert_eq!(recorded[0].schema.name, "review");
    assert_eq!(recorded[0].system_prompts, vec!["Be concise.".to_string()]);
}

#[tokio::test]
async fn structured_request_streams_a_partial_object_then_ends() {
    let provider = FakeProvider::new("fake").respond_with_structured(FakeStructuredResponse::new(
        json!({"summary": "Great crate.", "recommended": true}),
    ));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let mut stream = registry
        .structured("fake", "test-model", review_schema())
        .unwrap()
        .with_prompt("Review this crate.")
        .stream();

    let mut saw_partial = false;
    let mut end_response = None;

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StructuredStreamEvent::PartialObject { data } => {
                assert_eq!(data["summary"], "Great crate.");
                saw_partial = true;
            }
            StructuredStreamEvent::End { response } => end_response = Some(response),
        }
    }

    assert!(saw_partial, "expected at least one PartialObject event");
    let response = end_response.expect("stream should end with a StructuredStreamEvent::End");
    assert_eq!(response.data["recommended"], true);
}

#[tokio::test]
#[should_panic(expected = "no more canned structured responses queued")]
async fn structured_request_panics_with_no_canned_response() {
    let provider = FakeProvider::new("fake");

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let _ = registry
        .structured("fake", "test-model", review_schema())
        .unwrap()
        .with_prompt("Review this crate.")
        .generate()
        .await;
}
