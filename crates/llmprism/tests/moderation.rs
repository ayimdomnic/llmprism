//! Exercises moderation requests against `FakeProvider` -- no network access,
//! no feature flags required.

use llmprism::testing::{FakeModerationResponse, FakeProvider};
use llmprism::Registry;

#[tokio::test]
async fn moderation_request_returns_the_canned_result() {
    let provider = FakeProvider::new("fake")
        .respond_with_moderation(FakeModerationResponse::new().flagged(true));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .moderation("fake", "test-model")
        .unwrap()
        .with_input("some text")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.results.len(), 1);
    assert!(response.results[0].flagged);
}

#[tokio::test]
#[should_panic(expected = "no more canned moderation responses queued")]
async fn moderation_request_panics_with_no_canned_response() {
    let provider = FakeProvider::new("fake");

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let _ = registry
        .moderation("fake", "test-model")
        .unwrap()
        .with_input("some text")
        .generate()
        .await;
}
