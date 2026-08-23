//! Exercises embeddings requests against `FakeProvider` -- no network access,
//! no feature flags required.

use llmprism::testing::{FakeEmbeddingsResponse, FakeProvider};
use llmprism::Registry;

#[tokio::test]
async fn embeddings_request_returns_the_canned_vectors() {
    let provider = FakeProvider::new("fake").respond_with_embeddings(
        FakeEmbeddingsResponse::new()
            .with_embedding(vec![0.1, 0.2, 0.3])
            .with_embedding(vec![0.4, 0.5, 0.6]),
    );

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .embeddings("fake", "test-model")
        .unwrap()
        .with_input("first")
        .with_input("second")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.embeddings[0].vector, vec![0.1, 0.2, 0.3]);
    assert_eq!(response.embeddings[1].vector, vec![0.4, 0.5, 0.6]);
}

#[tokio::test]
#[should_panic(expected = "no more canned embeddings responses queued")]
async fn embeddings_request_panics_with_no_canned_response() {
    let provider = FakeProvider::new("fake");

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let _ = registry
        .embeddings("fake", "test-model")
        .unwrap()
        .with_input("some text")
        .generate()
        .await;
}
