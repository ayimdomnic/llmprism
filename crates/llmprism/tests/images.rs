//! Exercises image-generation requests against `FakeProvider` -- no network
//! access, no feature flags required.

use llmprism::testing::{FakeImagesResponse, FakeProvider};
use llmprism::value_objects::MediaData;
use llmprism::Registry;

#[tokio::test]
async fn images_request_returns_the_canned_image() {
    let provider = FakeProvider::new("fake")
        .respond_with_images(FakeImagesResponse::new().with_url("https://example.com/cat.png"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .images("fake", "test-model", "a cat")
        .unwrap()
        .with_count(1)
        .with_size("1024x1024")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.images.len(), 1);
    match &response.images[0].data {
        MediaData::Url(url) => assert_eq!(url, "https://example.com/cat.png"),
        MediaData::Base64(_) => panic!("expected a URL, got base64 data"),
    }
}

#[tokio::test]
#[should_panic(expected = "no more canned images responses queued")]
async fn images_request_panics_with_no_canned_response() {
    let provider = FakeProvider::new("fake");

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let _ = registry
        .images("fake", "test-model", "a cat")
        .unwrap()
        .generate()
        .await;
}
