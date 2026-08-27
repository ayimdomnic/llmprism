use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use llmprism::testing::{
    FakeEmbeddingsResponse, FakeImagesResponse, FakeModerationResponse, FakeProvider,
    FakeRerankResponse, FakeStructuredResponse, FakeTextResponse,
};
use llmprism::Registry;
use serde_json::{json, Value};
use tower::ServiceExt;

async fn post(app: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    (status, json)
}

async fn post_raw(app: axum::Router, path: &str, body: Value) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

#[tokio::test]
async fn text_round_trip_returns_the_scripted_reply() {
    let fake = FakeProvider::new("fake").respond_with(FakeTextResponse::new("hi there"));
    let mut registry = Registry::new();
    registry.register("fake", fake);
    let app = llmprism_axum::routes(registry);

    let (status, body) = post(
        app,
        "/v1/text",
        json!({
            "provider": "fake",
            "model": "test-model",
            "messages": [{"role": "user", "content": [{"Text": "hello"}]}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["text"], "hi there");
}

#[tokio::test]
async fn text_stream_emits_message_events_and_ends() {
    let fake = FakeProvider::new("fake").respond_with(FakeTextResponse::new("hi there"));
    let mut registry = Registry::new();
    registry.register("fake", fake);
    let app = llmprism_axum::routes(registry);

    let (status, body) = post_raw(
        app,
        "/v1/text/stream",
        json!({
            "provider": "fake",
            "model": "test-model",
            "messages": [{"role": "user", "content": [{"Text": "hello"}]}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("event: message"));
    assert!(body.contains("stream_start"));
    assert!(body.contains("stream_end"));
    assert!(!body.contains("event: error"));
}

#[tokio::test]
async fn structured_round_trip_returns_the_scripted_data() {
    let fake = FakeProvider::new("fake")
        .respond_with_structured(FakeStructuredResponse::new(json!({"greeting": "hi"})));
    let mut registry = Registry::new();
    registry.register("fake", fake);
    let app = llmprism_axum::routes(registry);

    let (status, body) = post(
        app,
        "/v1/structured",
        json!({
            "provider": "fake",
            "model": "test-model",
            "schema_name": "greeting",
            "schema": {"type": "object", "properties": {"greeting": {"type": "string"}}},
            "messages": [{"role": "user", "content": [{"Text": "hello"}]}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["greeting"], "hi");
}

#[tokio::test]
async fn moderation_round_trip_returns_the_scripted_result() {
    let fake = FakeProvider::new("fake")
        .respond_with_moderation(FakeModerationResponse::new().flagged(true));
    let mut registry = Registry::new();
    registry.register("fake", fake);
    let app = llmprism_axum::routes(registry);

    let (status, body) = post(
        app,
        "/v1/moderation",
        json!({"provider": "fake", "model": "test-model", "input": "some text"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["results"][0]["flagged"], true);
}

#[tokio::test]
async fn embeddings_round_trip_returns_the_scripted_vector() {
    let fake = FakeProvider::new("fake")
        .respond_with_embeddings(FakeEmbeddingsResponse::new().with_embedding(vec![0.1, 0.2, 0.3]));
    let mut registry = Registry::new();
    registry.register("fake", fake);
    let app = llmprism_axum::routes(registry);

    let (status, body) = post(
        app,
        "/v1/embeddings",
        json!({"provider": "fake", "model": "test-model", "input": ["hello"]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["embeddings"][0]["vector"], json!([0.1, 0.2, 0.3]));
}

#[tokio::test]
async fn rerank_round_trip_returns_the_scripted_scores() {
    let fake = FakeProvider::new("fake")
        .respond_with_rerank(FakeRerankResponse::new().with_result(0, 0.92));
    let mut registry = Registry::new();
    registry.register("fake", fake);
    let app = llmprism_axum::routes(registry);

    let (status, body) = post(
        app,
        "/v1/rerank",
        json!({
            "provider": "fake",
            "model": "test-model",
            "query": "a query",
            "documents": ["doc one"],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["results"][0]["relevance_score"], 0.92);
}

#[tokio::test]
async fn images_round_trip_returns_the_scripted_image() {
    let fake = FakeProvider::new("fake")
        .respond_with_images(FakeImagesResponse::new().with_url("https://example.com/cat.png"));
    let mut registry = Registry::new();
    registry.register("fake", fake);
    let app = llmprism_axum::routes(registry);

    let (status, body) = post(
        app,
        "/v1/images",
        json!({"provider": "fake", "model": "test-model", "prompt": "a cat"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["images"][0]["data"], "https://example.com/cat.png");
}

#[tokio::test]
async fn an_unknown_provider_maps_to_404() {
    let registry = Registry::new();
    let app = llmprism_axum::routes(registry);

    let (status, body) = post(
        app,
        "/v1/text",
        json!({
            "provider": "does-not-exist",
            "model": "test-model",
            "messages": [{"role": "user", "content": [{"Text": "hello"}]}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("does-not-exist"));
}
