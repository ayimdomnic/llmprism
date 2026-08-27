//! Exercises `POST /v1/audio/speech` and `POST /v1/audio/transcriptions`
//! against `FakeProvider`, driven through `tower::ServiceExt::oneshot` --
//! no real network access.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use llmprism::testing::{FakeAudioResponse, FakeProvider, FakeTranscriptionResponse};
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

#[tokio::test]
async fn speech_round_trip_returns_base64_encoded_audio() {
    let fake = FakeProvider::new("fake")
        .respond_with_audio(FakeAudioResponse::new(vec![1, 2, 3, 4], "audio/mpeg"));
    let mut registry = Registry::new();
    registry.register("fake", fake);
    let app = llmprism_axum::routes(registry);

    let (status, body) = post(
        app,
        "/v1/audio/speech",
        json!({
            "provider": "fake",
            "model": "tts-1",
            "input": "Hello, world!",
            "voice": "alloy",
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["audio"]["data"], BASE64.encode([1, 2, 3, 4]));
    assert_eq!(body["audio"]["mime_type"], "audio/mpeg");
}

#[tokio::test]
async fn transcriptions_round_trip_decodes_base64_and_returns_the_scripted_text() {
    let fake = Arc::new(
        FakeProvider::new("fake")
            .respond_with_transcription(FakeTranscriptionResponse::new("hello there")),
    );
    let mut registry = Registry::new();
    registry.register_arc("fake", fake.clone());
    let app = llmprism_axum::routes(registry);

    let (status, body) = post(
        app,
        "/v1/audio/transcriptions",
        json!({
            "provider": "fake",
            "model": "whisper-1",
            "audio": {
                "data": BASE64.encode([9, 9, 9]),
                "mime_type": "audio/mpeg",
                "filename": "clip.mp3",
            },
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["text"], "hello there");

    let recorded = fake.recorded_speech_to_text_requests();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].audio.data, vec![9, 9, 9]);
    assert_eq!(recorded[0].audio.filename, "clip.mp3");
}

#[tokio::test]
async fn invalid_base64_in_a_transcription_request_maps_to_400() {
    let fake = FakeProvider::new("fake")
        .respond_with_transcription(FakeTranscriptionResponse::new("unreachable"));
    let mut registry = Registry::new();
    registry.register("fake", fake);
    let app = llmprism_axum::routes(registry);

    let (status, _body) = post(
        app,
        "/v1/audio/transcriptions",
        json!({
            "provider": "fake",
            "model": "whisper-1",
            "audio": {
                "data": "not valid base64!!!",
                "mime_type": "audio/mpeg",
            },
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
}
