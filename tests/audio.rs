//! Exercises text-to-speech and speech-to-text requests against
//! `FakeProvider` -- no network access, no feature flags required.

use llmprism::audio::AudioInput;
use llmprism::testing::{FakeAudioResponse, FakeProvider, FakeTranscriptionResponse};
use llmprism::Registry;

#[tokio::test]
async fn text_to_speech_returns_the_canned_audio() {
    let provider = FakeProvider::new("fake")
        .respond_with_audio(FakeAudioResponse::new(vec![1, 2, 3, 4], "audio/mpeg"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .text_to_speech("fake", "test-model", "Hello, world!")
        .unwrap()
        .with_voice("alloy")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.audio.data, vec![1, 2, 3, 4]);
    assert_eq!(response.audio.mime_type, "audio/mpeg");
}

#[tokio::test]
#[should_panic(expected = "no more canned text_to_speech responses queued")]
async fn text_to_speech_panics_with_no_canned_response() {
    let provider = FakeProvider::new("fake");

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let _ = registry
        .text_to_speech("fake", "test-model", "Hello, world!")
        .unwrap()
        .generate()
        .await;
}

#[tokio::test]
async fn speech_to_text_returns_the_canned_transcription() {
    let provider = FakeProvider::new("fake")
        .respond_with_transcription(FakeTranscriptionResponse::new("hello there"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let audio = AudioInput::new(vec![1, 2, 3, 4], "audio/mpeg").with_filename("recording.mp3");

    let response = registry
        .speech_to_text("fake", "test-model", audio)
        .unwrap()
        .generate()
        .await
        .unwrap();

    assert_eq!(response.text, "hello there");
}

#[tokio::test]
#[should_panic(expected = "no more canned speech_to_text responses queued")]
async fn speech_to_text_panics_with_no_canned_response() {
    let provider = FakeProvider::new("fake");

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let audio = AudioInput::new(vec![1, 2, 3, 4], "audio/mpeg");
    let _ = registry
        .speech_to_text("fake", "test-model", audio)
        .unwrap()
        .generate()
        .await;
}
