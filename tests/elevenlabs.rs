//! Live smoke test against the real ElevenLabs API. Skipped unless
//! `ELEVENLABS_API_KEY` is set, so it stays out of the way of offline/CI
//! runs; run manually with a real key to confirm wire compatibility.
//! Requires `--features elevenlabs`.

#![cfg(feature = "elevenlabs")]

use llmprism::audio::AudioInput;
use llmprism::providers::elevenlabs::ElevenLabsProvider;
use llmprism::Registry;

#[tokio::test]
async fn live_audio_round_trip() {
    let Ok(api_key) = std::env::var("ELEVENLABS_API_KEY") else {
        eprintln!("skipping live_audio_round_trip: ELEVENLABS_API_KEY not set");
        return;
    };

    let mut registry = Registry::new();
    registry.register("elevenlabs", ElevenLabsProvider::new(api_key));

    // Speak a sentence with the default voice, then transcribe the audio
    // right back -- exercises both endpoints and confirms they're actually
    // compatible with each other, not just individually well-formed.
    let speech = registry
        .text_to_speech(
            "elevenlabs",
            "eleven_flash_v2_5",
            "The quick brown fox jumps over the lazy dog.",
        )
        .unwrap()
        .generate()
        .await
        .unwrap();

    assert!(!speech.audio.data.is_empty());

    let audio =
        AudioInput::new(speech.audio.data, speech.audio.mime_type).with_filename("speech.mp3");
    let transcription = registry
        .speech_to_text("elevenlabs", "scribe_v2", audio)
        .unwrap()
        .generate()
        .await
        .unwrap();

    assert!(
        transcription.text.to_lowercase().contains("fox"),
        "expected transcription to mention 'fox', got: {}",
        transcription.text
    );
}
