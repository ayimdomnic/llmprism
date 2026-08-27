//! Live smoke test against a real local Ollama server. Unlike every other
//! provider's live test (gated on an API key env var), this one is gated on
//! `OLLAMA_LIVE_TEST_MODEL` -- naming the model to test against -- since
//! Ollama needs no key but does need an actual running server with at least
//! one model pulled, which we can't assume CI (or a fresh checkout) has.
//! Skipped unless that's set, so it stays out of the way of offline/CI runs;
//! run manually (with Ollama running locally and a model pulled) to confirm
//! wire compatibility. Requires `--features ollama`.

#![cfg(feature = "ollama")]

use llmprism::providers::ollama::OllamaProvider;
use llmprism::{Provider, Registry};

#[test]
fn reports_its_own_name_not_openai() {
    assert_eq!(OllamaProvider::new().name(), "ollama");
}

#[test]
fn registers_and_resolves_the_normal_way() {
    let mut registry = Registry::new();
    registry.register(
        "ollama",
        OllamaProvider::with_base_url("http://localhost:11434/v1"),
    );

    assert!(registry.provider("ollama").is_ok());
}

#[tokio::test]
async fn live_text_generation_round_trip() {
    let Ok(model) = std::env::var("OLLAMA_LIVE_TEST_MODEL") else {
        eprintln!(
            "skipping live_text_generation_round_trip: OLLAMA_LIVE_TEST_MODEL not set \
             (set it to a model you've pulled locally, e.g. \"llama3.2\", with Ollama running)"
        );
        return;
    };

    let mut registry = Registry::new();
    registry.register("ollama", OllamaProvider::new());

    let response = registry
        .text("ollama", model)
        .unwrap()
        .with_prompt("Reply with exactly the word: pong")
        .with_max_tokens(16)
        .generate()
        .await
        .unwrap();

    let text = response.text.unwrap_or_default().to_lowercase();
    assert!(
        text.contains("pong"),
        "expected response to contain 'pong', got: {text}"
    );
}
