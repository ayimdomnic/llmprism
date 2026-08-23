//! Live smoke test against the real OpenAI API. Skipped unless `OPENAI_API_KEY` is
//! set, so it stays out of the way of offline/CI runs; run manually with a real key
//! to confirm wire compatibility. Requires `--features openai`.

#![cfg(feature = "openai")]

use futures::StreamExt;
use llmprism::providers::openai::OpenAiProvider;
use llmprism::stream_event::StreamEvent;
use llmprism::Registry;

#[tokio::test]
async fn live_text_generation_round_trip() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        eprintln!("skipping live_text_generation_round_trip: OPENAI_API_KEY not set");
        return;
    };

    let mut registry = Registry::new();
    registry.register("openai", OpenAiProvider::new(api_key));

    let response = registry
        .text("openai", "gpt-4o-mini")
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

#[tokio::test]
async fn live_stream_generation_round_trip() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        eprintln!("skipping live_stream_generation_round_trip: OPENAI_API_KEY not set");
        return;
    };

    let mut registry = Registry::new();
    registry.register("openai", OpenAiProvider::new(api_key));

    let mut stream = registry
        .text("openai", "gpt-4o-mini")
        .unwrap()
        .with_prompt("Reply with exactly the word: pong")
        .with_max_tokens(16)
        .stream();

    let mut text = String::new();
    let mut saw_stream_end = false;

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::TextDelta { text: delta } => text.push_str(&delta),
            StreamEvent::StreamEnd { response } => {
                saw_stream_end = true;
                assert_eq!(
                    response.text.as_deref().map(str::to_lowercase),
                    Some(text.to_lowercase())
                );
            }
            _ => {}
        }
    }

    assert!(saw_stream_end, "expected the stream to end with StreamEnd");
    assert!(
        text.to_lowercase().contains("pong"),
        "expected streamed text to contain 'pong', got: {text}"
    );
}
