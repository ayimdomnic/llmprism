//! Live smoke test against the real OpenAI API. Skipped unless `OPENAI_API_KEY` is
//! set, so it stays out of the way of offline/CI runs; run manually with a real key
//! to confirm wire compatibility. Requires `--features openai`.

#![cfg(feature = "openai")]

use futures::StreamExt;
use llmprism::providers::openai::OpenAiProvider;
use llmprism::schema::{NumberSchema, ObjectSchema, Schema, StringSchema};
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

#[tokio::test]
async fn live_structured_generation_round_trip() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        eprintln!("skipping live_structured_generation_round_trip: OPENAI_API_KEY not set");
        return;
    };

    let mut registry = Registry::new();
    registry.register("openai", OpenAiProvider::new(api_key));

    let schema = ObjectSchema::new("pong_response")
        .with_property(
            Schema::String(StringSchema::new("word").with_description("Always the word 'pong'")),
            true,
        )
        .with_property(
            Schema::Number(NumberSchema::new("length").with_description("Length of that word")),
            true,
        );

    let response = registry
        .structured("openai", "gpt-4o-mini", schema)
        .unwrap()
        .with_prompt("Reply with the word 'pong' and its character length.")
        .generate()
        .await
        .unwrap();

    assert_eq!(
        response.data["word"].as_str().map(str::to_lowercase),
        Some("pong".to_string())
    );
    assert_eq!(response.data["length"].as_u64(), Some(4));
}

#[tokio::test]
async fn live_moderation_round_trip() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        eprintln!("skipping live_moderation_round_trip: OPENAI_API_KEY not set");
        return;
    };

    let mut registry = Registry::new();
    registry.register("openai", OpenAiProvider::new(api_key));

    let response = registry
        .moderation("openai", "omni-moderation-latest")
        .unwrap()
        .with_input("The quick brown fox jumps over the lazy dog.")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.results.len(), 1);
    assert!(
        !response.results[0].flagged,
        "expected an innocuous sentence not to be flagged"
    );
}

#[tokio::test]
async fn live_embeddings_round_trip() {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        eprintln!("skipping live_embeddings_round_trip: OPENAI_API_KEY not set");
        return;
    };

    let mut registry = Registry::new();
    registry.register("openai", OpenAiProvider::new(api_key));

    let response = registry
        .embeddings("openai", "text-embedding-3-small")
        .unwrap()
        .with_input("The quick brown fox.")
        .with_input("jumps over the lazy dog.")
        .generate()
        .await
        .unwrap();

    assert_eq!(response.embeddings.len(), 2);
    assert!(!response.embeddings[0].vector.is_empty());
    assert!(!response.embeddings[1].vector.is_empty());
    assert_eq!(
        response.embeddings[0].vector.len(),
        response.embeddings[1].vector.len()
    );
}
