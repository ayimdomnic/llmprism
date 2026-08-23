//! Live smoke test against the real VoyageAI API. Skipped unless
//! `VOYAGEAI_API_KEY` is set, so it stays out of the way of offline/CI runs;
//! run manually with a real key to confirm wire compatibility. Requires
//! `--features voyageai`.

#![cfg(feature = "voyageai")]

use llmprism::providers::voyageai::VoyageAiProvider;
use llmprism::Registry;

#[tokio::test]
async fn live_embeddings_round_trip() {
    let Ok(api_key) = std::env::var("VOYAGEAI_API_KEY") else {
        eprintln!("skipping live_embeddings_round_trip: VOYAGEAI_API_KEY not set");
        return;
    };

    let mut registry = Registry::new();
    registry.register("voyageai", VoyageAiProvider::new(api_key));

    let response = registry
        .embeddings("voyageai", "voyage-2")
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
