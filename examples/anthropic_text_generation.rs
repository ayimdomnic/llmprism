//! The simplest possible request against Anthropic: ask a model a question,
//! print its answer.
//!
//! Run with:
//!
//! ```sh
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_text_generation --features anthropic
//! ```
//!
//! Everything below is identical to `text_generation.rs` except the provider
//! name and model -- that consistency across providers is this crate's whole
//! reason for existing.

#[tokio::main]
async fn main() {
    #[cfg(feature = "anthropic")]
    {
        let Ok(_) = std::env::var("ANTHROPIC_API_KEY") else {
            eprintln!("skipping: set ANTHROPIC_API_KEY to run this example against the real API");
            return;
        };

        let registry = llmprism::Registry::from_env();

        let response = registry
            .text("anthropic", "claude-3-5-haiku-20241022")
            .expect("anthropic should be registered since ANTHROPIC_API_KEY is set")
            .with_system_prompt("You are a concise assistant.")
            .with_prompt("What's the capital of France?")
            .with_max_tokens(100)
            .generate()
            .await
            .expect("request should succeed");

        println!("{}", response.text.unwrap_or_default());
    }

    #[cfg(not(feature = "anthropic"))]
    eprintln!("skipping: run with --features anthropic");
}
