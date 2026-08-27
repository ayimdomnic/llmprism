//! The simplest possible request: ask a model a question, print its answer.
//!
//! Run with:
//!
//! ```sh
//! OPENAI_API_KEY=sk-... cargo run --example text_generation --features openai
//! ```
//!
//! Swap `"openai"` for any other provider name you've registered (see
//! `Registry::from_env`'s docs for the full list of environment variables it
//! reads) and the rest of this example doesn't change -- that consistency
//! across providers is this crate's whole reason for existing.

#[tokio::main]
async fn main() {
    #[cfg(feature = "openai")]
    {
        let Ok(_) = std::env::var("OPENAI_API_KEY") else {
            eprintln!("skipping: set OPENAI_API_KEY to run this example against the real API");
            return;
        };

        let registry = llmprism::Registry::from_env();

        let response = registry
            .text("openai", "gpt-4o-mini")
            .expect("openai should be registered since OPENAI_API_KEY is set")
            .with_system_prompt("You are a concise assistant.")
            .with_prompt("What's the capital of France?")
            .with_max_tokens(100)
            .generate()
            .await
            .expect("request should succeed");

        println!("{}", response.text.unwrap_or_default());
    }

    #[cfg(not(feature = "openai"))]
    eprintln!("skipping: run with --features openai");
}
