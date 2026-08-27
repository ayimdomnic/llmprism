//! Streaming a reply incrementally instead of waiting for the whole thing.
//!
//! Run with:
//!
//! ```sh
//! OPENAI_API_KEY=sk-... cargo run --example streaming --features openai
//! ```

#[tokio::main]
async fn main() {
    #[cfg(feature = "openai")]
    {
        use futures::StreamExt;
        use llmprism::StreamEvent;

        let Ok(_) = std::env::var("OPENAI_API_KEY") else {
            eprintln!("skipping: set OPENAI_API_KEY to run this example against the real API");
            return;
        };

        let registry = llmprism::Registry::from_env();

        let mut stream = registry
            .text("openai", "gpt-4o-mini")
            .expect("openai should be registered since OPENAI_API_KEY is set")
            .with_prompt("Count from one to five, one number per line.")
            .stream();

        while let Some(event) = stream.next().await {
            if let StreamEvent::TextDelta { text } = event.expect("stream should not error") {
                print!("{text}");
            }
        }
        println!();
    }

    #[cfg(not(feature = "openai"))]
    eprintln!("skipping: run with --features openai");
}
