//! Streaming a reply from Anthropic incrementally instead of waiting for the
//! whole thing.
//!
//! Run with:
//!
//! ```sh
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_streaming --features anthropic
//! ```

#[tokio::main]
async fn main() {
    #[cfg(feature = "anthropic")]
    {
        use futures::StreamExt;
        use llmprism::StreamEvent;

        let Ok(_) = std::env::var("ANTHROPIC_API_KEY") else {
            eprintln!("skipping: set ANTHROPIC_API_KEY to run this example against the real API");
            return;
        };

        let registry = llmprism::Registry::from_env();

        let mut stream = registry
            .text("anthropic", "claude-3-5-haiku-20241022")
            .expect("anthropic should be registered since ANTHROPIC_API_KEY is set")
            .with_prompt("Count from one to five, one number per line.")
            .stream();

        while let Some(event) = stream.next().await {
            if let StreamEvent::TextDelta { text } = event.expect("stream should not error") {
                print!("{text}");
            }
        }
        println!();
    }

    #[cfg(not(feature = "anthropic"))]
    eprintln!("skipping: run with --features anthropic");
}
