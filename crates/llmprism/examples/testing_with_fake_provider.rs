//! Testing code that uses `llmprism` without calling a real provider or
//! needing an API key -- register a `FakeProvider` with a scripted response
//! under the same name your application code already uses.
//!
//! This one needs no API key and no Cargo feature flags, so it's the one
//! example in this directory you can just run as-is:
//!
//! ```sh
//! cargo run --example testing_with_fake_provider
//! ```

use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::Registry;

#[tokio::main]
async fn main() {
    let fake = FakeProvider::new("openai").respond_with(FakeTextResponse::new("Bonjour le monde!"));

    let mut registry = Registry::new();
    registry.register("openai", fake);

    // Everything past this point is exactly what your application code would
    // do against the real `openai` provider -- it has no way to tell the
    // difference.
    let response = registry
        .text("openai", "gpt-4o-mini")
        .unwrap()
        .with_prompt("Say hello in French.")
        .generate()
        .await
        .unwrap();

    println!("{}", response.text.as_deref().unwrap());
    assert_eq!(response.text.as_deref(), Some("Bonjour le monde!"));
}
