//! Getting a reply from Claude guaranteed to match a JSON shape you
//! describe, instead of free-form text you'd have to parse yourself.
//!
//! Anthropic has no native structured-output mode the way OpenAI does, so
//! this crate gets there differently under the hood: it forces Claude to
//! call a single synthetic tool shaped exactly like the requested schema
//! (see `providers::anthropic`'s module docs). You don't need to know that
//! to use it -- the API and the `StructuredResponse` you get back are
//! identical to `structured_output.rs`'s OpenAI version.
//!
//! Run with:
//!
//! ```sh
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example anthropic_structured_output --features anthropic
//! ```

#[tokio::main]
async fn main() {
    #[cfg(feature = "anthropic")]
    {
        use llmprism::schema::{ArraySchema, ObjectSchema, Schema, StringSchema};

        let Ok(_) = std::env::var("ANTHROPIC_API_KEY") else {
            eprintln!("skipping: set ANTHROPIC_API_KEY to run this example against the real API");
            return;
        };

        let registry = llmprism::Registry::from_env();

        let schema = ObjectSchema::new("recipe")
            .with_property(Schema::String(StringSchema::new("title")), true)
            .with_property(
                Schema::Array(
                    ArraySchema::new("ingredients", Schema::String(StringSchema::new("item")))
                        .with_description("Each ingredient, one per entry"),
                ),
                true,
            );

        let response = registry
            .structured("anthropic", "claude-3-5-haiku-20241022", schema)
            .expect("anthropic should be registered since ANTHROPIC_API_KEY is set")
            .with_prompt("A simple recipe for scrambled eggs.")
            .generate()
            .await
            .expect("request should succeed");

        // `response.data` is a `serde_json::Value` -- deserialize it into
        // your own type with `serde_json::from_value` if you'd rather work
        // with a typed struct than raw JSON.
        println!("{}", serde_json::to_string_pretty(&response.data).unwrap());
    }

    #[cfg(not(feature = "anthropic"))]
    eprintln!("skipping: run with --features anthropic");
}
