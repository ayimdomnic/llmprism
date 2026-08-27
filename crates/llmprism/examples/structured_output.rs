//! Getting a reply guaranteed to match a JSON shape you describe, instead of
//! free-form text you'd have to parse yourself.
//!
//! Run with:
//!
//! ```sh
//! OPENAI_API_KEY=sk-... cargo run --example structured_output --features openai
//! ```

#[tokio::main]
async fn main() {
    #[cfg(feature = "openai")]
    {
        use llmprism::schema::{ArraySchema, ObjectSchema, Schema, StringSchema};

        let Ok(_) = std::env::var("OPENAI_API_KEY") else {
            eprintln!("skipping: set OPENAI_API_KEY to run this example against the real API");
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
            .structured("openai", "gpt-4o-mini", schema)
            .expect("openai should be registered since OPENAI_API_KEY is set")
            .with_prompt("A simple recipe for scrambled eggs.")
            .generate()
            .await
            .expect("request should succeed");

        // `response.data` is a `serde_json::Value` -- deserialize it into
        // your own type with `serde_json::from_value` if you'd rather work
        // with a typed struct than raw JSON.
        println!("{}", serde_json::to_string_pretty(&response.data).unwrap());
    }

    #[cfg(not(feature = "openai"))]
    eprintln!("skipping: run with --features openai");
}
