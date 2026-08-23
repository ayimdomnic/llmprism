//! Giving a model a tool it can call, and letting `llmprism` handle the whole
//! back-and-forth automatically -- run the tool, send the result back,
//! continue the conversation.
//!
//! Run with:
//!
//! ```sh
//! OPENAI_API_KEY=sk-... cargo run --example tool_calling --features openai
//! ```

#[cfg(feature = "openai")]
mod weather_tool {
    use async_trait::async_trait;
    use llmprism::error::ToolError;
    use llmprism::schema::{ObjectSchema, Schema, StringSchema};
    use llmprism::tool::Tool;
    use llmprism::value_objects::ToolOutput;
    use serde_json::Value;

    /// A fake weather lookup -- stands in for a tool that would call a real
    /// API, hit a database, or run any other side effect you want the model
    /// to be able to trigger.
    pub struct GetWeather {
        parameters: ObjectSchema,
    }

    impl GetWeather {
        pub fn new() -> Self {
            Self {
                parameters: ObjectSchema::new("parameters").with_property(
                    Schema::String(StringSchema::new("city").with_description("City name")),
                    true,
                ),
            }
        }
    }

    impl Default for GetWeather {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl Tool for GetWeather {
        fn name(&self) -> &str {
            "get_weather"
        }

        fn description(&self) -> &str {
            "Looks up the current weather for a city"
        }

        fn parameters(&self) -> &ObjectSchema {
            &self.parameters
        }

        async fn call(&self, args: Value) -> Result<ToolOutput, ToolError> {
            let city = args
                .get("city")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Ok(ToolOutput::from(format!("It's sunny and 22C in {city}.")))
        }
    }
}

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
            .with_prompt("What's the weather like in Nairobi right now?")
            .with_tool(weather_tool::GetWeather::new())
            // Room for at least one tool call plus a follow-up reply that
            // uses its result.
            .with_max_steps(4)
            .generate()
            .await
            .expect("request should succeed");

        println!("{}", response.text.unwrap_or_default());
        println!("(took {} round trip(s))", response.steps.len());
    }

    #[cfg(not(feature = "openai"))]
    eprintln!("skipping: run with --features openai");
}
