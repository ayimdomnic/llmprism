//! Benchmarks `stream_loop::stream_text`'s own bookkeeping cost -- the
//! streaming counterpart to `tool_loop`'s benchmark, for the same reason:
//! `Provider::stream_text_once` is called once per round trip against a
//! conversation that keeps growing, so it's worth knowing this loop's
//! overhead in isolation from real network/SSE latency.

use async_trait::async_trait;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use futures::StreamExt;
use llmprism::error::ToolError;
use llmprism::schema::ObjectSchema;
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::tool::Tool;
use llmprism::value_objects::ToolOutput;
use llmprism::Registry;
use serde_json::{json, Value};

struct Echo {
    parameters: ObjectSchema,
}

impl Echo {
    fn new() -> Self {
        Self {
            parameters: ObjectSchema::new("parameters"),
        }
    }
}

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes its input straight back"
    }

    fn parameters(&self) -> &ObjectSchema {
        &self.parameters
    }

    async fn call(&self, args: Value) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::from(args.to_string()))
    }
}

/// Mirrors `tool_loop.rs`'s `registry_long_conversation` -- `tool_call_steps`
/// round trips, each with a `response_text`-sized assistant reply, before a
/// final plain-text answer.
fn registry_long_conversation(tool_call_steps: u32, response_text: &str) -> Registry {
    let mut fake = FakeProvider::new("fake");
    for step in 0..tool_call_steps {
        fake = fake.respond_with(FakeTextResponse::new(response_text).with_tool_call(
            format!("call_{step}"),
            "echo",
            json!({"n": step}),
        ));
    }
    fake = fake.respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", fake);
    registry
}

fn bench_stream_loop(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let long_response = "x".repeat(2_000);

    c.bench_function(
        "stream_loop/thirty_sequential_tool_calls_long_conversation",
        |b| {
            b.to_async(&runtime).iter_batched(
                || registry_long_conversation(30, &long_response),
                |registry| async move {
                    let mut stream = registry
                        .text("fake", "test-model")
                        .unwrap()
                        .with_prompt("hi")
                        .with_tool(Echo::new())
                        .with_max_steps(31)
                        .stream();

                    while let Some(event) = stream.next().await {
                        event.unwrap();
                    }
                },
                BatchSize::SmallInput,
            )
        },
    );
}

criterion_group!(benches, bench_stream_loop);
criterion_main!(benches);
