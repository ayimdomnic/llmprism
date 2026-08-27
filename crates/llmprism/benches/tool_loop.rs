//! Benchmarks `tool_loop::run_text`'s own bookkeeping cost -- message
//! cloning/appending, tool dispatch (sequential and concurrent), and request
//! cloning across round trips -- in isolation from network latency, by
//! running it against [`FakeProvider`]. This is the part of the crate whose
//! cost scales with how many tool-calling round trips an application drives,
//! so it's worth knowing its overhead independent of whichever real provider
//! is on the other end.

use async_trait::async_trait;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use llmprism::error::ToolError;
use llmprism::schema::ObjectSchema;
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::tool::Tool;
use llmprism::value_objects::ToolOutput;
use llmprism::Registry;
use serde_json::{json, Value};

struct Echo {
    parameters: ObjectSchema,
    concurrent: bool,
}

impl Echo {
    fn new(concurrent: bool) -> Self {
        Self {
            parameters: ObjectSchema::new("parameters"),
            concurrent,
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

    fn concurrent(&self) -> bool {
        self.concurrent
    }

    async fn call(&self, args: Value) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::from(args.to_string()))
    }
}

/// A fresh registry with a `FakeProvider` scripted to run `tool_call_steps`
/// tool-calling round trips (one tool call each) before a final plain-text
/// answer -- `tool_call_steps + 1` canned [`llmprism::text::Step`]s in total.
fn registry_scripted_for(tool_call_steps: u32, calls_per_step: u32) -> Registry {
    let mut fake = FakeProvider::new("fake");
    for step in 0..tool_call_steps {
        let mut response = FakeTextResponse::new("");
        for call in 0..calls_per_step {
            response =
                response.with_tool_call(format!("call_{step}_{call}"), "echo", json!({"n": call}));
        }
        fake = fake.respond_with(response);
    }
    fake = fake.respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", fake);
    registry
}

/// Like `registry_scripted_for`, but each round trip's assistant text is
/// `response_text` rather than empty, so the conversation's total byte size
/// actually grows step over step -- the scenario that distinguishes an O(n)
/// per-round-trip cost (`Provider::text_step` borrowing the request) from an
/// O(n) *per-round-trip clone of the whole accumulated conversation* (what it
/// cost before that method took `&TextRequest` instead of `TextRequest`).
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

fn bench_tool_loop(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let long_response = "x".repeat(2_000);

    c.bench_function("tool_loop/single_round_trip_no_tools", |b| {
        b.to_async(&runtime).iter_batched(
            || registry_scripted_for(0, 0),
            |registry| async move {
                registry
                    .text("fake", "test-model")
                    .unwrap()
                    .with_prompt("hi")
                    .generate()
                    .await
                    .unwrap()
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("tool_loop/five_sequential_tool_calls", |b| {
        b.to_async(&runtime).iter_batched(
            || registry_scripted_for(5, 1),
            |registry| async move {
                registry
                    .text("fake", "test-model")
                    .unwrap()
                    .with_prompt("hi")
                    .with_tool(Echo::new(false))
                    .with_max_steps(6)
                    .generate()
                    .await
                    .unwrap()
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("tool_loop/one_round_five_concurrent_tool_calls", |b| {
        b.to_async(&runtime).iter_batched(
            || registry_scripted_for(1, 5),
            |registry| async move {
                registry
                    .text("fake", "test-model")
                    .unwrap()
                    .with_prompt("hi")
                    .with_tool(Echo::new(true))
                    .with_max_steps(2)
                    .generate()
                    .await
                    .unwrap()
            },
            BatchSize::SmallInput,
        )
    });

    // A longer, more realistic conversation -- 30 tool-calling round trips,
    // each adding a ~2KB assistant reply plus its tool result to the
    // conversation. `Provider::text_step` only ever borrows the request now
    // (see its doc comment), so this no longer clones the whole,
    // ever-larger conversation on every round trip the way it used to.
    c.bench_function(
        "tool_loop/thirty_sequential_tool_calls_long_conversation",
        |b| {
            b.to_async(&runtime).iter_batched(
                || registry_long_conversation(30, &long_response),
                |registry| async move {
                    registry
                        .text("fake", "test-model")
                        .unwrap()
                        .with_prompt("hi")
                        .with_tool(Echo::new(false))
                        .with_max_steps(31)
                        .generate()
                        .await
                        .unwrap()
                },
                BatchSize::SmallInput,
            )
        },
    );
}

criterion_group!(benches, bench_tool_loop);
criterion_main!(benches);
