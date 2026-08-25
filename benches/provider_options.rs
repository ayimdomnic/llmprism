//! Benchmarks `client::merge_provider_options` -- the `provider_options`
//! escape hatch's underlying mechanism, run on every request that sets it
//! (and, cheaply, even when it doesn't). Requires the `http` feature, since
//! that's what gates `client` itself.

#![cfg(feature = "http")]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use llmprism::client::merge_provider_options;
use serde::Serialize;
use serde_json::json;

#[derive(Serialize)]
struct WireRequest {
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    messages: Vec<WireMessage>,
}

#[derive(Serialize)]
struct WireMessage {
    role: String,
    content: String,
}

fn sample_request() -> WireRequest {
    WireRequest {
        model: "gpt-4o-mini".to_string(),
        temperature: Some(0.7),
        max_tokens: Some(1024),
        messages: (0..10)
            .map(|i| WireMessage {
                role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: format!("This is message number {i} in the conversation."),
            })
            .collect(),
    }
}

fn bench_merge_provider_options(c: &mut Criterion) {
    let request = sample_request();
    let no_overrides = serde_json::Value::Null;
    let overrides = json!({"temperature": 0.2, "seed": 42, "top_p": 0.9});

    c.bench_function("merge_provider_options/no_overrides", |b| {
        b.iter(|| merge_provider_options(black_box(&request), black_box(&no_overrides)))
    });

    c.bench_function("merge_provider_options/with_overrides", |b| {
        b.iter(|| merge_provider_options(black_box(&request), black_box(&overrides)))
    });
}

criterion_group!(benches, bench_merge_provider_options);
criterion_main!(benches);
