//! Benchmarks the per-call overhead `Registry::wrap` adds -- each layer
//! wraps the provider in one more `dyn Provider` dispatch (see
//! `middleware.rs`'s `MiddlewareProvider`), so it's worth knowing how that
//! scales as an application stacks logging, retries, caching, and similar
//! middleware on top of a provider.

use async_trait::async_trait;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use llmprism::middleware::ProviderMiddleware;
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::Registry;

/// A middleware that does nothing beyond what `ProviderMiddleware`'s default
/// methods already do (call straight through to `next`) -- isolates the
/// dispatch overhead of one more layer from the cost of whatever that layer
/// actually does.
struct Noop;

#[async_trait]
impl ProviderMiddleware for Noop {}

fn registry_wrapped(layers: usize) -> Registry {
    let mut registry = Registry::new();
    registry.register(
        "fake",
        FakeProvider::new("fake").respond_with(FakeTextResponse::new("hi")),
    );
    for _ in 0..layers {
        registry.wrap("fake", Noop).unwrap();
    }
    registry
}

fn bench_middleware(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    for layers in [0usize, 1, 10] {
        c.bench_function(&format!("middleware/{layers}_layers"), |b| {
            b.to_async(&runtime).iter_batched(
                || registry_wrapped(layers),
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
    }
}

criterion_group!(benches, bench_middleware);
criterion_main!(benches);
