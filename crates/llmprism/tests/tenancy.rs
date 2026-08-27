//! Exercises `TenantRegistry`/`StaticTenantRegistry`/`UsageTrackingMiddleware`
//! against `FakeProvider` -- no network access, no feature flags required.

use llmprism::tenancy::{
    InMemoryUsageSink, RequestContext, StaticTenantRegistry, TenantRegistry,
    UsageTrackingMiddleware,
};
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::value_objects::Usage;
use llmprism::Registry;

#[tokio::test]
async fn each_tenant_resolves_to_its_own_registry() {
    let mut acme = Registry::new();
    acme.register(
        "openai",
        FakeProvider::new("openai").respond_with(FakeTextResponse::new("Hi, Acme.")),
    );

    let mut globex = Registry::new();
    globex.register(
        "openai",
        FakeProvider::new("openai").respond_with(FakeTextResponse::new("Hi, Globex.")),
    );

    let tenants = StaticTenantRegistry::new()
        .with_tenant("acme", acme)
        .with_tenant("globex", globex);

    let acme_registry = tenants.resolve(&RequestContext::new("acme")).await.unwrap();
    let acme_response = acme_registry
        .text("openai", "test-model")
        .unwrap()
        .with_prompt("hi")
        .generate()
        .await
        .unwrap();
    assert_eq!(acme_response.text.as_deref(), Some("Hi, Acme."));

    let globex_registry = tenants
        .resolve(&RequestContext::new("globex"))
        .await
        .unwrap();
    let globex_response = globex_registry
        .text("openai", "test-model")
        .unwrap()
        .with_prompt("hi")
        .generate()
        .await
        .unwrap();
    assert_eq!(globex_response.text.as_deref(), Some("Hi, Globex."));
}

#[tokio::test]
async fn an_unknown_tenant_fails_to_resolve() {
    let tenants = StaticTenantRegistry::new().with_tenant("acme", Registry::new());

    let result = tenants
        .resolve(&RequestContext::new("no-such-tenant"))
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn usage_tracking_middleware_records_usage_against_the_tenant_it_was_built_for() {
    let fake =
        FakeProvider::new("openai").respond_with(FakeTextResponse::new("hi").with_usage(Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            ..Default::default()
        }));

    let mut registry = Registry::new();
    registry.register("openai", fake);

    let sink = std::sync::Arc::new(InMemoryUsageSink::new());
    registry
        .wrap(
            "openai",
            UsageTrackingMiddleware::new("acme", std::sync::Arc::clone(&sink)),
        )
        .unwrap();

    registry
        .text("openai", "test-model")
        .unwrap()
        .with_prompt("hi")
        .generate()
        .await
        .unwrap();

    let recorded = sink.usage_for("acme");
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].prompt_tokens, 10);
    assert_eq!(recorded[0].completion_tokens, 5);
    assert!(sink.usage_for("globex").is_empty());
}
