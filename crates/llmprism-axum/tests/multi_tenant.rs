//! Exercises `routes_multi_tenant`/`TenantContext` against `FakeProvider`,
//! driven through `tower::ServiceExt::oneshot` -- no real network access.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use llmprism::tenancy::{RequestContext, StaticTenantRegistry};
use llmprism::testing::{FakeAudioResponse, FakeProvider, FakeTextResponse};
use llmprism::Registry;
use serde_json::{json, Value};
use tower::ServiceExt;

fn tenants() -> StaticTenantRegistry {
    let mut acme = Registry::new();
    acme.register(
        "fake",
        FakeProvider::new("fake").respond_with(FakeTextResponse::new("hi, acme")),
    );

    let mut globex = Registry::new();
    globex.register(
        "fake",
        FakeProvider::new("fake").respond_with(FakeTextResponse::new("hi, globex")),
    );

    StaticTenantRegistry::new()
        .with_tenant("acme", acme)
        .with_tenant("globex", globex)
}

fn request_body() -> Value {
    json!({
        "provider": "fake",
        "model": "test-model",
        "messages": [{"role": "user", "content": [{"Text": "hello"}]}],
    })
}

/// Builds a request carrying `context` in its extensions -- standing in
/// for what an application's own auth middleware would insert before
/// these routes ever run.
fn request_with_context(path: &str, context: RequestContext) -> Request<Body> {
    let mut request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&request_body()).unwrap()))
        .unwrap();
    request.extensions_mut().insert(context);
    request
}

#[tokio::test]
async fn a_request_reaches_the_right_tenants_provider() {
    let app = llmprism_axum::routes_multi_tenant(tenants());

    let response = app
        .oneshot(request_with_context(
            "/v1/text",
            RequestContext::new("acme"),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["text"], "hi, acme");
}

#[tokio::test]
async fn a_different_tenant_reaches_a_different_provider() {
    let app = llmprism_axum::routes_multi_tenant(tenants());

    let response = app
        .oneshot(request_with_context(
            "/v1/text",
            RequestContext::new("globex"),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["text"], "hi, globex");
}

#[tokio::test]
async fn a_request_with_no_tenant_context_is_rejected() {
    let app = llmprism_axum::routes_multi_tenant(tenants());

    let request = Request::builder()
        .method("POST")
        .uri("/v1/text")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&request_body()).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn an_unknown_tenant_maps_to_a_backend_error() {
    let app = llmprism_axum::routes_multi_tenant(tenants());

    let response = app
        .oneshot(request_with_context(
            "/v1/text",
            RequestContext::new("no-such-tenant"),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

/// `audio`'s multi-tenant handlers resolve their `Registry` the exact same
/// way every other capability's do -- one focused test confirming that
/// wiring is right is enough; `tests/audio.rs` already covers the
/// base64/error-mapping behavior itself against the single-tenant routes.
#[tokio::test]
async fn audio_routes_also_resolve_the_right_tenant() {
    let mut acme = Registry::new();
    acme.register(
        "fake",
        FakeProvider::new("fake")
            .respond_with_audio(FakeAudioResponse::new(vec![1, 2, 3], "audio/mpeg")),
    );
    let tenants = StaticTenantRegistry::new().with_tenant("acme", acme);
    let app = llmprism_axum::routes_multi_tenant(tenants);

    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/audio/speech")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "provider": "fake",
                "model": "tts-1",
                "input": "hi",
            }))
            .unwrap(),
        ))
        .unwrap();
    request.extensions_mut().insert(RequestContext::new("acme"));

    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
