//! Axum routes for [`llmprism`] -- mount every capability as an HTTP API with
//! one line:
//!
//! ```no_run
//! use llmprism::Registry;
//!
//! # async fn example() {
//! // Registered providers depend on which `llmprism` provider features
//! // your own Cargo.toml enables -- see `Registry::register`.
//! let registry = Registry::new();
//! let app = llmprism_axum::routes(registry);
//! let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
//! axum::serve(listener, app).await.unwrap();
//! # }
//! ```
//!
//! `routes` builds a self-contained [`Router`]; merge it into a larger
//! application with [`Router::merge`], layer your own `tower` middleware
//! (auth, rate limiting, tracing) on top the normal way, or serve it as-is.
//!
//! # Routes
//!
//! Every route is `POST`, takes a `provider` field in its JSON body naming
//! which provider registered in `Registry` to use (mirroring the
//! `--provider`/`--model` pairing the `llmprism` CLI already uses), and
//! returns the same response type the matching `Registry`/`PendingXRequest`
//! method would. See each module below for the exact request body shape and
//! a worked example.
//!
//! | Route | Module | Non-streaming response |
//! |---|---|---|
//! | `POST /v1/text` | [`text`] | [`llmprism::text::TextResponse`] |
//! | `POST /v1/text/stream` (SSE) | [`text`] | -- |
//! | `POST /v1/structured` | [`structured`] | [`llmprism::structured::StructuredResponse`] |
//! | `POST /v1/structured/stream` (SSE) | [`structured`] | -- |
//! | `POST /v1/moderation` | [`moderation`] | [`llmprism::moderation::ModerationResponse`] |
//! | `POST /v1/embeddings` | [`embeddings`] | [`llmprism::embeddings::EmbeddingsResponse`] |
//! | `POST /v1/rerank` | [`rerank`] | [`llmprism::rerank::RerankResponse`] |
//! | `POST /v1/images` | [`images`] | [`llmprism::images::ImagesResponse`] |
//!
//! # Errors
//!
//! Every route maps a failed `llmprism::Error` to an HTTP status and a
//! `{"error": {"message": ...}}` body -- see [`error::ApiError`] for the
//! exact mapping. A streaming route can't change HTTP status after the
//! response has started, so a mid-stream failure there becomes an SSE
//! `event: error` frame with the same message text instead of a failed
//! response.
//!
//! # What's deliberately out of scope
//!
//! - **Tool calling, approval handling, and MCP** aren't exposed over HTTP:
//!   a [`llmprism::Tool`] is arbitrary server-side code with a `call()`
//!   method, which has no JSON wire representation. Attach tools directly
//!   to a `Registry`/`PendingTextRequest` on your own server before handing
//!   requests to code that uses this crate, rather than expecting a client
//!   to send one.
//! - **Audio** (`/v1/audio/speech`, `/v1/audio/transcriptions`) isn't
//!   implemented yet -- binary request/response bodies need a deliberate
//!   design choice (base64 vs. multipart vs. raw bytes) that didn't fit
//!   this crate's first pass. See `ROADMAP.md` in the workspace root for
//!   what's tracked as a follow-up.
//!
//! # Multi-tenancy
//!
//! [`routes`] serves one fixed [`Registry`] -- the common case. An
//! application serving several tenants (different API keys, different
//! allowed providers, different default models per tenant) from the same
//! process uses [`routes_multi_tenant`] instead: it resolves the
//! `Registry` to use *per request*, from a
//! [`RequestContext`](llmprism::tenancy::RequestContext) your own auth
//! middleware attaches. This crate never verifies identity itself -- see
//! [`tenant::TenantContext`] for exactly what it expects your middleware
//! to have already done.
//!
//! # Testing your own code against this
//!
//! Register a [`llmprism::testing::FakeProvider`] into the `Registry` you
//! pass to [`routes`] and drive it with `tower::ServiceExt::oneshot` --
//! no real network socket needed. See this crate's own
//! `tests/routes.rs` for worked examples covering every route, including
//! the SSE ones, and `tests/multi_tenant.rs` for [`routes_multi_tenant`].

pub mod embeddings;
pub mod error;
pub mod images;
pub mod moderation;
pub mod rerank;
mod sse;
pub mod structured;
pub mod tenant;
pub mod text;

use std::sync::Arc;

use axum::routing::post;
use axum::Router;
use llmprism::tenancy::TenantRegistry;
use llmprism::Registry;

/// Builds a [`Router`] exposing every `llmprism` capability `registry` has
/// providers registered for, as a set of `POST` JSON endpoints. Merge it
/// into your own router with [`Router::merge`], or serve it directly.
pub fn routes(registry: Registry) -> Router {
    Router::new()
        .route("/v1/text", post(text::text))
        .route("/v1/text/stream", post(text::text_stream))
        .route("/v1/structured", post(structured::structured))
        .route("/v1/structured/stream", post(structured::structured_stream))
        .route("/v1/moderation", post(moderation::moderation))
        .route("/v1/embeddings", post(embeddings::embeddings))
        .route("/v1/rerank", post(rerank::rerank))
        .route("/v1/images", post(images::images))
        .with_state(Arc::new(registry))
}

/// Builds a [`Router`] with the same routes as [`routes`], but resolving a
/// per-request [`Registry`] from `tenant_registry` instead of serving one
/// fixed `Registry` -- see the [module docs](self#multi-tenancy).
///
/// Every route requires a [`tenant::TenantContext`] to already be
/// extractable from the request (i.e. your own auth middleware must run
/// before these routes and insert a
/// [`RequestContext`](llmprism::tenancy::RequestContext) into the
/// request's extensions) -- a request with none rejects with `401`. A
/// tenant `tenant_registry.resolve` doesn't recognize (the
/// `Error::Store` [`llmprism::tenancy::StaticTenantRegistry`] returns for
/// one) maps through [`error::ApiError`] to `502`, the same as any other
/// backend failure that isn't one of `Error`'s more specific variants --
/// deliberately not `404`, since `Error::Store` also covers a
/// `ConversationStore` backend genuinely failing, which is a server
/// problem, not a "this resource doesn't exist" one.
pub fn routes_multi_tenant(tenant_registry: impl TenantRegistry + 'static) -> Router {
    Router::new()
        .route("/v1/text", post(text::text_multi_tenant))
        .route("/v1/text/stream", post(text::text_stream_multi_tenant))
        .route("/v1/structured", post(structured::structured_multi_tenant))
        .route(
            "/v1/structured/stream",
            post(structured::structured_stream_multi_tenant),
        )
        .route("/v1/moderation", post(moderation::moderation_multi_tenant))
        .route("/v1/embeddings", post(embeddings::embeddings_multi_tenant))
        .route("/v1/rerank", post(rerank::rerank_multi_tenant))
        .route("/v1/images", post(images::images_multi_tenant))
        .with_state(Arc::new(tenant_registry) as Arc<dyn TenantRegistry>)
}
