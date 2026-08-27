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
//! Every route takes a `provider` field in its JSON body naming which
//! registered provider to use, mirroring the `--provider`/`--model` pairing
//! the `llmprism` CLI already uses -- there's no provider segment in the
//! route paths themselves. See each capability's module for its exact
//! request body shape.
//!
//! Routes: `POST /v1/text`, `POST /v1/text/stream` (SSE), `POST
//! /v1/structured`, `POST /v1/structured/stream` (SSE), `POST
//! /v1/moderation`, `POST /v1/embeddings`, `POST /v1/rerank`, `POST
//! /v1/images`.
//!
//! Tool calling, approval handling, and MCP wiring aren't exposed over HTTP
//! here -- a [`llmprism::Tool`] is arbitrary server-side code, which has no
//! JSON wire representation, so that stays something the server configures
//! directly against [`Registry`]/`PendingTextRequest`, not something a
//! client can send. Audio endpoints aren't included yet either; see
//! `ROADMAP.md` in the workspace root.

mod embeddings;
mod error;
mod images;
mod moderation;
mod rerank;
mod sse;
mod structured;
mod text;

use std::sync::Arc;

use axum::routing::post;
use axum::Router;
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
