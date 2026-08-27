//! `POST /v1/text` and `POST /v1/text/stream` -- text generation, with tool
//! calling and multi-turn conversation handled the same way
//! [`PendingTextRequest`](llmprism::text::PendingTextRequest) already does.
//!
//! # `POST /v1/text`
//!
//! Request body: [`TextRequestBody`]. Response: [`TextResponse`], the same
//! type [`PendingTextRequest::generate`](llmprism::text::PendingTextRequest::generate)
//! returns, serialized as-is.
//!
//! ```json
//! {
//!   "provider": "openai",
//!   "model": "gpt-4o-mini",
//!   "system_prompts": ["You are a helpful assistant."],
//!   "messages": [
//!     { "role": "user", "content": [{ "Text": "Say hello in one word." }] }
//!   ],
//!   "max_tokens": 256,
//!   "temperature": 0.7
//! }
//! ```
//!
//! # `POST /v1/text/stream`
//!
//! Same request body as above. Response is `text/event-stream`: each event
//! is `event: message` with a JSON-encoded
//! [`StreamEvent`](llmprism::StreamEvent) as its data, ending with a
//! `StreamEvent::StreamEnd`. A failure partway through the model's reply
//! arrives as `event: error` with a plain-text message instead of breaking
//! the HTTP response -- see [`crate`] for why.

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use llmprism::tenancy::TenantRegistry;
use llmprism::text::TextResponse;
use llmprism::value_objects::Message;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;
use crate::sse::sse_stream;
use crate::tenant::TenantContext;

/// The JSON body for `POST /v1/text` and `POST /v1/text/stream`.
///
/// Mirrors the `.with_*()` calls on
/// [`PendingTextRequest`](llmprism::text::PendingTextRequest) -- everything
/// wire-safe is here; tools, approval handlers, and stop conditions aren't,
/// since those are Rust trait objects with no JSON representation (see
/// [`crate`]'s module docs).
#[derive(Deserialize)]
pub struct TextRequestBody {
    /// Name of the provider registered in the `Registry` this router was
    /// built from (e.g. `"openai"`, `"anthropic"`) -- not a route segment,
    /// mirroring the CLI's `--provider` flag.
    pub provider: String,
    /// The model to target, e.g. `"gpt-4o-mini"`.
    pub model: String,
    /// Zero or more system prompts, sent in order before `messages`.
    #[serde(default)]
    pub system_prompts: Vec<String>,
    /// The conversation so far. Reuses [`Message`] directly -- pass the same
    /// JSON shape you'd get back from a prior non-streaming response's
    /// history.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Caps the model's reply length, in tokens.
    pub max_tokens: Option<u32>,
    /// Sampling temperature; higher is more random.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Strings that stop generation early if the model produces them.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// A fixed seed for reproducible output, on providers that support it.
    pub seed: Option<u64>,
}

fn build(
    registry: &Registry,
    body: TextRequestBody,
) -> Result<llmprism::text::PendingTextRequest, ApiError> {
    let mut request = registry.text(&body.provider, body.model)?;
    for system_prompt in body.system_prompts {
        request = request.with_system_prompt(system_prompt);
    }
    request = request.with_messages(body.messages);
    if let Some(max_tokens) = body.max_tokens {
        request = request.with_max_tokens(max_tokens);
    }
    if let Some(temperature) = body.temperature {
        request = request.with_temperature(temperature);
    }
    if let Some(top_p) = body.top_p {
        request = request.with_top_p(top_p);
    }
    for stop in body.stop_sequences {
        request = request.with_stop_sequence(stop);
    }
    if let Some(seed) = body.seed {
        request = request.with_seed(seed);
    }
    Ok(request)
}

pub(crate) async fn text(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<TextRequestBody>,
) -> Result<Json<TextResponse>, ApiError> {
    let response = build(&registry, body)?.generate().await?;
    Ok(Json(response))
}

pub(crate) async fn text_stream(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<TextRequestBody>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    let stream = build(&registry, body)?.stream();
    Ok(Sse::new(sse_stream(stream)).keep_alive(KeepAlive::default()))
}

pub(crate) async fn text_multi_tenant(
    State(tenants): State<Arc<dyn TenantRegistry>>,
    TenantContext(context): TenantContext,
    Json(body): Json<TextRequestBody>,
) -> Result<Json<TextResponse>, ApiError> {
    let registry = tenants.resolve(&context).await?;
    let response = build(&registry, body)?.generate().await?;
    Ok(Json(response))
}

pub(crate) async fn text_stream_multi_tenant(
    State(tenants): State<Arc<dyn TenantRegistry>>,
    TenantContext(context): TenantContext,
    Json(body): Json<TextRequestBody>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    let registry = tenants.resolve(&context).await?;
    let stream = build(&registry, body)?.stream();
    Ok(Sse::new(sse_stream(stream)).keep_alive(KeepAlive::default()))
}
