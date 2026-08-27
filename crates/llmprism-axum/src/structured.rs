//! `POST /v1/structured` and `POST /v1/structured/stream` -- ask the model
//! for a reply matching a JSON Schema you provide, instead of free-form
//! text.
//!
//! # `POST /v1/structured`
//!
//! Request body: [`StructuredRequestBody`]. `schema` is a raw JSON Schema
//! document, passed straight through to
//! [`ObjectSchema::from_raw_json_schema`] -- the same escape hatch the CLI's
//! `structured --schema-file` flag already uses, so anything you'd put in
//! that file works here too. Response: [`StructuredResponse`], the model's
//! reply already parsed as JSON matching your schema.
//!
//! ```json
//! {
//!   "provider": "openai",
//!   "model": "gpt-4o-mini",
//!   "schema_name": "recipe",
//!   "schema": {
//!     "type": "object",
//!     "properties": { "title": { "type": "string" } },
//!     "required": ["title"]
//!   },
//!   "messages": [
//!     { "role": "user", "content": [{ "Text": "A pasta recipe" }] }
//!   ]
//! }
//! ```
//!
//! # `POST /v1/structured/stream`
//!
//! Same request body as above. Response is `text/event-stream`: each
//! `event: message` carries a JSON-encoded
//! [`StructuredStreamEvent`](llmprism::structured::StructuredStreamEvent) --
//! a best-effort partial parse of the reply so far, then one final `End`
//! with the complete result. A mid-stream failure arrives as `event: error`
//! instead of breaking the response -- see [`crate`] for why. No
//! [`RepairStrategy`](llmprism::structured::RepairStrategy) is applied here,
//! unlike the non-streaming route -- a malformed final reply surfaces as an
//! error on the stream itself.

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use llmprism::schema::ObjectSchema;
use llmprism::structured::StructuredResponse;
use llmprism::value_objects::Message;
use llmprism::Registry;
use serde::Deserialize;
use serde_json::Value;

use crate::error::ApiError;
use crate::sse::sse_stream;

/// The JSON body for `POST /v1/structured` and `POST /v1/structured/stream`.
#[derive(Deserialize)]
pub struct StructuredRequestBody {
    /// Name of the provider registered in the `Registry` this router was
    /// built from.
    pub provider: String,
    /// The model to target.
    pub model: String,
    /// A name for the schema (Anthropic uses this as the synthetic tool
    /// name it forces the model to call; OpenAI's native structured-output
    /// mode uses it as the schema's `name` field).
    pub schema_name: String,
    /// The JSON Schema document the reply must match, passed through to
    /// [`ObjectSchema::from_raw_json_schema`] as-is.
    pub schema: Value,
    /// Zero or more system prompts, sent in order before `messages`.
    #[serde(default)]
    pub system_prompts: Vec<String>,
    /// The conversation so far.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Caps the model's reply length, in tokens.
    pub max_tokens: Option<u32>,
    /// Sampling temperature; higher is more random.
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// A fixed seed for reproducible output, on providers that support it.
    pub seed: Option<u64>,
}

fn build(
    registry: &Registry,
    body: StructuredRequestBody,
) -> Result<llmprism::structured::PendingStructuredRequest, ApiError> {
    let schema = ObjectSchema::from_raw_json_schema(body.schema_name, body.schema);
    let mut request = registry.structured(&body.provider, body.model, schema)?;
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
    if let Some(seed) = body.seed {
        request = request.with_seed(seed);
    }
    Ok(request)
}

pub(crate) async fn structured(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<StructuredRequestBody>,
) -> Result<Json<StructuredResponse>, ApiError> {
    let response = build(&registry, body)?.generate().await?;
    Ok(Json(response))
}

pub(crate) async fn structured_stream(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<StructuredRequestBody>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    let stream = build(&registry, body)?.stream();
    Ok(Sse::new(sse_stream(stream)).keep_alive(KeepAlive::default()))
}
