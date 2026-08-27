//! `POST /v1/structured` and `POST /v1/structured/stream`.

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

#[derive(Deserialize)]
pub(crate) struct StructuredRequestBody {
    provider: String,
    model: String,
    schema_name: String,
    schema: Value,
    #[serde(default)]
    system_prompts: Vec<String>,
    #[serde(default)]
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
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
