//! `POST /v1/text` and `POST /v1/text/stream`.

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use llmprism::text::TextResponse;
use llmprism::value_objects::Message;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;
use crate::sse::sse_stream;

#[derive(Deserialize)]
pub(crate) struct TextRequestBody {
    provider: String,
    model: String,
    #[serde(default)]
    system_prompts: Vec<String>,
    #[serde(default)]
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    seed: Option<u64>,
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
