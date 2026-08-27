//! `POST /v1/embeddings`.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use llmprism::embeddings::EmbeddingsResponse;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;

#[derive(Deserialize)]
pub(crate) struct EmbeddingsRequestBody {
    provider: String,
    model: String,
    input: Vec<String>,
    dimensions: Option<u32>,
}

pub(crate) async fn embeddings(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<EmbeddingsRequestBody>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    let mut request = registry.embeddings(&body.provider, body.model)?;
    for input in body.input {
        request = request.with_input(input);
    }
    if let Some(dimensions) = body.dimensions {
        request = request.with_dimensions(dimensions);
    }
    let response = request.generate().await?;
    Ok(Json(response))
}
