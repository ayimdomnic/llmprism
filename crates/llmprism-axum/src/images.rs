//! `POST /v1/images`.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use llmprism::images::ImagesResponse;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;

#[derive(Deserialize)]
pub(crate) struct ImagesRequestBody {
    provider: String,
    model: String,
    prompt: String,
    count: Option<u32>,
    size: Option<String>,
    quality: Option<String>,
    style: Option<String>,
}

pub(crate) async fn images(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<ImagesRequestBody>,
) -> Result<Json<ImagesResponse>, ApiError> {
    let mut request = registry.images(&body.provider, body.model, body.prompt)?;
    if let Some(count) = body.count {
        request = request.with_count(count);
    }
    if let Some(size) = body.size {
        request = request.with_size(size);
    }
    if let Some(quality) = body.quality {
        request = request.with_quality(quality);
    }
    if let Some(style) = body.style {
        request = request.with_style(style);
    }
    let response = request.generate().await?;
    Ok(Json(response))
}
