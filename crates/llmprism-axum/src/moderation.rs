//! `POST /v1/moderation`.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use llmprism::moderation::ModerationResponse;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;

#[derive(Deserialize)]
pub(crate) struct ModerationRequestBody {
    provider: String,
    model: String,
    input: String,
}

pub(crate) async fn moderation(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<ModerationRequestBody>,
) -> Result<Json<ModerationResponse>, ApiError> {
    let response = registry
        .moderation(&body.provider, body.model)?
        .with_input(body.input)
        .generate()
        .await?;
    Ok(Json(response))
}
