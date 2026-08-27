//! `POST /v1/moderation` -- check text against a provider's content-safety
//! classifier.
//!
//! Request body: [`ModerationRequestBody`]. Response: [`ModerationResponse`].
//! Not every provider has a moderation endpoint (Anthropic and Gemini don't)
//! -- calling this against one that doesn't returns
//! [`Error::Unsupported`](llmprism::Error::Unsupported), mapped to `501 Not
//! Implemented` by [`ApiError`].
//!
//! ```json
//! { "provider": "openai", "model": "omni-moderation-latest", "input": "some text" }
//! ```

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use llmprism::moderation::ModerationResponse;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;

/// The JSON body for `POST /v1/moderation`.
#[derive(Deserialize)]
pub struct ModerationRequestBody {
    /// Name of the provider registered in the `Registry` this router was
    /// built from.
    pub provider: String,
    /// The model to target.
    pub model: String,
    /// The text to classify.
    pub input: String,
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
