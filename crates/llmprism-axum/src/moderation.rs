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
use llmprism::moderation::{ModerationResponse, PendingModerationRequest};
use llmprism::tenancy::TenantRegistry;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;
use crate::tenant::TenantContext;

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

fn build(
    registry: &Registry,
    body: ModerationRequestBody,
) -> Result<PendingModerationRequest, ApiError> {
    Ok(registry
        .moderation(&body.provider, body.model)?
        .with_input(body.input))
}

pub(crate) async fn moderation(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<ModerationRequestBody>,
) -> Result<Json<ModerationResponse>, ApiError> {
    let response = build(&registry, body)?.generate().await?;
    Ok(Json(response))
}

pub(crate) async fn moderation_multi_tenant(
    State(tenants): State<Arc<dyn TenantRegistry>>,
    TenantContext(context): TenantContext,
    Json(body): Json<ModerationRequestBody>,
) -> Result<Json<ModerationResponse>, ApiError> {
    let registry = tenants.resolve(&context).await?;
    let response = build(&registry, body)?.generate().await?;
    Ok(Json(response))
}
