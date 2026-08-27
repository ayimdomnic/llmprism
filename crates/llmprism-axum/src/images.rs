//! `POST /v1/images` -- image generation.
//!
//! Request body: [`ImagesRequestBody`]. Response: [`ImagesResponse`], one or
//! more generated images, each either a URL or base64-encoded data
//! depending on what the provider returns.
//!
//! ```json
//! { "provider": "openai", "model": "dall-e-3", "prompt": "a cat astronaut", "count": 1 }
//! ```

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use llmprism::images::{ImagesResponse, PendingImagesRequest};
use llmprism::tenancy::TenantRegistry;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;
use crate::tenant::TenantContext;

/// The JSON body for `POST /v1/images`.
#[derive(Deserialize)]
pub struct ImagesRequestBody {
    /// Name of the provider registered in the `Registry` this router was
    /// built from.
    pub provider: String,
    /// The model to target.
    pub model: String,
    /// A plain-language description of the image to generate.
    pub prompt: String,
    /// Requests `n` images instead of the provider's default.
    pub count: Option<u32>,
    /// Requests a specific size (e.g. `"1024x1024"`) instead of the
    /// provider's default. Valid values depend on the model.
    pub size: Option<String>,
    /// Requests a specific quality (e.g. `"hd"`) instead of the provider's
    /// default. Valid values depend on the model.
    pub quality: Option<String>,
    /// Requests a specific style (e.g. `"vivid"`) instead of the provider's
    /// default. Valid values depend on the model; most models have no
    /// equivalent concept at all.
    pub style: Option<String>,
}

fn build(registry: &Registry, body: ImagesRequestBody) -> Result<PendingImagesRequest, ApiError> {
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
    Ok(request)
}

pub(crate) async fn images(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<ImagesRequestBody>,
) -> Result<Json<ImagesResponse>, ApiError> {
    let response = build(&registry, body)?.generate().await?;
    Ok(Json(response))
}

pub(crate) async fn images_multi_tenant(
    State(tenants): State<Arc<dyn TenantRegistry>>,
    TenantContext(context): TenantContext,
    Json(body): Json<ImagesRequestBody>,
) -> Result<Json<ImagesResponse>, ApiError> {
    let registry = tenants.resolve(&context).await?;
    let response = build(&registry, body)?.generate().await?;
    Ok(Json(response))
}
