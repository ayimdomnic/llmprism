//! `POST /v1/embeddings` -- turn text into numeric vectors for similarity
//! search, clustering, or retrieval.
//!
//! Request body: [`EmbeddingsRequestBody`]. Response: [`EmbeddingsResponse`],
//! one embedding per input, in order.
//!
//! ```json
//! { "provider": "openai", "model": "text-embedding-3-small", "input": ["hello", "world"] }
//! ```

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use llmprism::embeddings::{EmbeddingsResponse, PendingEmbeddingsRequest};
use llmprism::tenancy::TenantRegistry;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;
use crate::tenant::TenantContext;

/// The JSON body for `POST /v1/embeddings`.
#[derive(Deserialize)]
pub struct EmbeddingsRequestBody {
    /// Name of the provider registered in the `Registry` this router was
    /// built from.
    pub provider: String,
    /// The model to target.
    pub model: String,
    /// The pieces of text to embed -- one embedding comes back per entry,
    /// in the same order.
    pub input: Vec<String>,
    /// Requests a shorter output vector than the model's default, for
    /// models that support it.
    pub dimensions: Option<u32>,
}

fn build(
    registry: &Registry,
    body: EmbeddingsRequestBody,
) -> Result<PendingEmbeddingsRequest, ApiError> {
    let mut request = registry.embeddings(&body.provider, body.model)?;
    for input in body.input {
        request = request.with_input(input);
    }
    if let Some(dimensions) = body.dimensions {
        request = request.with_dimensions(dimensions);
    }
    Ok(request)
}

pub(crate) async fn embeddings(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<EmbeddingsRequestBody>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    let response = build(&registry, body)?.generate().await?;
    Ok(Json(response))
}

pub(crate) async fn embeddings_multi_tenant(
    State(tenants): State<Arc<dyn TenantRegistry>>,
    TenantContext(context): TenantContext,
    Json(body): Json<EmbeddingsRequestBody>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    let registry = tenants.resolve(&context).await?;
    let response = build(&registry, body)?.generate().await?;
    Ok(Json(response))
}
