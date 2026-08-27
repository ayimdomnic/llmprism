//! `POST /v1/rerank`.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use llmprism::rerank::RerankResponse;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;

#[derive(Deserialize)]
pub(crate) struct RerankRequestBody {
    provider: String,
    model: String,
    query: String,
    #[serde(default)]
    documents: Vec<String>,
    top_k: Option<u32>,
    return_documents: Option<bool>,
}

pub(crate) async fn rerank(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<RerankRequestBody>,
) -> Result<Json<RerankResponse>, ApiError> {
    let mut request = registry
        .rerank(&body.provider, body.model, body.query)?
        .with_documents(body.documents);
    if let Some(top_k) = body.top_k {
        request = request.with_top_k(top_k);
    }
    if let Some(return_documents) = body.return_documents {
        request = request.with_return_documents(return_documents);
    }
    let response = request.generate().await?;
    Ok(Json(response))
}
