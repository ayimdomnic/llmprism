//! `POST /v1/rerank` -- given a query and a list of documents, score and
//! sort them by relevance.
//!
//! Request body: [`RerankRequestBody`]. Response: [`RerankResponse`], every
//! document scored and sorted (most relevant first).
//!
//! ```json
//! {
//!   "provider": "voyageai",
//!   "model": "rerank-2",
//!   "query": "What is the capital of France?",
//!   "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."]
//! }
//! ```

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use llmprism::rerank::RerankResponse;
use llmprism::Registry;
use serde::Deserialize;

use crate::error::ApiError;

/// The JSON body for `POST /v1/rerank`.
#[derive(Deserialize)]
pub struct RerankRequestBody {
    /// Name of the provider registered in the `Registry` this router was
    /// built from.
    pub provider: String,
    /// The model to target.
    pub model: String,
    /// The query every document is scored against.
    pub query: String,
    /// The candidate documents to score and sort.
    #[serde(default)]
    pub documents: Vec<String>,
    /// Keeps only the `top_k` most relevant documents in the response,
    /// instead of every document just sorted.
    pub top_k: Option<u32>,
    /// Asks the provider to echo each document's own text back in the
    /// response.
    pub return_documents: Option<bool>,
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
