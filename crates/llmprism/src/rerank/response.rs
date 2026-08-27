use serde::{Deserialize, Serialize};

use crate::value_objects::{EmbeddingsUsage, Meta};

/// The result of a rerank call: every input document, scored and sorted by
/// relevance to the query (most relevant first).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    pub results: Vec<RankedDocument>,
    /// Reuses [`EmbeddingsUsage`] rather than introducing a third
    /// near-identical "just a token count" type -- reranking, like
    /// embedding, has no completion, cache, or thinking tokens to report.
    pub usage: EmbeddingsUsage,
    pub meta: Meta,
}

/// One scored document from a [`RerankResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedDocument {
    /// This document's position in the original `documents` list passed to
    /// the request -- use this to look the original text back up rather than
    /// relying on `document` being set, since it usually isn't (see
    /// [`PendingRerankRequest::with_return_documents`](super::request::PendingRerankRequest::with_return_documents)).
    pub index: usize,
    /// How relevant this document is to the query. Higher is more relevant;
    /// the exact scale is provider-specific (not necessarily 0.0-1.0), so
    /// only compare scores within the same response, not across providers or
    /// requests.
    pub relevance_score: f32,
    /// The document's own text, echoed back by the provider -- only present
    /// when the request asked for it, since you already have this list
    /// yourself otherwise.
    #[serde(default)]
    pub document: Option<String>,
}
