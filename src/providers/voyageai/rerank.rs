//! Wire format and mapping for VoyageAI's Rerank API (`POST /v1/rerank`).

use serde::{Deserialize, Serialize};

use crate::rerank::{RankedDocument, RerankRequest, RerankResponse};
use crate::value_objects::{EmbeddingsUsage, Meta};

#[derive(Debug, Serialize)]
pub struct ApiRequest {
    pub query: String,
    pub documents: Vec<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    pub return_documents: bool,
}

#[derive(Debug, Deserialize)]
pub struct ApiResponse {
    pub data: Vec<ApiResult>,
    pub model: String,
    pub usage: ApiUsage,
}

#[derive(Debug, Deserialize)]
pub struct ApiResult {
    pub index: usize,
    pub relevance_score: f32,
    #[serde(default)]
    pub document: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiUsage {
    /// Same shape as VoyageAI's embeddings usage -- one total token count,
    /// no prompt/completion split.
    pub total_tokens: u32,
}

pub fn build_request(request: &RerankRequest) -> ApiRequest {
    ApiRequest {
        query: request.query.clone(),
        documents: request.documents.clone(),
        model: request.model.clone(),
        top_k: request.top_k,
        return_documents: request.return_documents,
    }
}

pub fn parse_response(response: ApiResponse) -> RerankResponse {
    RerankResponse {
        results: response
            .data
            .into_iter()
            .map(|result| RankedDocument {
                index: result.index,
                relevance_score: result.relevance_score,
                document: result.document,
            })
            .collect(),
        usage: EmbeddingsUsage {
            prompt_tokens: response.usage.total_tokens,
        },
        meta: Meta {
            id: None,
            model: Some(response.model),
            rate_limits: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_request_passes_top_k_and_return_documents_through() {
        let mut request = RerankRequest::new("rerank-2.5", "capital of France");
        request.documents = vec!["Paris is in France.".to_string()];
        request.top_k = Some(1);
        request.return_documents = true;

        let wire_request = build_request(&request);

        assert_eq!(wire_request.top_k, Some(1));
        assert!(wire_request.return_documents);
        assert_eq!(wire_request.documents, vec!["Paris is in France."]);
    }

    #[test]
    fn parse_response_maps_results_and_usage() {
        let response = ApiResponse {
            data: vec![
                ApiResult {
                    index: 1,
                    relevance_score: 0.95,
                    document: None,
                },
                ApiResult {
                    index: 0,
                    relevance_score: 0.2,
                    document: Some("low relevance".to_string()),
                },
            ],
            model: "rerank-2.5".to_string(),
            usage: ApiUsage { total_tokens: 42 },
        };

        let parsed = parse_response(response);

        assert_eq!(parsed.results.len(), 2);
        assert_eq!(parsed.results[0].index, 1);
        assert_eq!(parsed.results[0].relevance_score, 0.95);
        assert_eq!(parsed.results[1].document.as_deref(), Some("low relevance"));
        assert_eq!(parsed.usage.prompt_tokens, 42);
    }
}
