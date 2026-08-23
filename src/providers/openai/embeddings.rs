//! Wire format and mapping for OpenAI's Embeddings API
//! (`POST /v1/embeddings`) -- its own endpoint and wire shape, same as
//! moderation.

use serde::{Deserialize, Serialize};

use crate::embeddings::{Embedding, EmbeddingsRequest, EmbeddingsResponse};
use crate::value_objects::{EmbeddingsUsage, Meta};

#[derive(Debug, Serialize)]
pub struct ApiRequest {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiResponse {
    pub model: String,
    pub data: Vec<ApiEmbedding>,
    pub usage: ApiUsage,
}

#[derive(Debug, Deserialize)]
pub struct ApiEmbedding {
    /// Which input this embedding corresponds to -- used to put `data` back
    /// into input order in [`parse_response`] rather than assuming the API
    /// already returned it that way.
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ApiUsage {
    pub prompt_tokens: u32,
}

pub fn build_request(request: &EmbeddingsRequest) -> ApiRequest {
    ApiRequest {
        model: request.model.clone(),
        input: request.input.clone(),
    }
}

pub fn parse_response(mut response: ApiResponse) -> EmbeddingsResponse {
    response.data.sort_by_key(|embedding| embedding.index);

    EmbeddingsResponse {
        embeddings: response
            .data
            .into_iter()
            .map(|embedding| Embedding {
                vector: embedding.embedding,
            })
            .collect(),
        usage: EmbeddingsUsage {
            prompt_tokens: response.usage.prompt_tokens,
        },
        meta: Meta {
            id: None,
            model: Some(response.model),
            rate_limits: Vec::new(),
        },
    }
}
