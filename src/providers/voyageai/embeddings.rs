//! Wire format and mapping for VoyageAI's Embeddings API
//! (`POST /v1/embeddings`) -- the only endpoint this provider has, since
//! VoyageAI is an embeddings specialist (frequently used alongside
//! Anthropic, which has no embeddings endpoint of its own).

use serde::{Deserialize, Serialize};

use crate::embeddings::{Embedding, EmbeddingsRequest, EmbeddingsResponse};
use crate::value_objects::{EmbeddingsUsage, Meta};

#[derive(Debug, Serialize)]
pub struct ApiRequest {
    pub input: Vec<String>,
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct ApiResponse {
    pub data: Vec<ApiEmbedding>,
    pub model: String,
    pub usage: ApiUsage,
}

#[derive(Debug, Deserialize)]
pub struct ApiEmbedding {
    /// Which input this embedding corresponds to -- used to put `data` back
    /// into input order in [`parse_response`], the same defensive choice
    /// OpenAI's embeddings mapping makes rather than assuming array order.
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ApiUsage {
    /// VoyageAI reports only a single token count for an embeddings call (no
    /// prompt/completion split, since there's no completion) -- this maps
    /// directly onto [`EmbeddingsUsage::prompt_tokens`].
    pub total_tokens: u32,
}

pub fn build_request(request: &EmbeddingsRequest) -> ApiRequest {
    ApiRequest {
        input: request.input.clone(),
        model: request.model.clone(),
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
            prompt_tokens: response.usage.total_tokens,
        },
        meta: Meta {
            id: None,
            model: Some(response.model),
            rate_limits: Vec::new(),
        },
    }
}
