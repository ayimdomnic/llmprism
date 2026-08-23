//! Wire format and mapping for Gemini's Embeddings API
//! (`POST /v1beta/models/{model}:batchEmbedContents`). Uses the batch
//! endpoint rather than the single-input `embedContent` one, since this
//! crate's [`EmbeddingsRequest`] accepts more than one input per call.

use serde::{Deserialize, Serialize};

use crate::embeddings::{Embedding, EmbeddingsRequest, EmbeddingsResponse};
use crate::value_objects::{EmbeddingsUsage, Meta};

#[derive(Debug, Serialize)]
pub struct BatchEmbedContentsRequest {
    pub requests: Vec<EmbedContentRequestEntry>,
}

/// One input's request. Somewhat redundant -- `model` is repeated on every
/// entry even though every entry in a batch shares the same model -- but
/// that's the shape Gemini's own API documents.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentRequestEntry {
    pub model: String,
    pub content: EmbedContentBody,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed_content_config: Option<EmbedContentConfig>,
}

#[derive(Debug, Serialize)]
pub struct EmbedContentBody {
    pub parts: Vec<EmbedContentPart>,
}

#[derive(Debug, Serialize)]
pub struct EmbedContentPart {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimensionality: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchEmbedContentsResponse {
    #[serde(default)]
    pub embeddings: Vec<EmbeddingValues>,
    #[serde(default)]
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingValues {
    #[serde(default)]
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(default)]
    pub prompt_token_count: u32,
}

pub fn build_request(request: &EmbeddingsRequest, model: &str) -> BatchEmbedContentsRequest {
    let model_path = format!("models/{model}");
    let config = request
        .dimensions
        .map(|output_dimensionality| EmbedContentConfig {
            output_dimensionality: Some(output_dimensionality),
        });

    BatchEmbedContentsRequest {
        requests: request
            .input
            .iter()
            .map(|text| EmbedContentRequestEntry {
                model: model_path.clone(),
                content: EmbedContentBody {
                    parts: vec![EmbedContentPart { text: text.clone() }],
                },
                embed_content_config: config.clone(),
            })
            .collect(),
    }
}

pub fn parse_response(response: BatchEmbedContentsResponse, model: &str) -> EmbeddingsResponse {
    EmbeddingsResponse {
        embeddings: response
            .embeddings
            .into_iter()
            .map(|embedding| Embedding {
                vector: embedding.values,
            })
            .collect(),
        usage: EmbeddingsUsage {
            prompt_tokens: response
                .usage_metadata
                .map(|usage| usage.prompt_token_count)
                .unwrap_or_default(),
        },
        meta: Meta {
            id: None,
            model: Some(model.to_string()),
            rate_limits: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_request_sends_one_entry_per_input_sharing_the_same_model() {
        let mut request = EmbeddingsRequest::new("text-embedding-004");
        request.input = vec!["hello".to_string(), "world".to_string()];

        let wire_request = build_request(&request, "text-embedding-004");

        assert_eq!(wire_request.requests.len(), 2);
        assert_eq!(wire_request.requests[0].model, "models/text-embedding-004");
        assert_eq!(wire_request.requests[0].content.parts[0].text, "hello");
        assert_eq!(wire_request.requests[1].content.parts[0].text, "world");
    }

    #[test]
    fn build_request_omits_embed_content_config_when_no_dimensions_are_set() {
        let mut request = EmbeddingsRequest::new("text-embedding-004");
        request.input = vec!["hello".to_string()];

        let wire_request = build_request(&request, "text-embedding-004");

        assert!(wire_request.requests[0].embed_content_config.is_none());
    }

    #[test]
    fn build_request_passes_dimensions_through_as_output_dimensionality() {
        let mut request = EmbeddingsRequest::new("text-embedding-004");
        request.input = vec!["hello".to_string()];
        request.dimensions = Some(256);

        let wire_request = build_request(&request, "text-embedding-004");

        assert_eq!(
            wire_request.requests[0]
                .embed_content_config
                .as_ref()
                .unwrap()
                .output_dimensionality,
            Some(256)
        );
    }
}
