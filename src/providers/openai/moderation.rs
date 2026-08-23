//! Wire format and mapping for OpenAI's Moderation API
//! (`POST /v1/moderations`) -- a distinct endpoint from Chat Completions, with
//! its own small wire shape, so (unlike Text/structured output) this doesn't
//! share `wire.rs`/`maps.rs`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::moderation::{ModerationRequest, ModerationResponse, ModerationResult};
use crate::value_objects::Meta;

#[derive(Debug, Serialize)]
pub struct ApiRequest {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiResponse {
    pub id: String,
    pub model: String,
    pub results: Vec<ApiResult>,
}

#[derive(Debug, Deserialize)]
pub struct ApiResult {
    pub flagged: bool,
    pub categories: HashMap<String, bool>,
    pub category_scores: HashMap<String, f64>,
}

pub fn build_request(request: &ModerationRequest) -> ApiRequest {
    ApiRequest {
        model: request.model.clone(),
        input: request.input.clone(),
    }
}

pub fn parse_response(response: ApiResponse) -> ModerationResponse {
    ModerationResponse {
        results: response
            .results
            .into_iter()
            .map(|result| ModerationResult {
                flagged: result.flagged,
                categories: result.categories,
                category_scores: result.category_scores,
            })
            .collect(),
        meta: Meta {
            id: Some(response.id),
            model: Some(response.model),
            rate_limits: Vec::new(),
        },
    }
}
