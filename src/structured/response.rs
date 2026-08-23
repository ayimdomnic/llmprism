use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::value_objects::{FinishReason, Meta, Usage};

/// The result of a structured-output request: `data` is JSON matching the shape
/// you asked for in [`StructuredRequest::schema`](super::StructuredRequest::schema),
/// ready to deserialize into your own type with `serde_json::from_value`.
///
/// Unlike [`TextResponse`](crate::text::TextResponse), there's no multi-step tool
/// -calling loop here and so no `steps` list -- a structured-output request is
/// always exactly one request/response round trip, regardless of which strategy
/// the provider uses underneath to get the model to comply with the schema (see
/// [`crate::structured`] for what that means).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredResponse {
    pub data: Value,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub meta: Meta,
}
