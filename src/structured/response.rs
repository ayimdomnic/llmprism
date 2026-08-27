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

/// One event in a streamed structured-output reply. See
/// [`PendingStructuredRequest::stream`](super::PendingStructuredRequest::stream).
///
/// Unlike [`StreamEvent`](crate::StreamEvent) (Text generation's streaming
/// counterpart), there's no multi-step loop here either -- a structured
/// request is always exactly one round trip, so the sequence is simply zero
/// or more [`PartialObject`](Self::PartialObject) events followed by exactly
/// one [`End`](Self::End).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StructuredStreamEvent {
    /// A best-effort parse of everything the model has generated so far,
    /// repaired into valid JSON (an open string closed, an open object's
    /// missing `}` added, and so on) by whichever provider produced this
    /// event. Fields the model hasn't reached yet simply aren't present --
    /// this isn't guaranteed to match [`StructuredRequest::schema`](super::StructuredRequest::schema)
    /// until the matching [`End`](Self::End) arrives, so treat it as a
    /// preview, not something to validate against the schema partway
    /// through.
    PartialObject { data: Value },
    /// The stream is complete. Carries the same final result
    /// [`generate`](super::PendingStructuredRequest::generate) would have
    /// returned.
    End { response: StructuredResponse },
}
