//! Streaming events -- what a streaming Text request yields incrementally,
//! instead of making you wait for the entire reply. See
//! [`PendingTextRequest::stream`](crate::text::PendingTextRequest::stream).

use serde::{Deserialize, Serialize};

use crate::text::TextResponse;
use crate::value_objects::{FinishReason, Meta, ToolCall, ToolResult, Usage};

/// One event in a streamed reply.
///
/// A single [`stream`](crate::text::PendingTextRequest::stream) call produces a
/// sequence shaped like: one [`StreamStart`](StreamEvent::StreamStart), then any
/// number of [`TextDelta`](StreamEvent::TextDelta) /
/// [`ToolCallDelta`](StreamEvent::ToolCallDelta) events as the model's reply
/// arrives, a [`ToolCall`](StreamEvent::ToolCall) once each tool call has fully
/// arrived, a [`StepFinish`](StreamEvent::StepFinish) once that round trip is
/// done -- and then, if the model asked to call a tool, a
/// [`ToolResult`](StreamEvent::ToolResult) for each one that ran, followed by
/// another `StreamStart` for the next round trip. The whole sequence ends with
/// exactly one [`StreamEnd`](StreamEvent::StreamEnd).
///
/// Errors are reported through the `Result` the stream itself yields (`Result
/// <StreamEvent, Error>`), not as a variant of this enum -- so you only need to
/// check one place for something having gone wrong.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A new request/response round trip has begun. Carries whatever metadata
    /// (response id, model) the provider makes available up front.
    StreamStart { meta: Meta },
    /// A chunk of reply text has arrived.
    TextDelta { text: String },
    /// Part of a tool call has arrived. Providers stream a tool call's
    /// arguments in incrementally, so you may see several of these -- for the
    /// same `index` -- before the matching
    /// [`ToolCall`](StreamEvent::ToolCall) event. Most applications can ignore
    /// this and just wait for the complete `ToolCall`; it's here for UIs that
    /// want to show a tool call being "typed" live.
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    /// A tool call has fully arrived, with arguments parsed as JSON and ready
    /// to run.
    ToolCall(ToolCall),
    /// A tool call finished running and this is its result, about to be sent
    /// back to the model.
    ToolResult(ToolResult),
    /// One request/response round trip has finished.
    StepFinish {
        usage: Usage,
        finish_reason: FinishReason,
    },
    /// The entire stream is complete -- no more events follow. Carries the same
    /// final result you'd get back from the non-streaming
    /// [`generate`](crate::text::PendingTextRequest::generate).
    StreamEnd { response: TextResponse },
}
