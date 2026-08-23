//! Streaming events -- **not implemented yet**. This module currently only
//! defines the [`StreamEvent`] shape so the rest of the crate can be written
//! against it; no provider produces these events today. Streaming support
//! (`Provider::stream_text_once` plus server-sent-events parsing) is planned for
//! a later release. Text generation without streaming (see
//! [`crate::text`]) already works.

use serde::{Deserialize, Serialize};

use crate::text::TextResponse;
use crate::value_objects::{FinishReason, ToolCall, Usage};

/// One event in a streamed reply, once streaming is implemented -- for example,
/// a small chunk of text as soon as the model generates it, rather than waiting
/// for the entire reply.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// The stream has begun.
    StreamStart,
    /// A chunk of reply text has arrived.
    TextDelta { text: String },
    /// Part of a tool call's arguments has arrived (arguments may be streamed in
    /// incrementally rather than all at once).
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    /// A tool call has finished arriving and is ready to be run.
    ToolCall(ToolCall),
    /// One round trip's worth of streaming has finished.
    StepFinish {
        usage: Usage,
        finish_reason: FinishReason,
    },
    /// The entire stream is complete; carries the same final result you'd get
    /// from the non-streaming [`generate`](crate::text::PendingTextRequest::generate).
    StreamEnd { response: TextResponse },
    /// Something went wrong partway through the stream.
    Error { message: String },
}
