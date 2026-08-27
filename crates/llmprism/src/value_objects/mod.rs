//! The plain data types shared across every provider and capability --
//! conversation messages, tool calls, token usage, and so on. If you're building
//! a request by hand (rather than through
//! [`PendingTextRequest`](crate::text::PendingTextRequest)'s builder methods) or
//! inspecting a response in detail, this is where the pieces live.

pub mod finish_reason;
pub mod message;
pub mod meta;
pub mod tool_call;
pub mod usage;

pub use finish_reason::FinishReason;
pub use message::{
    AssistantMessage, Media, MediaData, MediaPart, Message, SystemMessage, ToolResultMessage,
    UserMessage,
};
pub use meta::{Meta, RateLimit};
pub use tool_call::{ToolCall, ToolOutcome, ToolOutput, ToolResult};
pub use usage::{EmbeddingsUsage, Usage};
