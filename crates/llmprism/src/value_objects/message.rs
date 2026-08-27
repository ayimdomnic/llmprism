use serde::{Deserialize, Serialize};

use super::tool_call::{ToolCall, ToolResult};

/// One turn of a conversation with the model.
///
/// A [`TextRequest`](crate::text::TextRequest)'s `messages` field is a list of
/// these, oldest first. Most applications only ever construct
/// [`Message::User`] directly (via
/// [`PendingTextRequest::with_prompt`](crate::text::PendingTextRequest::with_prompt));
/// the other variants are mostly produced automatically by the tool-calling loop
/// as it records what the model said and what tools returned.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// Instructions for how the model should behave, appearing mid-conversation
    /// rather than as a one-time system prompt. Most applications should prefer
    /// [`PendingTextRequest::with_system_prompt`](crate::text::PendingTextRequest::with_system_prompt)
    /// instead, which is sent once, up front.
    System(SystemMessage),
    /// Something said by the person (or your application, on their behalf).
    User(UserMessage),
    /// Something the model said, possibly along with a request to call one or
    /// more tools.
    Assistant(AssistantMessage),
    /// The results of tool calls the model previously requested, sent back so
    /// the model can continue the conversation. Added automatically by
    /// [`crate::tool_loop::run_text`].
    ToolResult(ToolResultMessage),
}

/// See [`Message::System`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub content: String,
}

/// See [`Message::User`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    /// The message's content, as one or more parts -- usually just a single
    /// [`MediaPart::Text`], but potentially images/documents/audio/video for
    /// providers and models that accept them.
    pub content: Vec<MediaPart>,
}

impl UserMessage {
    /// Creates a plain-text user message with a single text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![MediaPart::Text(text.into())],
        }
    }
}

/// See [`Message::Assistant`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// The text the model said, if any (may be `None` if the model's entire
    /// reply was a tool call).
    pub content: Option<String>,
    /// Tools the model asked to call as part of this message.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// See [`Message::ToolResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultMessage {
    /// The results of every tool call being answered in this message.
    pub tool_results: Vec<ToolResult>,
}

/// One piece of a [`UserMessage`]'s content.
///
/// Support for the media variants varies by provider, since it varies by
/// what each provider's own API actually accepts: OpenAI translates `Image`
/// (there's no first-class document/PDF content part in its Chat Completions
/// API without a separate file-upload step this crate doesn't perform, and
/// no audio/video content part in messages at all); Anthropic translates
/// `Image` and `Document` (it has no audio/video message content at all);
/// Gemini translates all four, since its wire format has one uniform
/// mime-type-tagged part shape for every media kind rather than a distinct
/// one per kind. A variant a given provider doesn't translate is silently
/// dropped from the request rather than causing an error -- the same
/// "unsupported means it's just not there" convention every other capability
/// in this crate follows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediaPart {
    /// Plain text.
    Text(String),
    Image(Media),
    Document(Media),
    Audio(Media),
    Video(Media),
}

/// A piece of media attached to a message, either by URL or embedded directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Media {
    /// The media's MIME type (e.g. `"image/png"`), when known.
    pub mime_type: Option<String>,
    pub data: MediaData,
}

/// Where a [`Media`]'s bytes actually are.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MediaData {
    /// A URL the provider should fetch the media from.
    Url(String),
    /// The media's bytes, base64-encoded and embedded directly in the request.
    Base64(String),
}
