//! Serde structs mirroring the OpenAI Chat Completions API
//! (`POST /v1/chat/completions`) wire format.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ChatTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Only meaningful alongside `stream: Some(true)` -- asks the API to include
    /// one final chunk carrying token usage, since usage is otherwise omitted
    /// entirely from a streamed response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
    /// Set for structured-output requests: `{"type": "json_schema",
    /// "json_schema": {"name", "schema", "strict"}}`, which OpenAI enforces
    /// server-side -- the model's reply is guaranteed (not just prompted) to
    /// match `schema`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// A reasoning model's effort level (`"minimal"`, `"low"`, `"medium"`,
    /// `"high"`, and newer values on some models) -- see
    /// [`TextRequest::reasoning_effort`](crate::text::TextRequest::reasoning_effort).
    /// Non-reasoning models reject this outright rather than ignoring it, so
    /// it's only set when the caller explicitly asked for it.
    ///
    /// One nuance this crate doesn't handle for you: OpenAI's reasoning
    /// models want `max_completion_tokens` instead of the `max_tokens` this
    /// crate otherwise always sends, and reject requests that send the
    /// latter. If you hit that, leave
    /// [`with_max_tokens`](crate::text::PendingTextRequest::with_max_tokens)
    /// unset (so this crate omits `max_tokens` entirely) and set
    /// `max_completion_tokens` yourself via `provider_options` instead --
    /// see [`TextRequest::provider_options`](crate::text::TextRequest::provider_options).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System {
        content: String,
    },
    User {
        content: UserContent,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ChatToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// A user message's `content`: Chat Completions accepts either a plain
/// string or an array of typed parts. This crate sends the plain-string
/// shape whenever a message is text-only (the overwhelmingly common case,
/// and the shape this crate has always sent), switching to `Parts` only once
/// there's an image to include alongside the text -- see
/// [`super::maps::user_content`].
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Parts(Vec<UserContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ChatToolFunction,
}

#[derive(Debug, Serialize)]
pub struct ChatToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ChatToolCall>,
    /// Set instead of `content` when the model declines to comply with a
    /// structured-output request (e.g. the request violates OpenAI's usage
    /// policies) -- only relevant to the structured-output strategy.
    #[serde(default)]
    pub refusal: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<ChatPromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub struct ChatPromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

/// One `data: {...}` chunk from a streamed (`"stream": true`) chat completion.
/// Every field is optional/defaulted because a single stream is made up of many
/// of these, each carrying only whatever changed since the last one -- an early
/// chunk might have a role but no content, a middle chunk a content delta and
/// nothing else, and (with `stream_options.include_usage`) a final chunk with
/// `usage` set and an empty `choices` array.
#[derive(Debug, Deserialize)]
pub struct ChatStreamChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatStreamChoice>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatStreamChoice {
    #[serde(default)]
    pub delta: ChatStreamDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatStreamDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ChatStreamToolCallDelta>,
}

/// A tool call as it appears mid-stream: `index` identifies which tool call this
/// delta belongs to (a single chunk's `tool_calls` array covers all tool calls
/// in progress, not just one), and `id`/`function.name` typically arrive once,
/// on the first delta for that index, while `function.arguments` arrives as
/// many small fragments that must be concatenated in order.
#[derive(Debug, Deserialize)]
pub struct ChatStreamToolCallDelta {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ChatStreamFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatStreamFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}
