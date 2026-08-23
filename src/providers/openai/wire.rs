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
}

#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System {
        content: String,
    },
    User {
        content: String,
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
