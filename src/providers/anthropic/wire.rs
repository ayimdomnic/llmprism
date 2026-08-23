//! Serde structs mirroring the Anthropic Messages API (`POST /v1/messages`) wire
//! format.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    pub messages: Vec<MessageParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<MessagesTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// The `system` field: a plain string normally, matching the simplest shape
/// Anthropic accepts, or an array of one text block carrying a
/// `cache_control` breakpoint when prompt caching is requested (see
/// [`super::maps::build_system`]) -- Anthropic has no way to attach
/// `cache_control` to a plain string, only to a content block, so caching
/// the system prompt means sending it in the more verbose shape instead.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Serialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// A cache breakpoint on a content block. Only the `"ephemeral"` kind exists
/// today, so this crate doesn't model `kind` as a real field -- there's
/// nothing meaningful for a caller to set it to.
#[derive(Debug, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: &'static str,
}

impl CacheControl {
    pub fn ephemeral() -> Self {
        Self { kind: "ephemeral" }
    }
}

#[derive(Debug, Serialize)]
pub struct MessageParam {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        source: MediaSource,
    },
    Document {
        source: MediaSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    /// A content block type this crate doesn't translate yet -- extended
    /// thinking (`thinking`/`redacted_thinking`), citations, server-side
    /// tool use, and anything else Anthropic adds later. Without this,
    /// decoding a response containing one of these would fail outright with
    /// an `unknown variant` error instead of just ignoring the part this
    /// crate doesn't understand -- exactly the same reasoning behind
    /// [`StreamContentBlockStart::Other`], its streaming counterpart.
    #[serde(other)]
    Other,
}

/// Where a [`ContentBlock::Image`]/[`ContentBlock::Document`]'s bytes
/// actually are -- Anthropic's own union of the two ways this crate's
/// [`Media`](crate::value_objects::Media) can hold data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Serialize)]
pub struct MessagesTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: MessagesUsage,
}

#[derive(Debug, Deserialize)]
pub struct MessagesUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

/// One `event:`/`data:` pair from a streamed (`"stream": true`) Messages API
/// response. Unlike Chat Completions' uniform per-chunk shape, Anthropic's
/// stream is a sequence of distinctly-shaped, named events -- `message_start`
/// once, then `content_block_start`/`content_block_delta`/`content_block_stop`
/// for each piece of content (interleaved by `index` if there's more than one),
/// then `message_delta` and `message_stop`. This enum mirrors that directly via
/// serde's internally-tagged representation on the JSON body's own `"type"`
/// field (the SSE `event:` name carries the same information redundantly, so
/// this crate parses `data:` and ignores `event:`).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEventPayload {
    MessageStart {
        message: StreamMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: StreamContentBlockStart,
    },
    ContentBlockDelta {
        index: usize,
        delta: StreamDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: StreamMessageDelta,
        #[serde(default)]
        usage: Option<StreamMessageDeltaUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: StreamErrorDetail,
    },
    /// Catches any event type this crate doesn't specifically handle yet (e.g.
    /// extended-thinking or citation events), so an API addition doesn't turn
    /// into a hard decode failure for every consumer.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct StreamMessageStart {
    pub id: String,
    pub model: String,
    pub usage: MessagesUsage,
}

/// The content block a `content_block_start` event introduces. Only the two
/// shapes this crate translates into a [`StreamEvent`](crate::StreamEvent) are
/// named explicitly; anything else (thinking, citations, server-side tool
/// results, ...) falls into `Other` and its deltas are then safely ignored.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamContentBlockStart {
    Text {
        #[serde(default)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct StreamMessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StreamMessageDeltaUsage {
    #[serde(default)]
    pub output_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct StreamErrorDetail {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `thinking` block (Claude's extended-thinking output) used to fail
    /// this whole response's deserialization with an `unknown variant` error,
    /// since `ContentBlock` had no catch-all the way its streaming
    /// counterpart already did. It should now decode cleanly, with the
    /// unrecognized block simply becoming `ContentBlock::Other`.
    #[test]
    fn a_response_containing_an_unrecognized_content_block_type_still_decodes() {
        let json = r#"{
            "id": "msg_1",
            "model": "claude-3-5-haiku-20241022",
            "content": [
                {"type": "thinking", "thinking": "Let me think...", "signature": "abc"},
                {"type": "text", "text": "The answer is 4."}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;

        let response: MessagesResponse =
            serde_json::from_str(json).expect("should decode despite the unrecognized block");

        assert!(matches!(response.content[0], ContentBlock::Other));
        assert!(matches!(response.content[1], ContentBlock::Text { .. }));
    }
}
