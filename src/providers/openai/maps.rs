//! Translates between llmprism's provider-agnostic value objects and OpenAI's Chat
//! Completions wire format -- ported from Prism's `Providers/OpenAI/Maps/*`.

use serde_json::json;

use crate::error::Error;
use crate::schema::{to_json_schema, Schema};
use crate::structured::{StructuredRequest, StructuredResponse};
use crate::text::request::{TextRequest, ToolChoice};
use crate::text::response::Step;
use crate::tool::Tool;
use crate::value_objects::{
    FinishReason, Media, MediaData, MediaPart, Message, Meta, ToolCall, ToolOutcome, ToolResult,
    Usage, UserMessage,
};

use super::wire::{
    ChatChoice, ChatMessage, ChatRequest, ChatResponse, ChatTool, ChatToolCall,
    ChatToolCallFunction, ChatToolFunction, ImageUrl, UserContent, UserContentPart,
};

pub fn build_request(request: &TextRequest) -> ChatRequest {
    let mut messages = Vec::new();

    for system_prompt in &request.system_prompts {
        messages.push(ChatMessage::System {
            content: system_prompt.clone(),
        });
    }

    for message in &request.messages {
        push_message(&mut messages, message);
    }

    ChatRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        tools: request
            .tools
            .iter()
            .map(|tool| to_wire_tool(tool.as_ref()))
            .collect(),
        tool_choice: to_wire_tool_choice(&request.tool_choice, request.tools.is_empty()),
        stream: None,
        stream_options: None,
        response_format: None,
        reasoning_effort: request.reasoning_effort.clone(),
    }
}

/// Builds a Chat Completions request for a structured-output call, using
/// OpenAI's native `response_format: {"type": "json_schema", ...}` -- the API
/// enforces the schema server-side, so (unlike Anthropic's forced-tool-call
/// strategy) no tool is involved here at all.
pub fn build_structured_request(request: &StructuredRequest) -> ChatRequest {
    let mut messages = Vec::new();

    for system_prompt in &request.system_prompts {
        messages.push(ChatMessage::System {
            content: system_prompt.clone(),
        });
    }

    for message in &request.messages {
        push_message(&mut messages, message);
    }

    ChatRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        tools: Vec::new(),
        tool_choice: None,
        stream: None,
        stream_options: None,
        response_format: Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": request.schema.name,
                "schema": to_json_schema(&Schema::Object(request.schema.clone())),
                "strict": true,
            }
        })),
        reasoning_effort: request.reasoning_effort.clone(),
    }
}

fn push_message(messages: &mut Vec<ChatMessage>, message: &Message) {
    match message {
        Message::System(system) => messages.push(ChatMessage::System {
            content: system.content.clone(),
        }),
        Message::User(user) => messages.push(ChatMessage::User {
            content: user_content(user),
        }),
        Message::Assistant(assistant) => messages.push(ChatMessage::Assistant {
            content: assistant.content.clone(),
            tool_calls: assistant.tool_calls.iter().map(to_wire_tool_call).collect(),
        }),
        Message::ToolResult(tool_result) => {
            for result in &tool_result.tool_results {
                messages.push(ChatMessage::Tool {
                    tool_call_id: result.tool_call_id.clone(),
                    content: tool_result_text(result),
                });
            }
        }
    }
}

/// Builds a user message's `content`. Chat Completions has no first-class
/// document/PDF content part the way Anthropic and Gemini do -- OpenAI's
/// equivalent needs uploading through the separate Files API and referencing
/// it by id, a whole extra request/response round trip this crate's
/// synchronous "translate one message into one wire value" mapping functions
/// aren't shaped for -- and no audio/video content part in Chat Completions
/// messages at all, so [`MediaPart::Document`]/`Audio`/`Video` are silently
/// dropped here, the same as they are for every provider that has no wire
/// shape for them.
pub(crate) fn user_content(user: &UserMessage) -> UserContent {
    let has_image = user
        .content
        .iter()
        .any(|part| matches!(part, MediaPart::Image(_)));

    if !has_image {
        // Keep sending the plain-string shape whenever there's nothing but
        // text -- it's what this crate has always sent for a text-only
        // message, and there's no reason to switch every request to the
        // more verbose array shape just because *some* requests need it.
        return UserContent::Text(user_text(user));
    }

    let parts = user
        .content
        .iter()
        .filter_map(|part| match part {
            MediaPart::Text(text) => Some(UserContentPart::Text { text: text.clone() }),
            MediaPart::Image(media) => Some(UserContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: image_url_string(media),
                },
            }),
            MediaPart::Document(_) | MediaPart::Audio(_) | MediaPart::Video(_) => None,
        })
        .collect();

    UserContent::Parts(parts)
}

/// The `image_url.url` value for a [`MediaPart::Image`]: passed through
/// as-is for a real URL, or turned into a `data:` URI for embedded bytes --
/// OpenAI accepts both in the same field, with no separate "base64" shape
/// the way Anthropic and Gemini have.
fn image_url_string(media: &Media) -> String {
    match &media.data {
        MediaData::Url(url) => url.clone(),
        MediaData::Base64(data) => {
            let mime_type = media.mime_type.as_deref().unwrap_or("image/png");
            format!("data:{mime_type};base64,{data}")
        }
    }
}

fn user_text(user: &UserMessage) -> String {
    user.content
        .iter()
        .filter_map(|part| match part {
            MediaPart::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_result_text(result: &ToolResult) -> String {
    match &result.result {
        ToolOutcome::Output(output) => output.content.clone(),
        ToolOutcome::Error(message) => message.clone(),
    }
}

fn to_wire_tool_call(call: &ToolCall) -> ChatToolCall {
    ChatToolCall {
        id: call.id.clone(),
        kind: "function".to_string(),
        function: ChatToolCallFunction {
            name: call.name.clone(),
            arguments: call.arguments.to_string(),
        },
    }
}

fn to_wire_tool(tool: &dyn Tool) -> ChatTool {
    ChatTool {
        kind: "function",
        function: ChatToolFunction {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: to_json_schema(&crate::schema::Schema::Object(tool.parameters().clone())),
        },
    }
}

fn to_wire_tool_choice(choice: &ToolChoice, no_tools: bool) -> Option<serde_json::Value> {
    if no_tools {
        return None;
    }
    match choice {
        ToolChoice::Auto => Some(json!("auto")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Any => Some(json!("required")),
        ToolChoice::Tool(name) => Some(json!({"type": "function", "function": {"name": name}})),
    }
}

pub fn parse_response(response: ChatResponse) -> Step {
    let ChatResponse {
        id,
        model,
        choices,
        usage,
    } = response;

    let choice: ChatChoice = choices
        .into_iter()
        .next()
        .expect("chat completion response returns at least one choice");

    let tool_calls: Vec<ToolCall> = choice
        .message
        .tool_calls
        .iter()
        .map(from_wire_tool_call)
        .collect();

    let finish_reason = map_finish_reason(&choice.finish_reason);
    let usage = usage.map(map_usage).unwrap_or_default();

    Step {
        text: choice.message.content,
        tool_calls,
        finish_reason,
        usage,
        meta: Meta {
            id: Some(id),
            model: Some(model),
            rate_limits: Vec::new(),
        },
    }
}

/// Parses a Chat Completions response returned for a structured-output
/// request. Unlike [`parse_response`], this can fail: `content` is a JSON
/// *string* here (not already-structured data the way the rest of this crate's
/// types are), so decoding it is a second, fallible parse pass on top of the
/// one that already turned the HTTP body into a [`ChatResponse`].
pub fn parse_structured_response(
    response: ChatResponse,
    provider_name: &str,
) -> Result<StructuredResponse, Error> {
    let ChatResponse {
        id,
        model,
        choices,
        usage,
    } = response;

    let choice: ChatChoice = choices
        .into_iter()
        .next()
        .expect("chat completion response returns at least one choice");

    if let Some(refusal) = choice.message.refusal {
        return Err(Error::Provider {
            provider: provider_name.to_string(),
            status: 0,
            kind: Some("refusal".to_string()),
            message: refusal,
        });
    }

    let content = choice
        .message
        .content
        .ok_or_else(|| Error::StructuredDecode {
            provider: provider_name.to_string(),
            message: "response contained no content".to_string(),
        })?;

    let data = serde_json::from_str(&content).map_err(|e| Error::StructuredDecode {
        provider: provider_name.to_string(),
        message: e.to_string(),
    })?;

    Ok(StructuredResponse {
        data,
        finish_reason: map_finish_reason(&choice.finish_reason),
        usage: usage.map(map_usage).unwrap_or_default(),
        meta: Meta {
            id: Some(id),
            model: Some(model),
            rate_limits: Vec::new(),
        },
    })
}

fn from_wire_tool_call(call: &ChatToolCall) -> ToolCall {
    ToolCall {
        id: call.id.clone(),
        name: call.function.name.clone(),
        arguments: serde_json::from_str(&call.function.arguments)
            .unwrap_or(serde_json::Value::Null),
    }
}

/// Maps a Chat Completions `finish_reason` string to this crate's
/// [`FinishReason`]. Shared between the non-streaming response parser above and
/// the streaming chunk parser in `mod.rs`, since both encounter the exact same
/// set of strings.
pub(crate) fn map_finish_reason(finish_reason: &str) -> FinishReason {
    match finish_reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
}

/// Maps a Chat Completions `usage` object to this crate's [`Usage`]. Shared for
/// the same reason as [`map_finish_reason`] -- streaming and non-streaming
/// responses report usage in the identical shape.
pub(crate) fn map_usage(usage: super::wire::ChatUsage) -> Usage {
    Usage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        cache_write_tokens: None,
        cache_read_tokens: usage.prompt_tokens_details.and_then(|d| d.cached_tokens),
        thought_tokens: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_content_stays_a_plain_string_for_text_only_messages() {
        let user = UserMessage::text("hello");
        assert!(matches!(user_content(&user), UserContent::Text(text) if text == "hello"));
    }

    #[test]
    fn user_content_switches_to_parts_when_an_image_is_present() {
        let user = UserMessage {
            content: vec![
                MediaPart::Text("What's in this image?".to_string()),
                MediaPart::Image(Media {
                    mime_type: None,
                    data: MediaData::Base64("aGVsbG8=".to_string()),
                }),
                // No wire shape exists for documents in Chat Completions
                // messages, so this should be silently dropped.
                MediaPart::Document(Media {
                    mime_type: Some("application/pdf".to_string()),
                    data: MediaData::Url("https://example.com/doc.pdf".to_string()),
                }),
            ],
        };

        let UserContent::Parts(parts) = user_content(&user) else {
            panic!("expected UserContent::Parts once an image is present");
        };

        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], UserContentPart::Text { .. }));
        assert!(matches!(
            &parts[1],
            UserContentPart::ImageUrl { image_url } if image_url.url == "data:image/png;base64,aGVsbG8="
        ));
    }

    #[test]
    fn image_url_string_passes_a_real_url_through_unchanged() {
        let media = Media {
            mime_type: None,
            data: MediaData::Url("https://example.com/cat.png".to_string()),
        };
        assert_eq!(image_url_string(&media), "https://example.com/cat.png");
    }

    #[test]
    fn build_request_omits_reasoning_effort_by_default() {
        let request = TextRequest::new("gpt-4o-mini");
        let wire_request = build_request(&request);
        assert!(wire_request.reasoning_effort.is_none());
    }

    #[test]
    fn build_request_passes_reasoning_effort_through_when_set() {
        let mut request = TextRequest::new("o3-mini");
        request.reasoning_effort = Some("high".to_string());
        let wire_request = build_request(&request);
        assert_eq!(wire_request.reasoning_effort.as_deref(), Some("high"));
    }
}
