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
    FinishReason, MediaPart, Message, Meta, ToolCall, ToolOutcome, ToolResult, Usage,
};

use super::wire::{
    ChatChoice, ChatMessage, ChatRequest, ChatResponse, ChatTool, ChatToolCall,
    ChatToolCallFunction, ChatToolFunction,
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
    }
}

fn push_message(messages: &mut Vec<ChatMessage>, message: &Message) {
    match message {
        Message::System(system) => messages.push(ChatMessage::System {
            content: system.content.clone(),
        }),
        Message::User(user) => messages.push(ChatMessage::User {
            content: user_text(user),
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

fn user_text(user: &crate::value_objects::UserMessage) -> String {
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
