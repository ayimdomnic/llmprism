//! Translates between llmprism's provider-agnostic value objects and Anthropic's
//! Messages API wire format -- ported from Prism's `Providers/Anthropic/Maps/*`.
//!
//! Structurally distinct from OpenAI's mapping in two ways: tool results are a
//! `user`-role content block (`tool_result`) rather than a separate `tool` role, and
//! `max_tokens` is mandatory on every request.

use serde_json::json;

use crate::schema::to_json_schema;
use crate::text::request::{TextRequest, ToolChoice};
use crate::text::response::Step;
use crate::tool::Tool;
use crate::value_objects::{FinishReason, MediaPart, Message, Meta, ToolCall, ToolOutcome, Usage};

use super::wire::{ContentBlock, MessageParam, MessagesRequest, MessagesResponse, MessagesTool};

const DEFAULT_MAX_TOKENS: u32 = 4096;

pub fn build_request(request: &TextRequest) -> MessagesRequest {
    let system = if request.system_prompts.is_empty() {
        None
    } else {
        Some(request.system_prompts.join("\n\n"))
    };

    let mut messages = Vec::new();
    for message in &request.messages {
        push_message(&mut messages, message);
    }

    MessagesRequest {
        model: request.model.clone(),
        max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        system,
        messages,
        temperature: request.temperature,
        top_p: request.top_p,
        tools: request
            .tools
            .iter()
            .map(|tool| to_wire_tool(tool.as_ref()))
            .collect(),
        tool_choice: to_wire_tool_choice(&request.tool_choice, request.tools.is_empty()),
    }
}

fn push_message(messages: &mut Vec<MessageParam>, message: &Message) {
    match message {
        // Anthropic has no per-message system role; fold it into the top-level
        // `system` field instead (already done in `build_request`). A `SystemMessage`
        // appearing mid-conversation is treated as a user turn so it isn't dropped.
        Message::System(system) => messages.push(MessageParam {
            role: "user".to_string(),
            content: vec![ContentBlock::Text {
                text: system.content.clone(),
            }],
        }),
        Message::User(user) => {
            let content = user
                .content
                .iter()
                .filter_map(|part| match part {
                    MediaPart::Text(text) => Some(ContentBlock::Text { text: text.clone() }),
                    _ => None,
                })
                .collect();
            messages.push(MessageParam {
                role: "user".to_string(),
                content,
            });
        }
        Message::Assistant(assistant) => {
            let mut content = Vec::new();
            if let Some(text) = &assistant.content {
                content.push(ContentBlock::Text { text: text.clone() });
            }
            for call in &assistant.tool_calls {
                content.push(ContentBlock::ToolUse {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input: call.arguments.clone(),
                });
            }
            messages.push(MessageParam {
                role: "assistant".to_string(),
                content,
            });
        }
        Message::ToolResult(tool_result) => {
            let content = tool_result
                .tool_results
                .iter()
                .map(|result| match &result.result {
                    ToolOutcome::Output(output) => ContentBlock::ToolResult {
                        tool_use_id: result.tool_call_id.clone(),
                        content: output.content.clone(),
                        is_error: None,
                    },
                    ToolOutcome::Error(message) => ContentBlock::ToolResult {
                        tool_use_id: result.tool_call_id.clone(),
                        content: message.clone(),
                        is_error: Some(true),
                    },
                })
                .collect();
            messages.push(MessageParam {
                role: "user".to_string(),
                content,
            });
        }
    }
}

fn to_wire_tool(tool: &dyn Tool) -> MessagesTool {
    MessagesTool {
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        input_schema: to_json_schema(&crate::schema::Schema::Object(tool.parameters().clone())),
    }
}

fn to_wire_tool_choice(choice: &ToolChoice, no_tools: bool) -> Option<serde_json::Value> {
    if no_tools {
        return None;
    }
    match choice {
        ToolChoice::Auto => Some(json!({"type": "auto"})),
        ToolChoice::None => None,
        ToolChoice::Any => Some(json!({"type": "any"})),
        ToolChoice::Tool(name) => Some(json!({"type": "tool", "name": name})),
    }
}

pub fn parse_response(response: MessagesResponse) -> Step {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for block in &response.content {
        match block {
            ContentBlock::Text { text: block_text } => text.push_str(block_text),
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: input.clone(),
            }),
            ContentBlock::ToolResult { .. } => {}
        }
    }

    let finish_reason = match response.stop_reason.as_deref() {
        Some("end_turn") | Some("stop_sequence") => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        _ => FinishReason::Other,
    };

    Step {
        text: if text.is_empty() { None } else { Some(text) },
        tool_calls,
        finish_reason,
        usage: Usage {
            prompt_tokens: response.usage.input_tokens,
            completion_tokens: response.usage.output_tokens,
            cache_write_tokens: response.usage.cache_creation_input_tokens,
            cache_read_tokens: response.usage.cache_read_input_tokens,
            thought_tokens: None,
        },
        meta: Meta {
            id: Some(response.id),
            model: Some(response.model),
            rate_limits: Vec::new(),
        },
    }
}
