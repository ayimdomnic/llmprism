//! Translates between llmprism's provider-agnostic value objects and Gemini's
//! `generateContent` wire format -- ported from Prism's `Providers/Gemini/Maps/*`.
//!
//! Structurally distinct from both OpenAI's and Anthropic's mapping in a few
//! ways: there's no per-message system role (folded into a top-level
//! `systemInstruction`, the same idea as Anthropic's `system` field),
//! assistant turns use the role `"model"` rather than `"assistant"`, and tool
//! calls/results are ordinary parts within a turn (`functionCall`/
//! `functionResponse`) rather than a dedicated message role or content-block
//! type. Gemini's wire format also has no call-id concept the way OpenAI and
//! Anthropic do -- a `functionCall` part is just a name and arguments, and
//! the matching `functionResponse` sent back is correlated by *name*, not by
//! id. This crate's [`ToolCall`]/[`ToolResult`] types require an id, so
//! [`parse_response`] synthesizes one (`call_0`, `call_1`, ...) purely for
//! that internal bookkeeping -- it never goes over the wire.

use serde_json::{json, Value};

use crate::error::Error;
use crate::schema::{to_json_schema, Schema};
use crate::structured::{StructuredRequest, StructuredResponse};
use crate::text::request::{TextRequest, ToolChoice};
use crate::text::response::Step;
use crate::tool::Tool;
use crate::value_objects::{FinishReason, MediaPart, Message, Meta, ToolCall, ToolOutcome, Usage};

use super::wire::{
    Content, FunctionCallPart, FunctionCallingConfig, FunctionDeclaration, FunctionResponsePart,
    GenerateContentRequest, GenerateContentResponse, GenerationConfig, Part, SystemInstruction,
    ToolConfig, ToolDeclaration, UsageMetadata,
};

pub fn build_request(request: &TextRequest) -> GenerateContentRequest {
    let mut contents = Vec::new();
    for message in &request.messages {
        push_message(&mut contents, message);
    }

    GenerateContentRequest {
        contents,
        system_instruction: build_system_instruction(&request.system_prompts),
        generation_config: Some(GenerationConfig {
            max_output_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            response_mime_type: None,
            response_json_schema: None,
        }),
        tools: build_tools(&request.tools),
        tool_config: to_wire_tool_config(&request.tool_choice, request.tools.is_empty()),
    }
}

/// Builds a `generateContent` request for a structured-output call, using
/// Gemini's native `generationConfig.responseJsonSchema` -- like OpenAI's
/// `response_format`, the API enforces the schema server-side, so (unlike
/// Anthropic's forced-tool-call strategy) no tool is involved here at all.
pub fn build_structured_request(request: &StructuredRequest) -> GenerateContentRequest {
    let mut contents = Vec::new();
    for message in &request.messages {
        push_message(&mut contents, message);
    }

    GenerateContentRequest {
        contents,
        system_instruction: build_system_instruction(&request.system_prompts),
        generation_config: Some(GenerationConfig {
            max_output_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            response_mime_type: Some("application/json".to_string()),
            response_json_schema: Some(to_json_schema(&Schema::Object(request.schema.clone()))),
        }),
        tools: Vec::new(),
        tool_config: None,
    }
}

fn build_system_instruction(system_prompts: &[String]) -> Option<SystemInstruction> {
    if system_prompts.is_empty() {
        None
    } else {
        Some(SystemInstruction {
            parts: vec![Part::Text {
                text: system_prompts.join("\n\n"),
            }],
        })
    }
}

fn build_tools(tools: &[std::sync::Arc<dyn Tool>]) -> Vec<ToolDeclaration> {
    if tools.is_empty() {
        return Vec::new();
    }
    vec![ToolDeclaration {
        function_declarations: tools
            .iter()
            .map(|tool| to_wire_tool(tool.as_ref()))
            .collect(),
    }]
}

fn to_wire_tool(tool: &dyn Tool) -> FunctionDeclaration {
    FunctionDeclaration {
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        parameters_json_schema: to_json_schema(&Schema::Object(tool.parameters().clone())),
    }
}

fn to_wire_tool_config(choice: &ToolChoice, no_tools: bool) -> Option<ToolConfig> {
    if no_tools {
        return None;
    }
    let (mode, allowed_function_names) = match choice {
        ToolChoice::Auto => ("AUTO", None),
        ToolChoice::None => ("NONE", None),
        ToolChoice::Any => ("ANY", None),
        ToolChoice::Tool(name) => ("ANY", Some(vec![name.clone()])),
    };
    Some(ToolConfig {
        function_calling_config: FunctionCallingConfig {
            mode,
            allowed_function_names,
        },
    })
}

fn push_message(contents: &mut Vec<Content>, message: &Message) {
    match message {
        // Gemini has no per-message system role (like Anthropic); a
        // `SystemMessage` appearing mid-conversation, rather than up front via
        // `system_prompts`, is folded into a "user" turn instead of being
        // dropped.
        Message::System(system) => contents.push(Content {
            role: "user".to_string(),
            parts: vec![Part::Text {
                text: system.content.clone(),
            }],
        }),
        Message::User(user) => {
            let parts = user
                .content
                .iter()
                .filter_map(|part| match part {
                    MediaPart::Text(text) => Some(Part::Text { text: text.clone() }),
                    _ => None,
                })
                .collect();
            contents.push(Content {
                role: "user".to_string(),
                parts,
            });
        }
        Message::Assistant(assistant) => {
            let mut parts = Vec::new();
            if let Some(text) = &assistant.content {
                parts.push(Part::Text { text: text.clone() });
            }
            for call in &assistant.tool_calls {
                parts.push(Part::FunctionCall {
                    function_call: FunctionCallPart {
                        name: call.name.clone(),
                        args: call.arguments.clone(),
                    },
                });
            }
            contents.push(Content {
                role: "model".to_string(),
                parts,
            });
        }
        Message::ToolResult(tool_result) => {
            let parts = tool_result
                .tool_results
                .iter()
                .map(|result| {
                    let response = match &result.result {
                        ToolOutcome::Output(output) => json!({ "content": output.content }),
                        ToolOutcome::Error(message) => json!({ "error": message }),
                    };
                    Part::FunctionResponse {
                        function_response: FunctionResponsePart {
                            name: result.tool_name.clone(),
                            response,
                        },
                    }
                })
                .collect();
            contents.push(Content {
                role: "user".to_string(),
                parts,
            });
        }
    }
}

pub fn parse_response(
    response: GenerateContentResponse,
    provider_name: &str,
) -> Result<Step, Error> {
    let candidate = first_candidate(&response).map_err(|message| Error::Provider {
        provider: provider_name.to_string(),
        status: 0,
        kind: Some("blocked".to_string()),
        message,
    })?;

    let (text, tool_calls) = read_parts(candidate.content.clone());

    let finish_reason = if tool_calls.is_empty() {
        map_finish_reason(candidate.finish_reason.as_deref())
    } else {
        FinishReason::ToolCalls
    };

    Ok(Step {
        text,
        tool_calls,
        finish_reason,
        usage: response.usage_metadata.map(map_usage).unwrap_or_default(),
        meta: Meta {
            id: response.response_id,
            model: response.model_version,
            rate_limits: Vec::new(),
        },
    })
}

/// Parses a `generateContent` response returned for a structured-output
/// request -- the counterpart to [`build_structured_request`]'s native
/// `responseJsonSchema`. Like OpenAI's `response_format` (and unlike
/// Anthropic's forced tool call), the reply text *is* a JSON string matching
/// the schema, so it still needs a second, fallible parse pass here.
pub fn parse_structured_response(
    response: GenerateContentResponse,
    provider_name: &str,
) -> Result<StructuredResponse, Error> {
    let candidate = first_candidate(&response).map_err(|message| Error::StructuredDecode {
        provider: provider_name.to_string(),
        message,
    })?;

    let finish_reason = map_finish_reason(candidate.finish_reason.as_deref());
    let (text, _) = read_parts(candidate.content.clone());

    let text = text.ok_or_else(|| Error::StructuredDecode {
        provider: provider_name.to_string(),
        message: "response contained no text part with the structured output".to_string(),
    })?;

    let data: Value = serde_json::from_str(&text).map_err(|e| Error::StructuredDecode {
        provider: provider_name.to_string(),
        message: e.to_string(),
    })?;

    Ok(StructuredResponse {
        data,
        finish_reason,
        usage: response.usage_metadata.map(map_usage).unwrap_or_default(),
        meta: Meta {
            id: response.response_id,
            model: response.model_version,
            rate_limits: Vec::new(),
        },
    })
}

/// Gemini returns an empty `candidates` array (with a reason in
/// `promptFeedback` instead) when it declines to generate anything at all for
/// the request -- most commonly a safety block. Both response parsers treat
/// that the same way: as a failure carrying whatever reason Gemini gave, not
/// a response with no content.
fn first_candidate(response: &GenerateContentResponse) -> Result<&super::wire::Candidate, String> {
    response.candidates.first().ok_or_else(|| {
        response
            .prompt_feedback
            .as_ref()
            .and_then(|feedback| feedback.block_reason.clone())
            .unwrap_or_else(|| "no candidates returned".to_string())
    })
}

fn read_parts(content: Option<Content>) -> (Option<String>, Vec<ToolCall>) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    if let Some(content) = content {
        for part in content.parts {
            match part {
                Part::Text { text: part_text } => text.push_str(&part_text),
                Part::FunctionCall { function_call } => tool_calls.push(ToolCall {
                    id: format!("call_{}", tool_calls.len()),
                    name: function_call.name,
                    arguments: function_call.args,
                }),
                Part::FunctionResponse { .. } | Part::Other(_) => {}
            }
        }
    }

    (if text.is_empty() { None } else { Some(text) }, tool_calls)
}

/// Maps a candidate's `finishReason` to this crate's [`FinishReason`]. Gemini
/// has no distinct reason for "the model wants to call a tool" -- a
/// function-calling turn is typically still reported as `STOP` -- so callers
/// only fall back to this when [`read_parts`] found no tool calls; see
/// [`parse_response`].
pub(crate) fn map_finish_reason(finish_reason: Option<&str>) -> FinishReason {
    match finish_reason {
        None | Some("STOP") => FinishReason::Stop,
        Some("MAX_TOKENS") => FinishReason::Length,
        Some("SAFETY") | Some("RECITATION") | Some("BLOCKLIST") | Some("PROHIBITED_CONTENT") => {
            FinishReason::ContentFilter
        }
        Some(_) => FinishReason::Other,
    }
}

/// Maps a `usageMetadata` object to this crate's [`Usage`]. Shared for the
/// same reason as [`map_finish_reason`] -- streaming chunks report usage in
/// the identical shape.
pub(crate) fn map_usage(usage: UsageMetadata) -> Usage {
    Usage {
        prompt_tokens: usage.prompt_token_count,
        completion_tokens: usage.candidates_token_count,
        cache_write_tokens: None,
        cache_read_tokens: usage.cached_content_token_count,
        thought_tokens: None,
    }
}
