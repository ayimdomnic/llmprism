//! Translates between llmprism's provider-agnostic value objects and Anthropic's
//! Messages API wire format -- ported from Prism's `Providers/Anthropic/Maps/*`.
//!
//! Structurally distinct from OpenAI's mapping in two ways: tool results are a
//! `user`-role content block (`tool_result`) rather than a separate `tool` role, and
//! `max_tokens` is mandatory on every request.

use serde_json::{json, Value};

use crate::error::Error;
use crate::schema::{to_json_schema, Schema};
use crate::structured::{StructuredRequest, StructuredResponse};
use crate::text::request::{TextRequest, ToolChoice};
use crate::text::response::Step;
use crate::tool::Tool;
use crate::value_objects::{
    FinishReason, Media, MediaData, MediaPart, Message, Meta, ToolCall, ToolOutcome, Usage,
};

use super::wire::{
    CacheControl, ContentBlock, MediaSource, MessageParam, MessagesRequest, MessagesResponse,
    MessagesTool, SystemBlock, SystemPrompt, ThinkingConfig,
};

const DEFAULT_MAX_TOKENS: u32 = 4096;

pub fn build_request(request: &TextRequest) -> MessagesRequest {
    let system = build_system(&request.system_prompts, request.cache_system_prompt);

    let mut messages = Vec::new();
    for message in &request.messages {
        push_message(&mut messages, message);
    }

    // Anthropic rejects a request where `max_tokens` isn't strictly greater
    // than `thinking.budget_tokens` -- raise it here rather than leaving a
    // caller who only called `with_thinking_budget` (without also raising
    // `with_max_tokens` to match) to hit an API error that doesn't explain
    // the actual constraint.
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let max_tokens = match request.thinking_budget {
        Some(budget_tokens) => max_tokens.max(budget_tokens + 1),
        None => max_tokens,
    };

    MessagesRequest {
        model: request.model.clone(),
        max_tokens,
        system,
        messages,
        temperature: request.temperature,
        top_p: request.top_p,
        tools: build_tools(&request.tools, &request.provider_tools),
        tool_choice: to_wire_tool_choice(
            &request.tool_choice,
            request.tools.is_empty() && request.provider_tools.is_empty(),
        ),
        stream: None,
        thinking: request.thinking_budget.map(ThinkingConfig::enabled),
        stop_sequences: request.stop_sequences.clone(),
    }
}

/// Builds a Messages API request for a structured-output call, using
/// Anthropic's *forced tool call* strategy: since Anthropic has no native
/// structured-output response format the way OpenAI does, this defines a
/// single synthetic tool shaped exactly like the requested schema, forces the
/// model to call it (`tool_choice: {"type": "tool", "name": ...}`), and
/// [`parse_structured_response`] reads that tool call's `input` back out as
/// the result -- the model never actually "calls a tool" from the
/// application's point of view, this is purely a way to get schema-shaped
/// output.
pub fn build_structured_request(request: &StructuredRequest) -> MessagesRequest {
    let system = build_system(&request.system_prompts, request.cache_system_prompt);

    let mut messages = Vec::new();
    for message in &request.messages {
        push_message(&mut messages, message);
    }

    let tool_name = request.schema.name.clone();
    let description = request
        .schema
        .description
        .clone()
        .unwrap_or_else(|| "Structured output matching the requested schema.".to_string());

    MessagesRequest {
        model: request.model.clone(),
        max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        system,
        messages,
        temperature: request.temperature,
        top_p: request.top_p,
        tools: vec![serde_json::to_value(MessagesTool {
            name: tool_name.clone(),
            description,
            input_schema: to_json_schema(&Schema::Object(request.schema.clone())),
        })
        .expect("MessagesTool always serializes to JSON")],
        tool_choice: Some(json!({"type": "tool", "name": tool_name})),
        stream: None,
        // Never set here: Anthropic rejects combining extended thinking with
        // a forced tool call, which is exactly the strategy this function
        // uses to get schema-shaped output -- see
        // `TextRequest::thinking_budget`'s doc comment (there's no
        // `thinking_budget` field on `StructuredRequest` at all, for the
        // same reason).
        thinking: None,
        // `StructuredRequest` has no `stop_sequences` field either -- see
        // its own doc comment for why (a stop string can truncate otherwise
        // -valid JSON).
        stop_sequences: Vec::new(),
    }
}

/// Builds the `system` field: `None` if there's nothing to send, a plain
/// string in the common case, or -- when `cache_system_prompt` is set -- a
/// single text block carrying an `"ephemeral"` [`CacheControl`] breakpoint
/// instead, since Anthropic can only attach `cache_control` to a content
/// block, not a plain string. See [`TextRequest::cache_system_prompt`].
fn build_system(system_prompts: &[String], cache_system_prompt: bool) -> Option<SystemPrompt> {
    if system_prompts.is_empty() {
        return None;
    }

    let text = system_prompts.join("\n\n");

    if cache_system_prompt {
        Some(SystemPrompt::Blocks(vec![SystemBlock {
            kind: "text",
            text,
            cache_control: Some(CacheControl::ephemeral()),
        }]))
    } else {
        Some(SystemPrompt::Text(text))
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
                    MediaPart::Image(media) => Some(ContentBlock::Image {
                        source: to_media_source(media, "image/png"),
                    }),
                    MediaPart::Document(media) => Some(ContentBlock::Document {
                        source: to_media_source(media, "application/pdf"),
                    }),
                    // Anthropic's Messages API has no audio/video content
                    // block at all -- unlike Image/Document, there's no wire
                    // shape to translate these into.
                    MediaPart::Audio(_) | MediaPart::Video(_) => None,
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

/// Translates this crate's provider-agnostic [`Media`] into Anthropic's own
/// `source` union. `default_mime_type` fills in when `media.mime_type` wasn't
/// set -- Anthropic requires a `media_type` on a base64 source (there's
/// nothing to infer it from the way a URL's extension might hint at one), so
/// a value has to come from somewhere; the caller picks a sensible default
/// for the content type it's building (e.g. `"image/png"` for
/// [`ContentBlock::Image`], `"application/pdf"` for
/// [`ContentBlock::Document`]).
fn to_media_source(media: &Media, default_mime_type: &str) -> MediaSource {
    match &media.data {
        MediaData::Url(url) => MediaSource::Url { url: url.clone() },
        MediaData::Base64(data) => MediaSource::Base64 {
            media_type: media
                .mime_type
                .clone()
                .unwrap_or_else(|| default_mime_type.to_string()),
            data: data.clone(),
        },
    }
}

/// Builds the combined `tools` array: this crate's own [`Tool`]s (each
/// serialized to its [`MessagesTool`] shape) followed by any provider-native
/// tools passed through as-is -- both live in the one array Anthropic reads
/// tool declarations from, so there's no separate field for the latter.
fn build_tools(tools: &[std::sync::Arc<dyn Tool>], provider_tools: &[Value]) -> Vec<Value> {
    let mut wire_tools: Vec<Value> = tools
        .iter()
        .map(|tool| {
            serde_json::to_value(to_wire_tool(tool.as_ref()))
                .expect("MessagesTool always serializes to JSON")
        })
        .collect();
    wire_tools.extend(provider_tools.iter().cloned());
    wire_tools
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
            // Image/Document blocks are something *this crate* only ever
            // sends (as part of a user turn), never something Anthropic
            // sends back as part of its own reply -- but the match must
            // still be exhaustive.
            ContentBlock::Image { .. }
            | ContentBlock::Document { .. }
            | ContentBlock::ToolResult { .. }
            | ContentBlock::Other => {}
        }
    }

    Step {
        text: if text.is_empty() { None } else { Some(text) },
        tool_calls,
        finish_reason: map_finish_reason(response.stop_reason.as_deref()),
        usage: map_usage(response.usage),
        meta: Meta {
            id: Some(response.id),
            model: Some(response.model),
            rate_limits: Vec::new(),
        },
    }
}

/// Parses a Messages API response returned for a structured-output request --
/// the counterpart to [`build_structured_request`]'s forced tool call. The
/// forced tool's `input` *is* the structured data, already decoded JSON (no
/// second parse pass needed the way OpenAI's string-encoded `content` needs
/// one).
pub fn parse_structured_response(
    response: MessagesResponse,
    provider_name: &str,
) -> Result<StructuredResponse, Error> {
    let tool_use = response.content.iter().find_map(|block| match block {
        ContentBlock::ToolUse { input, .. } => Some(input.clone()),
        _ => None,
    });

    let finish_reason = map_structured_finish_reason(response.stop_reason.as_deref());
    let usage = map_usage(response.usage);
    let meta = Meta {
        id: Some(response.id),
        model: Some(response.model),
        rate_limits: Vec::new(),
    };

    let data = tool_use.ok_or_else(|| {
        // The model didn't call the forced tool at all -- most likely it
        // replied with plain text instead (a refusal, or an explanation of
        // why it couldn't comply). That text, when present, is exactly what
        // a `RepairStrategy` needs to work with, so it's worth surfacing as
        // `raw` even though this isn't the "malformed JSON" case OpenAI's
        // equivalent handles.
        let raw = response
            .content
            .iter()
            .find_map(|block| match block {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();

        Error::StructuredDecode {
            provider: provider_name.to_string(),
            message: "response contained no tool_use block with the structured output".to_string(),
            context: Box::new(crate::error::StructuredDecodeContext {
                raw,
                finish_reason,
                usage,
                meta: meta.clone(),
            }),
        }
    })?;

    Ok(StructuredResponse {
        data,
        finish_reason,
        usage,
        meta,
    })
}

/// Maps a Messages API `stop_reason` to this crate's [`FinishReason`]. Shared
/// between the non-streaming response parser above and the streaming event
/// parser in `mod.rs` (there, it's the `message_delta` event's `delta
/// .stop_reason`) -- both encounter the identical set of strings.
pub(crate) fn map_finish_reason(stop_reason: Option<&str>) -> FinishReason {
    match stop_reason {
        Some("end_turn") | Some("stop_sequence") => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        _ => FinishReason::Other,
    }
}

/// [`map_finish_reason`], but for a forced-tool-call structured request
/// (both [`parse_structured_response`] and `stream_structured_once`'s
/// streaming counterpart in `mod.rs`): a `"tool_use"` stop reason is the
/// *success* case here -- the model complied with the forced call -- so
/// it's reported as a normal `Stop`, not `ToolCalls`, unlike plain text
/// generation where `"tool_use"` genuinely means the model wants to call a
/// tool. Any other stop reason (most notably `"max_tokens"`, meaning the
/// output may be truncated/invalid JSON) is preserved via the normal
/// mapping so callers can still tell something went wrong.
pub(crate) fn map_structured_finish_reason(stop_reason: Option<&str>) -> FinishReason {
    match stop_reason {
        Some("tool_use") => FinishReason::Stop,
        other => map_finish_reason(other),
    }
}

/// Maps a Messages API `usage` object to this crate's [`Usage`]. Shared for the
/// same reason as [`map_finish_reason`] -- streaming's `message_start` event
/// carries this same shape.
pub(crate) fn map_usage(usage: super::wire::MessagesUsage) -> Usage {
    Usage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        cache_write_tokens: usage.cache_creation_input_tokens,
        cache_read_tokens: usage.cache_read_input_tokens,
        thought_tokens: None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::wire::MessagesUsage;
    use super::*;
    use crate::value_objects::UserMessage;

    /// A block type this crate doesn't translate (extended thinking, in this
    /// case) shouldn't stop `parse_response` from reading the text and tool
    /// calls that *are* recognized in the same response.
    #[test]
    fn parse_response_ignores_unrecognized_content_blocks() {
        let response = MessagesResponse {
            id: "msg_1".to_string(),
            model: "claude-3-5-haiku-20241022".to_string(),
            content: vec![
                ContentBlock::Other,
                ContentBlock::Text {
                    text: "The answer is 4.".to_string(),
                },
            ],
            stop_reason: Some("end_turn".to_string()),
            usage: MessagesUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let step = parse_response(response);

        assert_eq!(step.text.as_deref(), Some("The answer is 4."));
        assert!(step.tool_calls.is_empty());
    }

    #[test]
    fn push_message_maps_image_and_document_parts() {
        let user = UserMessage {
            content: vec![
                MediaPart::Text("What's in this file?".to_string()),
                MediaPart::Image(Media {
                    mime_type: None,
                    data: MediaData::Base64("aGVsbG8=".to_string()),
                }),
                MediaPart::Document(Media {
                    mime_type: Some("application/pdf".to_string()),
                    data: MediaData::Url("https://example.com/doc.pdf".to_string()),
                }),
                // No wire shape exists for these in Anthropic's Messages API,
                // so they're expected to be silently dropped rather than
                // producing a malformed request.
                MediaPart::Audio(Media {
                    mime_type: None,
                    data: MediaData::Url("https://example.com/clip.mp3".to_string()),
                }),
            ],
        };

        let mut messages = Vec::new();
        push_message(&mut messages, &Message::User(user));

        assert_eq!(messages.len(), 1);
        let content = &messages[0].content;
        assert_eq!(content.len(), 3);

        assert!(matches!(content[0], ContentBlock::Text { .. }));
        assert!(matches!(
            &content[1],
            ContentBlock::Image {
                source: MediaSource::Base64 { media_type, .. }
            } if media_type == "image/png"
        ));
        assert!(matches!(
            &content[2],
            ContentBlock::Document {
                source: MediaSource::Url { url }
            } if url == "https://example.com/doc.pdf"
        ));
    }

    #[test]
    fn build_system_sends_a_plain_string_when_caching_is_off() {
        let system = build_system(&["Be concise.".to_string()], false);
        assert!(matches!(system, Some(SystemPrompt::Text(text)) if text == "Be concise."));
    }

    #[test]
    fn build_system_sends_a_cached_block_when_caching_is_on() {
        let system = build_system(&["Be concise.".to_string()], true);
        let Some(SystemPrompt::Blocks(blocks)) = system else {
            panic!("expected SystemPrompt::Blocks when cache_system_prompt is true");
        };

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "Be concise.");
        assert!(blocks[0].cache_control.is_some());
    }

    #[test]
    fn build_system_is_none_with_no_system_prompts_even_when_caching_is_on() {
        assert!(build_system(&[], true).is_none());
    }

    #[test]
    fn build_request_sends_no_thinking_config_by_default() {
        let request = TextRequest::new("claude-3-5-haiku-20241022");
        let wire_request = build_request(&request);
        assert!(wire_request.thinking.is_none());
        assert_eq!(wire_request.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn build_request_sends_thinking_config_when_a_budget_is_set() {
        let mut request = TextRequest::new("claude-3-5-haiku-20241022");
        request.thinking_budget = Some(2000);

        let wire_request = build_request(&request);

        assert!(wire_request.thinking.is_some());
        assert_eq!(wire_request.thinking.unwrap().budget_tokens, 2000);
    }

    #[test]
    fn build_request_raises_max_tokens_above_an_unmet_thinking_budget() {
        // The default max_tokens (4096) isn't enough headroom above a
        // larger thinking budget -- Anthropic requires max_tokens to be
        // strictly greater than budget_tokens, so this should get raised
        // rather than sent as-is and rejected by the API.
        let mut request = TextRequest::new("claude-3-5-haiku-20241022");
        request.thinking_budget = Some(8000);

        let wire_request = build_request(&request);

        assert!(wire_request.max_tokens > 8000);
    }

    #[test]
    fn build_request_respects_an_explicit_max_tokens_already_above_the_budget() {
        let mut request = TextRequest::new("claude-3-5-haiku-20241022");
        request.max_tokens = Some(20_000);
        request.thinking_budget = Some(8000);

        let wire_request = build_request(&request);

        assert_eq!(wire_request.max_tokens, 20_000);
    }

    #[test]
    fn build_structured_request_never_sets_thinking_even_if_requested() {
        // `StructuredRequest` has no `thinking_budget` field at all -- this
        // just confirms the wire request's `thinking` stays `None`
        // regardless, since it's forced-tool-call under the hood and
        // Anthropic rejects combining that with extended thinking.
        let schema = crate::schema::ObjectSchema::new("answer");
        let request = StructuredRequest::new("claude-3-5-haiku-20241022", schema);
        let wire_request = build_structured_request(&request);
        assert!(wire_request.thinking.is_none());
    }

    #[test]
    fn build_request_appends_provider_tools_alongside_user_tools() {
        let mut request = TextRequest::new("claude-3-5-haiku-20241022");
        request.provider_tools = vec![json!({
            "type": "web_search_20250305",
            "name": "web_search",
        })];

        let wire_request = build_request(&request);

        assert_eq!(wire_request.tools.len(), 1);
        assert_eq!(wire_request.tools[0]["type"], "web_search_20250305");
    }

    #[test]
    fn build_request_sends_a_tool_choice_when_only_provider_tools_are_present() {
        // Regression check: `to_wire_tool_choice`'s "any tools at all"
        // decision must look at `provider_tools` too, not just user-defined
        // `tools` -- otherwise a request with only a provider tool attached
        // would silently omit `tool_choice` even though `ToolChoice::Auto`
        // is meaningful here (letting the model decide whether to use it).
        let mut request = TextRequest::new("claude-3-5-haiku-20241022");
        request.provider_tools = vec![json!({"type": "web_search_20250305", "name": "web_search"})];

        let wire_request = build_request(&request);

        assert!(wire_request.tool_choice.is_some());
    }

    #[test]
    fn build_request_passes_stop_sequences_through() {
        let mut request = TextRequest::new("claude-3-5-haiku-20241022");
        request.stop_sequences = vec!["STOP".to_string()];

        let wire_request = build_request(&request);

        assert_eq!(wire_request.stop_sequences, vec!["STOP"]);
    }
}
