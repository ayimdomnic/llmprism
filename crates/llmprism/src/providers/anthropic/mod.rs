//! The Anthropic provider, talking to the Messages API
//! (`api.anthropic.com/v1/messages`). Enable with the `anthropic` Cargo feature.

mod config;
mod maps;
mod wire;

use std::collections::HashMap;

use async_stream::try_stream;
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::stream::{BoxStream, StreamExt};
use reqwest_middleware::ClientWithMiddleware;
use serde_json::Value;

use crate::client::{build_http_client, merge_provider_options, ErrorMapper};
use crate::error::Error;
use crate::provider::Provider;
use crate::stream_event::StreamEvent;
use crate::structured::{StructuredRequest, StructuredResponse, StructuredStreamEvent};
use crate::text::{Step, TextRequest};
use crate::value_objects::{Meta, ToolCall, Usage};

use self::config::{ANTHROPIC_VERSION, DEFAULT_BASE_URL};
use self::wire::{MessagesResponse, StreamContentBlockStart, StreamDelta, StreamEventPayload};

/// A [`Provider`] backed by Anthropic's Messages API.
///
/// # Example
///
/// ```no_run
/// use llmprism::providers::anthropic::AnthropicProvider;
/// use llmprism::Registry;
///
/// let mut registry = Registry::new();
/// registry.register(
///     "anthropic",
///     AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY").unwrap()),
/// );
/// ```
///
/// If you're happy reading the API key from `ANTHROPIC_API_KEY` yourself, you
/// likely don't need to construct this directly -- see
/// [`Registry::from_env`](crate::Registry::from_env).
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    version: String,
    client: ClientWithMiddleware,
}

impl AnthropicProvider {
    /// Creates a provider that talks to the real Anthropic API using `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Creates a provider pointed at a different base URL -- useful for a
    /// proxy or gateway that mirrors the Messages API shape rather than
    /// `api.anthropic.com` itself, and for testing against a mock server
    /// (see this crate's own `tests/`, e.g. `structured_streaming.rs`).
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            version: ANTHROPIC_VERSION.to_string(),
            client: build_http_client(),
        }
    }

    /// Replaces the underlying HTTP client -- an escape hatch for anything
    /// [`build_http_client`] doesn't cover
    /// (a request timeout, a proxy, a custom retry policy). See
    /// `OpenAiProvider::with_client`'s docs (not linked here since that type
    /// only exists with the `openai` feature enabled) for a full example;
    /// this works the same way.
    pub fn with_client(mut self, client: ClientWithMiddleware) -> Self {
        self.client = client;
        self
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn text_step(&self, request: &TextRequest) -> Result<Step, Error> {
        let wire_request = maps::build_request(request);
        let body = merge_provider_options(&wire_request, &request.provider_options)?;

        let http_response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version)
            .json(&body)
            .send()
            .await?;

        let status = http_response.status();
        let headers = http_response.headers().clone();
        let body_text = http_response.text().await?;

        if !status.is_success() {
            let mapper = ErrorMapper {
                provider: self.name(),
            };
            return Err(mapper.map_error_response(status, &headers, &body_text));
        }

        let wire_response: MessagesResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        Ok(maps::parse_response(wire_response))
    }

    async fn stream_text_once(
        &self,
        request: &TextRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let mut wire_request = maps::build_request(request);
        wire_request.stream = Some(true);
        let body = merge_provider_options(&wire_request, &request.provider_options)?;

        let http_response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version)
            .json(&body)
            .send()
            .await?;

        let status = http_response.status();
        if !status.is_success() {
            let headers = http_response.headers().clone();
            let body_text = http_response.text().await?;
            let mapper = ErrorMapper {
                provider: self.name(),
            };
            return Err(mapper.map_error_response(status, &headers, &body_text));
        }

        let provider_name = self.name().to_string();
        let mut events = http_response.bytes_stream().eventsource();

        let stream = try_stream! {
            let mut blocks: HashMap<usize, BlockState> = HashMap::new();
            let mut finish_reason = None;
            let mut usage = Usage::default();

            while let Some(event) = events.next().await {
                let event = event.map_err(|e| Error::StreamDecode {
                    provider: provider_name.clone(),
                    message: e.to_string(),
                })?;

                let data = event.data.trim();
                if data.is_empty() {
                    continue;
                }

                let payload: StreamEventPayload =
                    serde_json::from_str(data).map_err(|e| Error::StreamDecode {
                        provider: provider_name.clone(),
                        message: e.to_string(),
                    })?;

                match payload {
                    StreamEventPayload::MessageStart { message } => {
                        usage = maps::map_usage(message.usage);
                        yield StreamEvent::StreamStart {
                            meta: Meta {
                                id: Some(message.id),
                                model: Some(message.model),
                                rate_limits: Vec::new(),
                            },
                        };
                    }
                    StreamEventPayload::ContentBlockStart { index, content_block } => match content_block {
                        StreamContentBlockStart::Text { text } => {
                            blocks.insert(index, BlockState::Text);
                            if !text.is_empty() {
                                yield StreamEvent::TextDelta { text };
                            }
                        }
                        StreamContentBlockStart::ToolUse { id, name } => {
                            blocks.insert(
                                index,
                                BlockState::ToolUse {
                                    id,
                                    name,
                                    arguments: String::new(),
                                },
                            );
                        }
                        StreamContentBlockStart::Other => {
                            blocks.insert(index, BlockState::Other);
                        }
                    },
                    StreamEventPayload::ContentBlockDelta { index, delta } => match delta {
                        StreamDelta::TextDelta { text } => {
                            yield StreamEvent::TextDelta { text };
                        }
                        StreamDelta::InputJsonDelta { partial_json } => {
                            if let Some(BlockState::ToolUse { id, name, arguments }) =
                                blocks.get_mut(&index)
                            {
                                arguments.push_str(&partial_json);
                                yield StreamEvent::ToolCallDelta {
                                    index,
                                    id: Some(id.clone()),
                                    name: Some(name.clone()),
                                    arguments_delta: partial_json,
                                };
                            }
                        }
                        StreamDelta::Other => {}
                    },
                    StreamEventPayload::ContentBlockStop { index } => {
                        if let Some(BlockState::ToolUse { id, name, arguments }) = blocks.remove(&index)
                        {
                            yield StreamEvent::ToolCall(ToolCall {
                                id,
                                name,
                                arguments: serde_json::from_str(&arguments).unwrap_or(Value::Null),
                            });
                        }
                    }
                    StreamEventPayload::MessageDelta { delta, usage: delta_usage } => {
                        finish_reason = Some(maps::map_finish_reason(delta.stop_reason.as_deref()));
                        if let Some(delta_usage) = delta_usage {
                            usage.completion_tokens = delta_usage.output_tokens;
                        }
                    }
                    StreamEventPayload::MessageStop => break,
                    StreamEventPayload::Ping | StreamEventPayload::Unknown => {}
                    StreamEventPayload::Error { error } => {
                        Err(Error::Provider {
                            provider: provider_name.clone(),
                            status: 0,
                            kind: error.kind,
                            message: error.message,
                        })?;
                    }
                }
            }

            yield StreamEvent::StepFinish {
                usage,
                finish_reason: finish_reason.unwrap_or(crate::value_objects::FinishReason::Stop),
            };
        };

        Ok(stream.boxed())
    }

    async fn structured(&self, request: StructuredRequest) -> Result<StructuredResponse, Error> {
        let wire_request = maps::build_structured_request(&request);
        let body = merge_provider_options(&wire_request, &request.provider_options)?;

        let http_response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version)
            .json(&body)
            .send()
            .await?;

        let status = http_response.status();
        let headers = http_response.headers().clone();
        let body_text = http_response.text().await?;

        if !status.is_success() {
            let mapper = ErrorMapper {
                provider: self.name(),
            };
            return Err(mapper.map_error_response(status, &headers, &body_text));
        }

        let wire_response: MessagesResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        maps::parse_structured_response(wire_response, self.name())
    }

    async fn stream_structured_once(
        &self,
        request: &StructuredRequest,
    ) -> Result<BoxStream<'static, Result<StructuredStreamEvent, Error>>, Error> {
        // Reuses the same forced-tool-call request `structured` sends
        // (`build_structured_request`) -- the schema-constrained output
        // arrives as a `tool_use` content block's `input_json_delta` events,
        // read with the same event dispatch `stream_text_once` already has
        // for `ContentBlockDelta`/`ContentBlockStop`, just narrowed to the
        // one tool-use block a forced call always produces exactly one of.
        let mut wire_request = maps::build_structured_request(request);
        wire_request.stream = Some(true);
        let body = merge_provider_options(&wire_request, &request.provider_options)?;

        let http_response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version)
            .json(&body)
            .send()
            .await?;

        let status = http_response.status();
        if !status.is_success() {
            let headers = http_response.headers().clone();
            let body_text = http_response.text().await?;
            let mapper = ErrorMapper {
                provider: self.name(),
            };
            return Err(mapper.map_error_response(status, &headers, &body_text));
        }

        let provider_name = self.name().to_string();
        let mut events = http_response.bytes_stream().eventsource();

        let stream = try_stream! {
            let mut meta = Meta::default();
            let mut stop_reason: Option<String> = None;
            let mut usage = Usage::default();
            // Text seen outside the tool-use block, kept only in case the
            // model doesn't call the forced tool at all (a refusal, most
            // likely) -- then it's exactly what `Error::StructuredDecode`'s
            // `raw` should carry, the same as the non-streaming path.
            let mut text = String::new();
            let mut arguments = String::new();
            let mut in_tool_use = false;

            while let Some(event) = events.next().await {
                let event = event.map_err(|e| Error::StreamDecode {
                    provider: provider_name.clone(),
                    message: e.to_string(),
                })?;

                let data = event.data.trim();
                if data.is_empty() {
                    continue;
                }

                let payload: StreamEventPayload =
                    serde_json::from_str(data).map_err(|e| Error::StreamDecode {
                        provider: provider_name.clone(),
                        message: e.to_string(),
                    })?;

                match payload {
                    StreamEventPayload::MessageStart { message } => {
                        usage = maps::map_usage(message.usage);
                        meta = Meta {
                            id: Some(message.id),
                            model: Some(message.model),
                            rate_limits: Vec::new(),
                        };
                    }
                    StreamEventPayload::ContentBlockStart { content_block, .. } => {
                        in_tool_use = matches!(content_block, StreamContentBlockStart::ToolUse { .. });
                    }
                    StreamEventPayload::ContentBlockDelta { delta, .. } => match delta {
                        StreamDelta::InputJsonDelta { partial_json } if in_tool_use => {
                            arguments.push_str(&partial_json);
                            // A partial parse failing isn't a stream error --
                            // `fix_json` is designed to always succeed, but a
                            // defensive skip here just means this particular
                            // chunk contributes no preview, not that
                            // anything is wrong.
                            let fixed = partial_json_fixer::fix_json(&arguments);
                            if let Ok(data) = serde_json::from_str(&fixed) {
                                yield StructuredStreamEvent::PartialObject { data };
                            }
                        }
                        StreamDelta::TextDelta { text: delta } => text.push_str(&delta),
                        StreamDelta::InputJsonDelta { .. } | StreamDelta::Other => {}
                    },
                    StreamEventPayload::ContentBlockStop { .. } => {
                        in_tool_use = false;
                    }
                    StreamEventPayload::MessageDelta { delta, usage: delta_usage } => {
                        stop_reason = delta.stop_reason;
                        if let Some(delta_usage) = delta_usage {
                            usage.completion_tokens = delta_usage.output_tokens;
                        }
                    }
                    StreamEventPayload::MessageStop => break,
                    StreamEventPayload::Ping | StreamEventPayload::Unknown => {}
                    StreamEventPayload::Error { error } => {
                        Err(Error::Provider {
                            provider: provider_name.clone(),
                            status: 0,
                            kind: error.kind,
                            message: error.message,
                        })?;
                    }
                }
            }

            let finish_reason = maps::map_structured_finish_reason(stop_reason.as_deref());

            if arguments.is_empty() {
                Err(Error::StructuredDecode {
                    provider: provider_name.clone(),
                    message: "response contained no tool_use block with the structured output".to_string(),
                    context: Box::new(crate::error::StructuredDecodeContext {
                        raw: text,
                        finish_reason,
                        usage,
                        meta: meta.clone(),
                    }),
                })?;
            }

            let data: Value = serde_json::from_str(&arguments).map_err(|e| Error::StructuredDecode {
                provider: provider_name.clone(),
                message: e.to_string(),
                context: Box::new(crate::error::StructuredDecodeContext {
                    raw: arguments.clone(),
                    finish_reason,
                    usage,
                    meta: meta.clone(),
                }),
            })?;

            yield StructuredStreamEvent::End {
                response: StructuredResponse { data, finish_reason, usage, meta },
            };
        };

        Ok(stream.boxed())
    }
}

/// Tracks, per content-block `index`, enough state to translate that block's
/// `content_block_delta`/`content_block_stop` events into [`StreamEvent`]s.
/// Anthropic interleaves multiple blocks by index (e.g. text, then a tool call,
/// then more text), so this can't just be a single running buffer.
enum BlockState {
    Text,
    ToolUse {
        id: String,
        name: String,
        arguments: String,
    },
    /// A content block type this crate doesn't translate yet (thinking,
    /// citations, server-side tool results, ...) -- its deltas are received and
    /// safely ignored rather than causing a decode error.
    Other,
}
