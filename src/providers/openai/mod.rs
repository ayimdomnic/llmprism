//! The OpenAI provider. Text generation and structured output talk to the
//! Chat Completions API (`api.openai.com/v1/chat/completions`); moderation,
//! embeddings, image generation, and both audio endpoints each talk to their
//! own separate endpoint (`/v1/moderations`, `/v1/embeddings`,
//! `/v1/images/generations`, `/v1/audio/speech`, `/v1/audio/transcriptions`
//! -- see `moderation.rs`/`embeddings.rs`/`images.rs`/`audio.rs`). Enable
//! with the `openai` Cargo feature.

mod audio;
mod config;
mod embeddings;
mod images;
mod maps;
mod moderation;
mod wire;

use std::collections::BTreeMap;

use async_stream::try_stream;
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::stream::{BoxStream, StreamExt};
use reqwest_middleware::ClientWithMiddleware;
use serde_json::{json, Value};

use crate::audio::{
    AudioResponse, SpeechToTextRequest, TextToSpeechRequest, TranscriptionResponse,
};
use crate::client::{build_http_client, ErrorMapper};
use crate::embeddings::{EmbeddingsRequest, EmbeddingsResponse};
use crate::error::Error;
use crate::images::{ImagesRequest, ImagesResponse};
use crate::moderation::{ModerationRequest, ModerationResponse};
use crate::provider::Provider;
use crate::stream_event::StreamEvent;
use crate::structured::{StructuredRequest, StructuredResponse};
use crate::text::{Step, TextRequest};
use crate::value_objects::{FinishReason, Meta, ToolCall, Usage};

use self::config::DEFAULT_BASE_URL;
use self::wire::{ChatResponse, ChatStreamChunk};

/// A [`Provider`] backed by OpenAI's Chat Completions API.
///
/// # Example
///
/// ```no_run
/// use llmprism::providers::openai::OpenAiProvider;
/// use llmprism::Registry;
///
/// let mut registry = Registry::new();
/// registry.register("openai", OpenAiProvider::new(std::env::var("OPENAI_API_KEY").unwrap()));
/// ```
///
/// If you're happy reading the API key from `OPENAI_API_KEY` yourself, you
/// likely don't need to construct this directly -- see
/// [`Registry::from_env`](crate::Registry::from_env).
pub struct OpenAiProvider {
    api_key: String,
    base_url: String,
    client: ClientWithMiddleware,
}

impl OpenAiProvider {
    /// Creates a provider that talks to the real OpenAI API using `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Creates a provider pointed at a different base URL -- useful for an
    /// OpenAI-compatible endpoint (a local model server, a proxy, an
    /// alternative provider that mirrors OpenAI's API shape) rather than
    /// `api.openai.com` itself. `base_url` should not include a trailing
    /// `/chat/completions`; that's appended automatically.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: build_http_client(),
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
        let wire_request = maps::build_request(&request);

        let http_response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&wire_request)
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

        let wire_response: ChatResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        Ok(maps::parse_response(wire_response))
    }

    async fn stream_text_once(
        &self,
        request: TextRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let mut wire_request = maps::build_request(&request);
        wire_request.stream = Some(true);
        // Usage is otherwise omitted entirely from a streamed response; this asks
        // for one extra chunk at the end (empty `choices`, `usage` set) carrying
        // it, so streaming and non-streaming responses report usage the same way.
        wire_request.stream_options = Some(json!({"include_usage": true}));

        let http_response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&wire_request)
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
            let mut tool_calls: BTreeMap<usize, ToolCallAccumulator> = BTreeMap::new();
            let mut meta_sent = false;
            let mut finish_reason = FinishReason::Stop;
            let mut usage = Usage::default();

            while let Some(event) = events.next().await {
                let event = event.map_err(|e| Error::StreamDecode {
                    provider: provider_name.clone(),
                    message: e.to_string(),
                })?;

                let data = event.data.trim();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }

                let chunk: ChatStreamChunk =
                    serde_json::from_str(data).map_err(|e| Error::StreamDecode {
                        provider: provider_name.clone(),
                        message: e.to_string(),
                    })?;

                if !meta_sent {
                    yield StreamEvent::StreamStart {
                        meta: Meta {
                            id: chunk.id.clone(),
                            model: chunk.model.clone(),
                            rate_limits: Vec::new(),
                        },
                    };
                    meta_sent = true;
                }

                if let Some(choice) = chunk.choices.into_iter().next() {
                    if let Some(content) = choice.delta.content {
                        if !content.is_empty() {
                            yield StreamEvent::TextDelta { text: content };
                        }
                    }

                    for delta in choice.delta.tool_calls {
                        let index = delta.index;
                        let mut arguments_delta = None;

                        {
                            let entry = tool_calls.entry(index).or_default();
                            if let Some(id) = delta.id {
                                entry.id = Some(id);
                            }
                            if let Some(function) = delta.function {
                                if let Some(name) = function.name {
                                    entry.name = Some(name);
                                }
                                if let Some(args) = function.arguments {
                                    if !args.is_empty() {
                                        entry.arguments.push_str(&args);
                                        arguments_delta = Some(args);
                                    }
                                }
                            }
                        }

                        if let Some(arguments_delta) = arguments_delta {
                            let entry = &tool_calls[&index];
                            yield StreamEvent::ToolCallDelta {
                                index,
                                id: entry.id.clone(),
                                name: entry.name.clone(),
                                arguments_delta,
                            };
                        }
                    }

                    if let Some(fr) = choice.finish_reason {
                        finish_reason = maps::map_finish_reason(&fr);
                    }
                }

                if let Some(chunk_usage) = chunk.usage {
                    usage = maps::map_usage(chunk_usage);
                }
            }

            for (_, accumulated) in tool_calls {
                yield StreamEvent::ToolCall(ToolCall {
                    id: accumulated.id.unwrap_or_default(),
                    name: accumulated.name.unwrap_or_default(),
                    arguments: serde_json::from_str(&accumulated.arguments).unwrap_or(Value::Null),
                });
            }

            yield StreamEvent::StepFinish { usage, finish_reason };
        };

        Ok(stream.boxed())
    }

    async fn structured(&self, request: StructuredRequest) -> Result<StructuredResponse, Error> {
        let wire_request = maps::build_structured_request(&request);

        let http_response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&wire_request)
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

        let wire_response: ChatResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        maps::parse_structured_response(wire_response, self.name())
    }

    async fn moderation(&self, request: ModerationRequest) -> Result<ModerationResponse, Error> {
        let wire_request = moderation::build_request(&request);

        let http_response = self
            .client
            .post(format!("{}/moderations", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&wire_request)
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

        let wire_response: moderation::ApiResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        Ok(moderation::parse_response(wire_response))
    }

    async fn embeddings(&self, request: EmbeddingsRequest) -> Result<EmbeddingsResponse, Error> {
        let wire_request = embeddings::build_request(&request);

        let http_response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&wire_request)
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

        let wire_response: embeddings::ApiResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        Ok(embeddings::parse_response(wire_response))
    }

    async fn images(&self, request: ImagesRequest) -> Result<ImagesResponse, Error> {
        let wire_request = images::build_request(&request);

        let http_response = self
            .client
            .post(format!("{}/images/generations", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&wire_request)
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

        let wire_response: images::ApiResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        images::parse_response(wire_response, self.name())
    }

    async fn text_to_speech(&self, request: TextToSpeechRequest) -> Result<AudioResponse, Error> {
        let wire_request = audio::build_speech_request(&request);

        let http_response = self
            .client
            .post(format!("{}/audio/speech", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&wire_request)
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

        let mime_type = http_response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.to_string());
        let data = http_response.bytes().await?;

        Ok(audio::parse_speech_response(data.to_vec(), mime_type))
    }

    async fn speech_to_text(
        &self,
        request: SpeechToTextRequest,
    ) -> Result<TranscriptionResponse, Error> {
        let part = reqwest::multipart::Part::bytes(request.audio.data.clone())
            .file_name(request.audio.filename.clone())
            .mime_str(&request.audio.mime_type)?;
        let form = reqwest::multipart::Form::new()
            .text("model", request.model.clone())
            .part("file", part);

        let http_response = self
            .client
            .post(format!("{}/audio/transcriptions", self.base_url))
            .bearer_auth(&self.api_key)
            .multipart(form)
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

        let wire_response: audio::TranscriptionApiResponse = serde_json::from_str(&body_text)
            .map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        Ok(audio::parse_transcription_response(wire_response))
    }
}

/// Accumulates one tool call's `id`/`name`/`arguments` across however many
/// stream chunks it's spread over, keyed by `index` since a single chunk's
/// `tool_calls` array may contain deltas for several tool calls in progress at
/// once.
#[derive(Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}
