//! The Gemini provider, talking to Google's Generative Language API
//! (`generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`,
//! and `:streamGenerateContent?alt=sse` for streaming). Enable with the
//! `gemini` Cargo feature.
//!
//! Authenticates via the `x-goog-api-key` header rather than the `?key=`
//! query parameter Gemini's own docs usually show first -- both work, but a
//! header keeps the API key out of the URL, which matters here specifically
//! because `reqwest::Error`'s own `Display` output includes the request URL:
//! a transport-level failure (a dropped connection, a DNS error) would
//! otherwise risk leaking the key into anything that logs or prints the
//! resulting [`Error`].

mod config;
mod maps;
mod wire;

use async_stream::try_stream;
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::stream::{BoxStream, StreamExt};
use reqwest_middleware::ClientWithMiddleware;

use crate::client::{build_http_client, ErrorMapper};
use crate::error::Error;
use crate::provider::Provider;
use crate::stream_event::StreamEvent;
use crate::structured::{StructuredRequest, StructuredResponse};
use crate::text::{Step, TextRequest};
use crate::value_objects::{FinishReason, Meta, ToolCall, Usage};

use self::config::DEFAULT_BASE_URL;
use self::wire::{GenerateContentResponse, Part};

/// A [`Provider`] backed by Google's Gemini API.
///
/// # Example
///
/// ```no_run
/// use llmprism::providers::gemini::GeminiProvider;
/// use llmprism::Registry;
///
/// let mut registry = Registry::new();
/// registry.register(
///     "gemini",
///     GeminiProvider::new(std::env::var("GEMINI_API_KEY").unwrap()),
/// );
/// ```
///
/// If you're happy reading the API key from `GEMINI_API_KEY` yourself, you
/// likely don't need to construct this directly -- see
/// [`Registry::from_env`](crate::Registry::from_env).
pub struct GeminiProvider {
    api_key: String,
    base_url: String,
    client: ClientWithMiddleware,
}

impl GeminiProvider {
    /// Creates a provider that talks to the real Gemini API using `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Creates a provider pointed at a different base URL.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: build_http_client(),
        }
    }
}

#[async_trait]
impl Provider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
        let model = request.model.clone();
        let wire_request = maps::build_request(&request);

        let http_response = self
            .client
            .post(format!("{}/models/{model}:generateContent", self.base_url))
            .header("x-goog-api-key", &self.api_key)
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

        let wire_response: GenerateContentResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        maps::parse_response(wire_response, self.name())
    }

    async fn stream_text_once(
        &self,
        request: TextRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let model = request.model.clone();
        let wire_request = maps::build_request(&request);

        let http_response = self
            .client
            .post(format!(
                "{}/models/{model}:streamGenerateContent?alt=sse",
                self.base_url
            ))
            .header("x-goog-api-key", &self.api_key)
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
            let mut meta_sent = false;
            let mut finish_reason = FinishReason::Stop;
            let mut usage = Usage::default();
            let mut call_index = 0usize;

            while let Some(event) = events.next().await {
                let event = event.map_err(|e| Error::StreamDecode {
                    provider: provider_name.clone(),
                    message: e.to_string(),
                })?;

                let data = event.data.trim();
                if data.is_empty() {
                    continue;
                }

                let chunk: GenerateContentResponse =
                    serde_json::from_str(data).map_err(|e| Error::StreamDecode {
                        provider: provider_name.clone(),
                        message: e.to_string(),
                    })?;

                if !meta_sent {
                    yield StreamEvent::StreamStart {
                        meta: Meta {
                            id: chunk.response_id.clone(),
                            model: chunk.model_version.clone(),
                            rate_limits: Vec::new(),
                        },
                    };
                    meta_sent = true;
                }

                if let Some(candidate) = chunk.candidates.into_iter().next() {
                    if let Some(content) = candidate.content {
                        // Unlike OpenAI/Anthropic, Gemini doesn't stream a
                        // function call's arguments in as incremental JSON
                        // fragments -- each `functionCall` part already
                        // carries its complete `args`, so there's no partial
                        // state to accumulate across chunks the way there is
                        // for text.
                        for part in content.parts {
                            match part {
                                Part::Text { text } => {
                                    if !text.is_empty() {
                                        yield StreamEvent::TextDelta { text };
                                    }
                                }
                                Part::FunctionCall { function_call } => {
                                    let id = format!("call_{call_index}");
                                    call_index += 1;
                                    yield StreamEvent::ToolCallDelta {
                                        index: call_index - 1,
                                        id: Some(id.clone()),
                                        name: Some(function_call.name.clone()),
                                        arguments_delta: function_call.args.to_string(),
                                    };
                                    yield StreamEvent::ToolCall(ToolCall {
                                        id,
                                        name: function_call.name,
                                        arguments: function_call.args,
                                    });
                                    finish_reason = FinishReason::ToolCalls;
                                }
                                Part::FunctionResponse { .. } | Part::Other(_) => {}
                            }
                        }
                    }

                    if finish_reason != FinishReason::ToolCalls {
                        if let Some(fr) = candidate.finish_reason {
                            finish_reason = maps::map_finish_reason(Some(&fr));
                        }
                    }
                }

                if let Some(chunk_usage) = chunk.usage_metadata {
                    usage = maps::map_usage(chunk_usage);
                }
            }

            yield StreamEvent::StepFinish { usage, finish_reason };
        };

        Ok(stream.boxed())
    }

    async fn structured(&self, request: StructuredRequest) -> Result<StructuredResponse, Error> {
        let model = request.model.clone();
        let wire_request = maps::build_structured_request(&request);

        let http_response = self
            .client
            .post(format!("{}/models/{model}:generateContent", self.base_url))
            .header("x-goog-api-key", &self.api_key)
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

        let wire_response: GenerateContentResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        maps::parse_structured_response(wire_response, self.name())
    }
}
