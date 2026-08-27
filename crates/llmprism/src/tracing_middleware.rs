//! [`GenAiTracingMiddleware`] -- instruments every [`Provider`] call with a
//! [`tracing`] span following the OpenTelemetry Semantic Conventions for
//! Generative AI (the `gen_ai.*` attribute namespace -- still in
//! [Development status](https://github.com/open-telemetry/semantic-conventions-genai)
//! as of this writing, so expect it to keep evolving; this module uses the
//! current names, including renames like `gen_ai.system` ->
//! `gen_ai.provider.name` and `gen_ai.usage.prompt_tokens` ->
//! `gen_ai.usage.input_tokens`).
//!
//! This crate depends only on [`tracing`] here, not `opentelemetry` itself:
//! bridge to a real OTel backend from your own application with
//! [`tracing-opentelemetry`](https://docs.rs/tracing-opentelemetry) instead
//! of this crate picking an OTel SDK version for you -- `tracing-opentelemetry`
//! and `opentelemetry` don't share version numbers (e.g.
//! `tracing-opentelemetry` 0.33 needs `opentelemetry` ^0.32), so pin an
//! exact compatible pair rather than assuming they match.
//!
//! ```
//! use llmprism::testing::{FakeProvider, FakeTextResponse};
//! use llmprism::tracing_middleware::GenAiTracingMiddleware;
//! use llmprism::Registry;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let mut registry = Registry::new();
//! registry.register("fake", FakeProvider::new("fake").respond_with(FakeTextResponse::new("hi")));
//! registry.wrap("fake", GenAiTracingMiddleware).unwrap();
//!
//! registry.text("fake", "test-model").unwrap().with_prompt("hello").generate().await.unwrap();
//! # }
//! ```
//!
//! # A note on span names
//!
//! The semantic conventions describe a span name shaped like
//! `"{gen_ai.operation.name} {gen_ai.request.model}"` (e.g. `"chat
//! gpt-4o-mini"`). A [`tracing`] span's name, though, is fixed at compile
//! time per callsite -- it can't be built from a runtime value the way the
//! model name is here. This isn't unique to llmprism: the published
//! `rig-core` crate hits the same constraint and settles for a static span
//! name too. Every span below uses a static name for its operation
//! (`"gen_ai.chat"`, `"gen_ai.embeddings"`, and so on) and carries the model
//! as an ordinary `gen_ai.request.model` field instead -- any exporter or UI
//! that groups or filters by attributes rather than by span name sees the
//! same information either way.
//!
//! # What isn't covered yet
//!
//! - **Tool execution.** The semantic conventions describe a separate
//!   `execute_tool` span for a tool call actually running, but that happens
//!   inside [`crate::tool_loop`]/[`crate::stream_loop`], outside any
//!   [`ProviderMiddleware`]'s reach (which only wraps [`Provider`] calls, not
//!   the tool calls those loops make on your behalf).
//! - **Streaming completion.** [`stream_text_once`](Provider::stream_text_once)'s
//!   span covers only *establishing* the stream -- a stream's actual
//!   completion (final usage, finish reason) happens later, as the caller
//!   consumes the returned [`StreamEvent`]s, not inside this call.

use async_trait::async_trait;
use futures::stream::BoxStream;
use tracing::field::Empty;
use tracing::Instrument;

use crate::audio::{
    AudioResponse, SpeechToTextRequest, TextToSpeechRequest, TranscriptionResponse,
};
use crate::embeddings::{EmbeddingsRequest, EmbeddingsResponse};
use crate::error::Error;
use crate::images::{ImagesRequest, ImagesResponse};
use crate::middleware::ProviderMiddleware;
use crate::moderation::{ModerationRequest, ModerationResponse};
use crate::provider::Provider;
use crate::rerank::{RerankRequest, RerankResponse};
use crate::stream_event::StreamEvent;
use crate::structured::{StructuredRequest, StructuredResponse};
use crate::text::{Step, TextRequest};

/// See the [module docs](self) for what this instruments and its current
/// limitations. Has no state of its own -- attach it with
/// [`Registry::wrap`](crate::Registry::wrap).
#[derive(Debug, Clone, Copy, Default)]
pub struct GenAiTracingMiddleware;

#[async_trait]
impl ProviderMiddleware for GenAiTracingMiddleware {
    async fn text_step(&self, request: TextRequest, next: &dyn Provider) -> Result<Step, Error> {
        let span = tracing::info_span!(
            "gen_ai.chat",
            "gen_ai.operation.name" = "chat",
            "gen_ai.provider.name" = next.name(),
            "gen_ai.request.model" = %request.model,
            "gen_ai.request.stream" = false,
            "gen_ai.request.temperature" = ?request.temperature,
            "gen_ai.request.max_tokens" = ?request.max_tokens,
            "gen_ai.request.top_p" = ?request.top_p,
            "gen_ai.response.model" = Empty,
            "gen_ai.usage.input_tokens" = Empty,
            "gen_ai.usage.output_tokens" = Empty,
        );
        async move {
            let result = next.text_step(&request).await;
            if let Ok(step) = &result {
                record_response(&step.meta, Some(&step.usage));
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn stream_text_once(
        &self,
        request: TextRequest,
        next: &dyn Provider,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let span = tracing::info_span!(
            "gen_ai.chat",
            "gen_ai.operation.name" = "chat",
            "gen_ai.provider.name" = next.name(),
            "gen_ai.request.model" = %request.model,
            "gen_ai.request.stream" = true,
            "gen_ai.request.temperature" = ?request.temperature,
            "gen_ai.request.max_tokens" = ?request.max_tokens,
            "gen_ai.request.top_p" = ?request.top_p,
        );
        async move { next.stream_text_once(&request).await }
            .instrument(span)
            .await
    }

    async fn structured(
        &self,
        request: StructuredRequest,
        next: &dyn Provider,
    ) -> Result<StructuredResponse, Error> {
        let span = tracing::info_span!(
            "gen_ai.structured_output",
            "gen_ai.operation.name" = "structured_output",
            "gen_ai.provider.name" = next.name(),
            "gen_ai.request.model" = %request.model,
            "gen_ai.request.temperature" = ?request.temperature,
            "gen_ai.request.max_tokens" = ?request.max_tokens,
            "gen_ai.request.top_p" = ?request.top_p,
            "gen_ai.response.model" = Empty,
            "gen_ai.usage.input_tokens" = Empty,
            "gen_ai.usage.output_tokens" = Empty,
        );
        async move {
            let result = next.structured(request).await;
            if let Ok(response) = &result {
                record_response(&response.meta, Some(&response.usage));
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn moderation(
        &self,
        request: ModerationRequest,
        next: &dyn Provider,
    ) -> Result<ModerationResponse, Error> {
        let span = tracing::info_span!(
            "gen_ai.moderation",
            "gen_ai.operation.name" = "moderation",
            "gen_ai.provider.name" = next.name(),
            "gen_ai.request.model" = %request.model,
            "gen_ai.response.model" = Empty,
        );
        async move {
            let result = next.moderation(request).await;
            if let Ok(response) = &result {
                record_response(&response.meta, None);
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn embeddings(
        &self,
        request: EmbeddingsRequest,
        next: &dyn Provider,
    ) -> Result<EmbeddingsResponse, Error> {
        let span = tracing::info_span!(
            "gen_ai.embeddings",
            "gen_ai.operation.name" = "embeddings",
            "gen_ai.provider.name" = next.name(),
            "gen_ai.request.model" = %request.model,
            "gen_ai.response.model" = Empty,
            "gen_ai.usage.input_tokens" = Empty,
        );
        async move {
            let result = next.embeddings(request).await;
            if let Ok(response) = &result {
                record_response_model(&response.meta);
                tracing::Span::current().record(
                    "gen_ai.usage.input_tokens",
                    response.usage.prompt_tokens as u64,
                );
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        next: &dyn Provider,
    ) -> Result<RerankResponse, Error> {
        let span = tracing::info_span!(
            "gen_ai.rerank",
            "gen_ai.operation.name" = "rerank",
            "gen_ai.provider.name" = next.name(),
            "gen_ai.request.model" = %request.model,
            "gen_ai.response.model" = Empty,
            "gen_ai.usage.input_tokens" = Empty,
        );
        async move {
            let result = next.rerank(request).await;
            if let Ok(response) = &result {
                record_response_model(&response.meta);
                tracing::Span::current().record(
                    "gen_ai.usage.input_tokens",
                    response.usage.prompt_tokens as u64,
                );
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn images(
        &self,
        request: ImagesRequest,
        next: &dyn Provider,
    ) -> Result<ImagesResponse, Error> {
        let span = tracing::info_span!(
            "gen_ai.image_generation",
            "gen_ai.operation.name" = "image_generation",
            "gen_ai.provider.name" = next.name(),
            "gen_ai.request.model" = %request.model,
            "gen_ai.response.model" = Empty,
        );
        async move {
            let result = next.images(request).await;
            if let Ok(response) = &result {
                record_response_model(&response.meta);
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn text_to_speech(
        &self,
        request: TextToSpeechRequest,
        next: &dyn Provider,
    ) -> Result<AudioResponse, Error> {
        let span = tracing::info_span!(
            "gen_ai.text_to_speech",
            "gen_ai.operation.name" = "text_to_speech",
            "gen_ai.provider.name" = next.name(),
            "gen_ai.request.model" = %request.model,
            "gen_ai.response.model" = Empty,
        );
        async move {
            let result = next.text_to_speech(request).await;
            if let Ok(response) = &result {
                record_response_model(&response.meta);
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn speech_to_text(
        &self,
        request: SpeechToTextRequest,
        next: &dyn Provider,
    ) -> Result<TranscriptionResponse, Error> {
        let span = tracing::info_span!(
            "gen_ai.speech_to_text",
            "gen_ai.operation.name" = "speech_to_text",
            "gen_ai.provider.name" = next.name(),
            "gen_ai.request.model" = %request.model,
            "gen_ai.response.model" = Empty,
        );
        async move {
            let result = next.speech_to_text(request).await;
            if let Ok(response) = &result {
                record_response_model(&response.meta);
            }
            result
        }
        .instrument(span)
        .await
    }
}

/// Records `gen_ai.response.model` (from `meta`) and, when `usage` is
/// `Some`, `gen_ai.usage.input_tokens`/`gen_ai.usage.output_tokens` on the
/// current span -- the common tail shared by every capability that reports
/// token usage. Must be called from inside the span it's recording onto
/// (via [`Instrument`]), since it reads [`tracing::Span::current`].
fn record_response(meta: &crate::value_objects::Meta, usage: Option<&crate::value_objects::Usage>) {
    record_response_model(meta);
    if let Some(usage) = usage {
        let span = tracing::Span::current();
        span.record("gen_ai.usage.input_tokens", usage.prompt_tokens as u64);
        span.record("gen_ai.usage.output_tokens", usage.completion_tokens as u64);
    }
}

/// Just the `gen_ai.response.model` half of [`record_response`], for the
/// capabilities that report no token usage at all (moderation, images, and
/// audio) but still report which model actually ran.
fn record_response_model(meta: &crate::value_objects::Meta) {
    tracing::Span::current().record("gen_ai.response.model", meta.model.as_deref());
}
