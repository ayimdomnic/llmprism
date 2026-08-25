//! [`ProviderMiddleware`] -- wraps a [`Provider`] to intercept its calls:
//! transform a request before it's sent, transform or replace a response
//! after it comes back, or skip the call entirely (a cache hit, a policy
//! rejection). Start with [`Registry::wrap`](crate::Registry::wrap).

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::audio::{
    AudioResponse, SpeechToTextRequest, TextToSpeechRequest, TranscriptionResponse,
};
use crate::embeddings::{EmbeddingsRequest, EmbeddingsResponse};
use crate::error::Error;
use crate::images::{ImagesRequest, ImagesResponse};
use crate::moderation::{ModerationRequest, ModerationResponse};
use crate::provider::Provider;
use crate::rerank::{RerankRequest, RerankResponse};
use crate::stream_event::StreamEvent;
use crate::structured::{StructuredRequest, StructuredResponse};
use crate::text::{Step, TextRequest};

/// One method per [`Provider`] capability, each defaulting to a plain
/// passthrough to `next` -- override only the ones you actually want to
/// intercept.
///
/// A middleware can inspect or rewrite `request` before calling `next`,
/// inspect or rewrite the result `next` returns, or skip calling `next`
/// altogether and return its own result (a cached response, a policy
/// rejection reported as an [`Error`]). Middlewares compose: wrapping an
/// already-wrapped provider (via repeated [`Registry::wrap`](crate::Registry::wrap)
/// calls) nests the new middleware around the existing one, outermost first.
///
/// # Example
///
/// A middleware that logs how long each text generation call took:
///
/// ```
/// use async_trait::async_trait;
/// use llmprism::middleware::ProviderMiddleware;
/// use llmprism::text::{Step, TextRequest};
/// use llmprism::{Error, Provider};
/// use std::time::Instant;
///
/// struct LogTiming;
///
/// #[async_trait]
/// impl ProviderMiddleware for LogTiming {
///     async fn text_step(&self, request: TextRequest, next: &dyn Provider) -> Result<Step, Error> {
///         let started = Instant::now();
///         let result = next.text_step(&request).await;
///         println!("text_step took {:?}", started.elapsed());
///         result
///     }
/// }
/// ```
#[async_trait]
pub trait ProviderMiddleware: Send + Sync {
    /// Intercepts [`Provider::text_step`]. Takes `request` by value, unlike
    /// [`Provider::text_step`] itself -- a middleware commonly wants to mutate
    /// it before forwarding (add a header-equivalent field, inject a system
    /// prompt), which is far more ergonomic against an owned value. The one
    /// extra clone this costs happens only for a middleware-wrapped provider,
    /// not on the default, unwrapped path every other provider call takes.
    async fn text_step(&self, request: TextRequest, next: &dyn Provider) -> Result<Step, Error> {
        next.text_step(&request).await
    }

    /// Intercepts [`Provider::stream_text_once`]. The `Result` this returns
    /// covers only starting the stream -- to intercept individual
    /// [`StreamEvent`]s once streaming has begun, wrap the returned stream
    /// (e.g. with `futures::StreamExt::map`) before returning it.
    async fn stream_text_once(
        &self,
        request: TextRequest,
        next: &dyn Provider,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        next.stream_text_once(&request).await
    }

    /// Intercepts [`Provider::structured`].
    async fn structured(
        &self,
        request: StructuredRequest,
        next: &dyn Provider,
    ) -> Result<StructuredResponse, Error> {
        next.structured(request).await
    }

    /// Intercepts [`Provider::moderation`].
    async fn moderation(
        &self,
        request: ModerationRequest,
        next: &dyn Provider,
    ) -> Result<ModerationResponse, Error> {
        next.moderation(request).await
    }

    /// Intercepts [`Provider::embeddings`].
    async fn embeddings(
        &self,
        request: EmbeddingsRequest,
        next: &dyn Provider,
    ) -> Result<EmbeddingsResponse, Error> {
        next.embeddings(request).await
    }

    /// Intercepts [`Provider::rerank`].
    async fn rerank(
        &self,
        request: RerankRequest,
        next: &dyn Provider,
    ) -> Result<RerankResponse, Error> {
        next.rerank(request).await
    }

    /// Intercepts [`Provider::images`].
    async fn images(
        &self,
        request: ImagesRequest,
        next: &dyn Provider,
    ) -> Result<ImagesResponse, Error> {
        next.images(request).await
    }

    /// Intercepts [`Provider::text_to_speech`].
    async fn text_to_speech(
        &self,
        request: TextToSpeechRequest,
        next: &dyn Provider,
    ) -> Result<AudioResponse, Error> {
        next.text_to_speech(request).await
    }

    /// Intercepts [`Provider::speech_to_text`].
    async fn speech_to_text(
        &self,
        request: SpeechToTextRequest,
        next: &dyn Provider,
    ) -> Result<TranscriptionResponse, Error> {
        next.speech_to_text(request).await
    }
}

/// A [`Provider`] that routes every capability call through a
/// [`ProviderMiddleware`] before reaching the wrapped provider. You'll
/// normally get one of these from [`Registry::wrap`](crate::Registry::wrap)
/// rather than constructing it directly.
pub struct MiddlewareProvider {
    inner: Arc<dyn Provider>,
    middleware: Arc<dyn ProviderMiddleware>,
}

impl MiddlewareProvider {
    /// Wraps `inner` so every capability call goes through `middleware` first.
    pub fn new(inner: Arc<dyn Provider>, middleware: impl ProviderMiddleware + 'static) -> Self {
        Self {
            inner,
            middleware: Arc::new(middleware),
        }
    }
}

#[async_trait]
impl Provider for MiddlewareProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn text_step(&self, request: &TextRequest) -> Result<Step, Error> {
        self.middleware
            .text_step(request.clone(), self.inner.as_ref())
            .await
    }

    async fn stream_text_once(
        &self,
        request: &TextRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        self.middleware
            .stream_text_once(request.clone(), self.inner.as_ref())
            .await
    }

    async fn structured(&self, request: StructuredRequest) -> Result<StructuredResponse, Error> {
        self.middleware
            .structured(request, self.inner.as_ref())
            .await
    }

    async fn moderation(&self, request: ModerationRequest) -> Result<ModerationResponse, Error> {
        self.middleware
            .moderation(request, self.inner.as_ref())
            .await
    }

    async fn embeddings(&self, request: EmbeddingsRequest) -> Result<EmbeddingsResponse, Error> {
        self.middleware
            .embeddings(request, self.inner.as_ref())
            .await
    }

    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, Error> {
        self.middleware.rerank(request, self.inner.as_ref()).await
    }

    async fn images(&self, request: ImagesRequest) -> Result<ImagesResponse, Error> {
        self.middleware.images(request, self.inner.as_ref()).await
    }

    async fn text_to_speech(&self, request: TextToSpeechRequest) -> Result<AudioResponse, Error> {
        self.middleware
            .text_to_speech(request, self.inner.as_ref())
            .await
    }

    async fn speech_to_text(
        &self,
        request: SpeechToTextRequest,
    ) -> Result<TranscriptionResponse, Error> {
        self.middleware
            .speech_to_text(request, self.inner.as_ref())
            .await
    }
}
