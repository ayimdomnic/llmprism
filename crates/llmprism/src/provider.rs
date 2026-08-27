//! The [`Provider`] trait -- the interface every LLM backend implements.

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::audio::{
    AudioResponse, SpeechToTextRequest, TextToSpeechRequest, TranscriptionResponse,
};
use crate::embeddings::{EmbeddingsRequest, EmbeddingsResponse};
use crate::error::Error;
use crate::images::{ImagesRequest, ImagesResponse};
use crate::moderation::{ModerationRequest, ModerationResponse};
use crate::rerank::{RerankRequest, RerankResponse};
use crate::stream_event::StreamEvent;
use crate::structured::{StructuredRequest, StructuredResponse, StructuredStreamEvent};
use crate::text::{Step, TextRequest};

/// A single LLM backend -- OpenAI, Anthropic, or any other service you connect
/// this crate to.
///
/// You won't often call methods on `Provider` directly. Instead, you register a
/// provider with a [`Registry`](crate::Registry) under a name, and the registry's
/// fluent builders (like [`Registry::text`](crate::Registry::text)) call into it
/// for you.
///
/// Every capability method has a default implementation that returns
/// [`Error::Unsupported`]. That means a provider only needs to override the
/// methods it actually implements -- a provider with no moderation endpoint simply
/// never overrides `moderation` (once that method exists), and calling it returns
/// a clear error instead of failing to compile or panicking.
///
/// # Implementing a custom provider
///
/// You can plug in your own provider -- a self-hosted model server, an internal
/// gateway, a provider this crate doesn't support yet -- by implementing this
/// trait and registering it, exactly the way the built-in OpenAI and Anthropic
/// providers work:
///
/// ```
/// use async_trait::async_trait;
/// use llmprism::{Error, Provider};
/// use llmprism::text::{Step, TextRequest};
///
/// struct MyProvider;
///
/// #[async_trait]
/// impl Provider for MyProvider {
///     fn name(&self) -> &str { "my-provider" }
///
///     async fn text_step(&self, request: &TextRequest) -> Result<Step, Error> {
///         // Translate `request` into your backend's wire format, send it, and
///         // translate the response back into a `Step`.
///         # unimplemented!()
///     }
/// }
/// ```
#[async_trait]
pub trait Provider: Send + Sync {
    /// A short, lowercase identifier for this provider, used in error messages
    /// (and, when registered normally, as the key applications pass to
    /// [`Registry::text`](crate::Registry::text)).
    fn name(&self) -> &str;

    /// Performs exactly one request/response round trip against the provider's
    /// text-generation endpoint -- send the current conversation, get back one
    /// reply.
    ///
    /// This method deliberately does *not* handle multi-step tool calling: if the
    /// model wants to call a tool, running that tool and sending the result back
    /// for another round trip is handled once, centrally, for every provider, by
    /// [`crate::tool_loop::run_text`]. That's also what
    /// [`PendingTextRequest::generate`](crate::text::PendingTextRequest::generate)
    /// calls under the hood -- so implementing this one method is enough to get
    /// full tool-calling support for free.
    ///
    /// Takes `request` by reference, not by value: `run_text` calls this once per
    /// round trip against a conversation that keeps growing (each round trip
    /// appends the model's tool calls and their results as new messages), so
    /// taking ownership here would force cloning the entire, ever-larger request
    /// on every single round trip. A `Provider` only ever needs to read `request`
    /// to build its wire-format body, never to consume it.
    async fn text_step(&self, request: &TextRequest) -> Result<Step, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "text"))
    }

    /// The streaming counterpart to [`text_step`](Self::text_step): performs
    /// exactly one request/response round trip, but instead of waiting for the
    /// whole reply, returns a stream that yields [`StreamEvent`]s as they arrive.
    ///
    /// Like `text_step`, this only needs to handle a single round trip -- the
    /// multi-step tool-calling loop for streaming lives once, centrally, in
    /// [`crate::stream_loop::stream_text`] (what
    /// [`PendingTextRequest::stream`](crate::text::PendingTextRequest::stream)
    /// calls), the same way [`crate::tool_loop::run_text`] centralizes it for the
    /// non-streaming case. Takes `request` by reference for the same reason
    /// [`text_step`](Self::text_step) does.
    ///
    /// The outer `Result` is for failures that happen before any events can be
    /// produced at all (e.g. the initial HTTP request itself failing); once the
    /// stream is returned, each item is its own `Result` so a mid-stream decode
    /// failure doesn't need to abort the whole method call before it starts.
    async fn stream_text_once(
        &self,
        request: &TextRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "stream_text"))
    }

    /// Performs a structured-output request: send a conversation plus a schema,
    /// get back one reply guaranteed to match that schema's shape.
    ///
    /// Unlike `text_step`, there's no separate "loop" layer above this method --
    /// a structured-output request is always exactly one round trip, so this
    /// method fully implements the capability by itself rather than being one
    /// piece of something centralized elsewhere. What differs between providers
    /// is *how* they get the model to comply with the schema (see
    /// [`crate::structured`] for why); that strategy lives entirely inside each
    /// provider's own implementation of this method.
    async fn structured(&self, request: StructuredRequest) -> Result<StructuredResponse, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "structured"))
    }

    /// The streaming counterpart to [`structured`](Self::structured): instead
    /// of waiting for the whole reply, returns a stream that yields
    /// [`StructuredStreamEvent::PartialObject`]s as the model's JSON output
    /// arrives, ending with exactly one
    /// [`StructuredStreamEvent::End`]. Like `structured`, this is always
    /// exactly one round trip -- there's no separate loop layer above it.
    ///
    /// Not every provider that supports [`structured`](Self::structured)
    /// necessarily supports this too; see [`crate::structured`] for which
    /// ones do.
    async fn stream_structured_once(
        &self,
        request: &StructuredRequest,
    ) -> Result<BoxStream<'static, Result<StructuredStreamEvent, Error>>, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "stream_structured"))
    }

    /// Checks one or more pieces of text against the provider's content-safety
    /// classifier. Not every provider has a moderation endpoint -- Anthropic,
    /// notably, doesn't -- so this defaults to `Error::Unsupported` like every
    /// other optional capability here.
    async fn moderation(&self, request: ModerationRequest) -> Result<ModerationResponse, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "moderation"))
    }

    /// Turns one or more pieces of text into embedding vectors.
    async fn embeddings(&self, request: EmbeddingsRequest) -> Result<EmbeddingsResponse, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "embeddings"))
    }

    /// Scores and sorts a list of documents by relevance to a query. See
    /// [`crate::rerank`] for how this differs from [`embeddings`](Self::embeddings).
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "rerank"))
    }

    /// Generates one or more images from a text prompt.
    async fn images(&self, request: ImagesRequest) -> Result<ImagesResponse, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "images"))
    }

    /// Turns text into spoken audio.
    async fn text_to_speech(&self, request: TextToSpeechRequest) -> Result<AudioResponse, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "text_to_speech"))
    }

    /// Transcribes spoken audio into text.
    async fn speech_to_text(
        &self,
        request: SpeechToTextRequest,
    ) -> Result<TranscriptionResponse, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "speech_to_text"))
    }
}
