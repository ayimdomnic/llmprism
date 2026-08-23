//! Providers that speak OpenAI's Chat Completions wire format but are hosted
//! by a different vendor. Each one here is a thin wrapper around
//! [`OpenAiProvider`], pointed at that
//! vendor's own base URL -- the same reuse Prism's PHP source uses for this
//! exact set of providers, and the reason `OpenAiProvider::with_base_url`
//! exists as a public constructor in the first place.
//!
//! # Scope: Text generation only
//!
//! Every provider here implements [`Provider::text_step`] and
//! [`Provider::stream_text_once`] -- a Chat Completions-shaped endpoint is
//! the one thing that's genuinely true of all of them; it's the definition
//! of "OpenAI-compatible" in practice. Structured output, moderation,
//! embeddings, images, and audio are deliberately *not* wired up here, even
//! though some of these vendors happen to support some of them: whether
//! `response_format: {"type": "json_schema", "strict": true}` is actually
//! enforced (rather than silently ignored), or whether a `/embeddings`
//! -shaped endpoint even exists, varies per vendor and isn't something this
//! crate can verify once and rely on staying true -- especially for a proxy
//! like OpenRouter, which routes to many different underlying models with
//! very different capabilities. Calling one of those methods on a provider
//! in this module returns [`Error::Unsupported`],
//! the same as any other provider that hasn't implemented a capability, and
//! that's a deliberate scope decision, not an oversight.
//!
//! # Choosing a base URL
//!
//! Each provider's default matches that vendor's own published
//! OpenAI-compatibility docs at the time this was written. Vendors do change
//! these occasionally (and some, like Z.ai, publish more than one depending
//! on which product tier you're on); every provider here also has a
//! `with_base_url` constructor for pointing at a different URL -- a proxy, a
//! self-hosted gateway, or an updated endpoint -- without waiting for a new
//! release of this crate.

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::Error;
use crate::provider::Provider;
use crate::providers::openai::OpenAiProvider;
use crate::stream_event::StreamEvent;
use crate::text::{Step, TextRequest};

/// Defines one OpenAI-compatible provider struct: a thin wrapper around
/// [`OpenAiProvider`] with a fixed `name` (used in error messages and as the
/// registry key convention) and default base URL, delegating
/// [`Provider::text_step`]/[`Provider::stream_text_once`] straight through.
///
/// `feature` is a plain string literal naming the Cargo feature that gates
/// every item this generates (the struct and both `impl` blocks) -- kept
/// separate from the doc comment(s), which only apply to the struct itself,
/// since repeating a multi-line doc comment on every generated `impl` block
/// too would just be noise.
macro_rules! openai_compatible_provider {
    (
        $(#[$doc:meta])*
        feature = $feature:literal,
        $struct_name:ident, $name:literal, $default_base_url:literal
    ) => {
        $(#[$doc])*
        #[cfg(feature = $feature)]
        pub struct $struct_name {
            inner: OpenAiProvider,
        }

        #[cfg(feature = $feature)]
        impl $struct_name {
            /// Creates a provider using
            #[doc = concat!("`", $default_base_url, "`")]
            /// as the base URL.
            pub fn new(api_key: impl Into<String>) -> Self {
                Self::with_base_url(api_key, $default_base_url)
            }

            /// Creates a provider pointed at a different base URL -- see the
            /// [module docs](self) for why you might need this.
            pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
                Self {
                    inner: OpenAiProvider::with_name_and_base_url($name, api_key, base_url),
                }
            }

            /// Replaces the underlying HTTP client -- an escape hatch for
            /// anything [`build_http_client`](crate::client::build_http_client)
            /// doesn't cover (a request timeout, a proxy, a custom retry
            /// policy). See
            /// [`OpenAiProvider::with_client`] for a full example; this
            /// works the same way.
            pub fn with_client(mut self, client: reqwest_middleware::ClientWithMiddleware) -> Self {
                self.inner = self.inner.with_client(client);
                self
            }
        }

        #[cfg(feature = $feature)]
        #[async_trait]
        impl Provider for $struct_name {
            fn name(&self) -> &str {
                $name
            }

            async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
                self.inner.text_step(request).await
            }

            async fn stream_text_once(
                &self,
                request: TextRequest,
            ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
                self.inner.stream_text_once(request).await
            }
        }
    };
}

openai_compatible_provider!(
    /// [Groq](https://groq.com) -- fast open-weight model inference.
    feature = "groq",
    GroqProvider,
    "groq",
    "https://api.groq.com/openai/v1"
);

openai_compatible_provider!(
    /// [DeepSeek](https://www.deepseek.com).
    feature = "deepseek",
    DeepSeekProvider,
    "deepseek",
    "https://api.deepseek.com/v1"
);

openai_compatible_provider!(
    /// [Mistral AI](https://mistral.ai).
    feature = "mistral",
    MistralProvider,
    "mistral",
    "https://api.mistral.ai/v1"
);

openai_compatible_provider!(
    /// [xAI](https://x.ai)'s Grok models.
    feature = "xai",
    XaiProvider,
    "xai",
    "https://api.x.ai/v1"
);

openai_compatible_provider!(
    /// [OpenRouter](https://openrouter.ai) -- a single API routing to many
    /// underlying providers and models. Because the model actually serving
    /// a request can vary, treat capability support (and even exact
    /// behavior) as whatever the routed-to model happens to provide, not a
    /// fixed guarantee.
    feature = "openrouter",
    OpenRouterProvider,
    "openrouter",
    "https://openrouter.ai/api/v1"
);

openai_compatible_provider!(
    /// [Perplexity](https://www.perplexity.ai)'s Sonar API.
    feature = "perplexity",
    PerplexityProvider,
    "perplexity",
    "https://api.perplexity.ai"
);

openai_compatible_provider!(
    /// [Z.ai](https://z.ai)'s GLM models. Z.ai publishes more than one base
    /// URL depending on plan (a general endpoint and a coding-specific one);
    /// this default is the general endpoint -- pass your plan's endpoint to
    /// [`ZaiProvider::with_base_url`] if you're on a different one.
    feature = "zai",
    ZaiProvider,
    "zai",
    "https://api.z.ai/api/paas/v4"
);
