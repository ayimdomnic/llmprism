//! The [`Provider`] trait -- the interface every LLM backend implements.

use async_trait::async_trait;

use crate::error::Error;
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
///     async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
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
    async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
        let _ = request;
        Err(Error::unsupported(self.name(), "text"))
    }
}
