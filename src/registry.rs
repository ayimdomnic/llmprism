//! [`Registry`] -- where providers are registered and looked up by name.

use std::collections::HashMap;
use std::sync::Arc;

use crate::embeddings::PendingEmbeddingsRequest;
use crate::error::Error;
use crate::moderation::PendingModerationRequest;
use crate::provider::Provider;
use crate::schema::ObjectSchema;
use crate::structured::PendingStructuredRequest;
use crate::text::PendingTextRequest;

/// Maps provider names (like `"openai"`) to the [`Provider`] that handles them.
///
/// This is the front door of the crate: your application code asks the registry
/// for a provider by name rather than constructing a provider directly. That one
/// level of indirection is what lets tests swap in a
/// [`FakeProvider`](crate::testing::FakeProvider) under the same name with no
/// changes anywhere else -- the code under test never knows the difference.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "openai")]
/// # {
/// use llmprism::providers::openai::OpenAiProvider;
/// use llmprism::Registry;
///
/// let mut registry = Registry::new();
/// registry.register("openai", OpenAiProvider::new("sk-..."));
///
/// // Later, anywhere in your application:
/// let builder = registry.text("openai", "gpt-4o-mini");
/// assert!(builder.is_ok());
/// # }
/// ```
#[derive(Default, Clone)]
pub struct Registry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl Registry {
    /// Creates an empty registry with no providers. Useful in tests, where you
    /// typically only want to register a [`FakeProvider`](crate::testing::FakeProvider).
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `provider` under `name`. If a provider was already registered
    /// under that name, it's replaced -- this is exactly how tests swap in a fake:
    /// register it under the same name your application code already uses (e.g.
    /// `"openai"`).
    pub fn register(&mut self, name: impl Into<String>, provider: impl Provider + 'static) {
        self.providers.insert(name.into(), Arc::new(provider));
    }

    /// Like [`register`](Self::register), but for when you already have a shared
    /// `Arc<dyn Provider>` -- for example, one reused across multiple registries.
    pub fn register_arc(&mut self, name: impl Into<String>, provider: Arc<dyn Provider>) {
        self.providers.insert(name.into(), provider);
    }

    /// Looks up the provider registered under `name`.
    ///
    /// You'll rarely need to call this directly -- prefer a capability method like
    /// [`text`](Self::text), which calls this for you and hands back a ready-to-use
    /// request builder.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownProvider`] if nothing is registered under `name`.
    pub fn provider(&self, name: &str) -> Result<Arc<dyn Provider>, Error> {
        self.providers
            .get(name)
            .cloned()
            .ok_or_else(|| Error::UnknownProvider(name.to_string()))
    }

    /// Starts a text-generation request against the provider registered under
    /// `provider_name`, targeting `model`. Chain `.with_*()` calls on the returned
    /// builder to describe the request, then call `.generate()` to run it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownProvider`] if `provider_name` isn't registered.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "openai")]
    /// # async fn example() -> Result<(), llmprism::Error> {
    /// use llmprism::Registry;
    ///
    /// let registry = Registry::from_env();
    /// let response = registry
    ///     .text("openai", "gpt-4o-mini")?
    ///     .with_prompt("Say hello in one word.")
    ///     .generate()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn text(
        &self,
        provider_name: &str,
        model: impl Into<String>,
    ) -> Result<PendingTextRequest, Error> {
        let provider = self.provider(provider_name)?;
        Ok(PendingTextRequest::new(provider, model))
    }

    /// Starts a structured-output request against the provider registered
    /// under `provider_name`, targeting `model` and requiring the reply to
    /// match `schema`. Chain `.with_*()` calls on the returned builder to
    /// describe the conversation, then call `.generate()` to run it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownProvider`] if `provider_name` isn't registered.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "openai")]
    /// # async fn example() -> Result<(), llmprism::Error> {
    /// use llmprism::schema::{ObjectSchema, Schema, StringSchema};
    /// use llmprism::Registry;
    ///
    /// let schema = ObjectSchema::new("greeting")
    ///     .with_property(Schema::String(StringSchema::new("message")), true);
    ///
    /// let registry = Registry::from_env();
    /// let response = registry
    ///     .structured("openai", "gpt-4o-mini", schema)?
    ///     .with_prompt("Greet the user warmly.")
    ///     .generate()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn structured(
        &self,
        provider_name: &str,
        model: impl Into<String>,
        schema: ObjectSchema,
    ) -> Result<PendingStructuredRequest, Error> {
        let provider = self.provider(provider_name)?;
        Ok(PendingStructuredRequest::new(provider, model, schema))
    }

    /// Starts a moderation request against the provider registered under
    /// `provider_name`, targeting `model`. Add one or more `.with_input(...)`
    /// calls on the returned builder, then call `.generate()` to run it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownProvider`] if `provider_name` isn't registered.
    /// If the provider itself has no moderation endpoint (Anthropic, for
    /// instance), `.generate()` on the returned builder will fail with
    /// [`Error::Unsupported`] instead.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "openai")]
    /// # async fn example() -> Result<(), llmprism::Error> {
    /// use llmprism::Registry;
    ///
    /// let registry = Registry::from_env();
    /// let response = registry
    ///     .moderation("openai", "omni-moderation-latest")?
    ///     .with_input("Some user-submitted text.")
    ///     .generate()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn moderation(
        &self,
        provider_name: &str,
        model: impl Into<String>,
    ) -> Result<PendingModerationRequest, Error> {
        let provider = self.provider(provider_name)?;
        Ok(PendingModerationRequest::new(provider, model))
    }

    /// Starts an embeddings request against the provider registered under
    /// `provider_name`, targeting `model`. Add one or more
    /// `.with_input(...)` calls on the returned builder, then call
    /// `.generate()` to run it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownProvider`] if `provider_name` isn't
    /// registered.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "openai")]
    /// # async fn example() -> Result<(), llmprism::Error> {
    /// use llmprism::Registry;
    ///
    /// let registry = Registry::from_env();
    /// let response = registry
    ///     .embeddings("openai", "text-embedding-3-small")?
    ///     .with_input("The quick brown fox.")
    ///     .generate()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn embeddings(
        &self,
        provider_name: &str,
        model: impl Into<String>,
    ) -> Result<PendingEmbeddingsRequest, Error> {
        let provider = self.provider(provider_name)?;
        Ok(PendingEmbeddingsRequest::new(provider, model))
    }

    /// Builds a registry from the first-party providers that have their API key
    /// set in the environment -- `OPENAI_API_KEY` for OpenAI, `ANTHROPIC_API_KEY`
    /// for Anthropic, and so on -- and whose Cargo feature is enabled. A provider
    /// whose key isn't set is simply left out, rather than causing an error; if you
    /// then try to use it, you'll get [`Error::UnknownProvider`] at the point of
    /// use, which is usually clearer than failing at startup.
    ///
    /// This is the fastest way to get started; reach for [`Registry::new`] plus
    /// manual [`register`](Self::register) calls when you want more control (a
    /// custom base URL, a provider that reads its key from somewhere other than an
    /// env var, and so on).
    #[cfg(feature = "http")]
    pub fn from_env() -> Self {
        let mut registry = Self::new();

        #[cfg(feature = "openai")]
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            registry.register(
                "openai",
                crate::providers::openai::OpenAiProvider::new(api_key),
            );
        }

        #[cfg(feature = "anthropic")]
        if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
            registry.register(
                "anthropic",
                crate::providers::anthropic::AnthropicProvider::new(api_key),
            );
        }

        registry
    }
}
