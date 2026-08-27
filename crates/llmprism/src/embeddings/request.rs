use std::sync::Arc;

use crate::error::Error;
use crate::provider::Provider;

use super::response::EmbeddingsResponse;

/// The immutable, provider-agnostic shape of one embeddings call: turn one or
/// more pieces of text into vectors.
#[derive(Clone)]
pub struct EmbeddingsRequest {
    pub model: String,
    /// The text to embed. An embeddings call can process several inputs at
    /// once; [`EmbeddingsResponse::embeddings`] comes back in the same order.
    pub input: Vec<String>,
    /// Requests a shorter output vector than the model's default, for models
    /// that support it (OpenAI's `text-embedding-3-*` models, and VoyageAI's
    /// newer models). `None` leaves this up to the model's own default;
    /// asking a model that doesn't support this for a specific dimension
    /// count is a provider-level error, not something this crate validates
    /// up front.
    pub dimensions: Option<u32>,
    /// Extra provider-specific fields to send alongside this request, for
    /// options this crate doesn't model as a typed field yet. Must be a JSON
    /// object to have any effect: each of its top-level keys is merged into
    /// (and, if it collides with one of this crate's own fields, overrides)
    /// the request body actually sent to the provider. The default,
    /// `Value::Null`, sends nothing extra.
    pub provider_options: serde_json::Value,
}

impl EmbeddingsRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: Vec::new(),
            dimensions: None,
            provider_options: serde_json::Value::Null,
        }
    }
}

/// The fluent, chainable way to build and run an embeddings request.
///
/// Get one of these from [`Registry::embeddings`](crate::Registry::embeddings),
/// add one or more [`with_input`](Self::with_input) calls, then
/// [`generate`](Self::generate).
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
///
/// println!("{} dimensions", response.embeddings[0].vector.len());
/// # Ok(())
/// # }
/// ```
pub struct PendingEmbeddingsRequest {
    provider: Arc<dyn Provider>,
    request: EmbeddingsRequest,
}

impl PendingEmbeddingsRequest {
    /// Starts a new builder for `provider`, targeting `model`, with no
    /// inputs queued yet. You'll normally get one of these from
    /// [`Registry::embeddings`](crate::Registry::embeddings) rather than
    /// calling this directly.
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            request: EmbeddingsRequest::new(model),
        }
    }

    /// Adds one piece of text to embed. Call this more than once to embed
    /// several inputs in a single request.
    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.request.input.push(input.into());
        self
    }

    /// Requests a shorter output vector than the model's default, for
    /// models that support it.
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.request.dimensions = Some(dimensions);
        self
    }

    /// Freezes the builder's current state into an [`EmbeddingsRequest`]
    /// without sending it.
    pub fn to_request(&self) -> EmbeddingsRequest {
        self.request.clone()
    }

    /// Sends the request and returns one embedding per input, in order.
    pub async fn generate(self) -> Result<EmbeddingsResponse, Error> {
        self.provider.embeddings(self.request).await
    }
}
