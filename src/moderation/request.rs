use std::sync::Arc;

use crate::error::Error;
use crate::provider::Provider;

use super::response::ModerationResponse;

/// The immutable, provider-agnostic shape of one moderation call: check one or
/// more pieces of text against the provider's content-safety classifier.
#[derive(Clone)]
pub struct ModerationRequest {
    pub model: String,
    /// The text to classify. A moderation call can check several inputs at
    /// once; [`ModerationResponse::results`] comes back in the same order.
    pub input: Vec<String>,
    /// Extra provider-specific fields to send alongside this request, for
    /// options this crate doesn't model as a typed field yet. Must be a JSON
    /// object to have any effect: each of its top-level keys is merged into
    /// (and, if it collides with one of this crate's own fields, overrides)
    /// the request body actually sent to the provider. The default,
    /// `Value::Null`, sends nothing extra.
    pub provider_options: serde_json::Value,
}

impl ModerationRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: Vec::new(),
            provider_options: serde_json::Value::Null,
        }
    }
}

/// The fluent, chainable way to build and run a moderation request.
///
/// Get one of these from [`Registry::moderation`](crate::Registry::moderation),
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
///     .moderation("openai", "omni-moderation-latest")?
///     .with_input("Some user-submitted text.")
///     .generate()
///     .await?;
///
/// if response.results[0].flagged {
///     println!("flagged!");
/// }
/// # Ok(())
/// # }
/// ```
pub struct PendingModerationRequest {
    provider: Arc<dyn Provider>,
    request: ModerationRequest,
}

impl PendingModerationRequest {
    /// Starts a new builder for `provider`, targeting `model`, with no inputs
    /// queued yet. You'll normally get one of these from
    /// [`Registry::moderation`](crate::Registry::moderation) rather than
    /// calling this directly.
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            request: ModerationRequest::new(model),
        }
    }

    /// Adds one piece of text to classify. Call this more than once to check
    /// several inputs in a single request.
    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.request.input.push(input.into());
        self
    }

    /// Freezes the builder's current state into a [`ModerationRequest`]
    /// without sending it.
    pub fn to_request(&self) -> ModerationRequest {
        self.request.clone()
    }

    /// Sends the request and returns one result per input, in order.
    pub async fn generate(self) -> Result<ModerationResponse, Error> {
        self.provider.moderation(self.request).await
    }
}
