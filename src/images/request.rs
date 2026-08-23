use std::sync::Arc;

use crate::error::Error;
use crate::provider::Provider;

use super::response::ImagesResponse;

/// The immutable, provider-agnostic shape of one image-generation call.
#[derive(Clone)]
pub struct ImagesRequest {
    pub model: String,
    pub prompt: String,
    /// How many images to generate. `None` leaves this up to the provider's
    /// own default (usually one).
    pub n: Option<u32>,
    /// A provider-specific size string (e.g. `"1024x1024"`). Kept as a plain
    /// string rather than an enum since valid sizes vary by model and
    /// provider; `None` leaves this up to the provider's own default.
    pub size: Option<String>,
    /// Escape hatch for provider-specific options this crate doesn't model
    /// directly yet (style, quality, and so on). Interpretation is entirely
    /// up to the provider.
    pub provider_options: serde_json::Value,
}

impl ImagesRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            n: None,
            size: None,
            provider_options: serde_json::Value::Null,
        }
    }
}

/// The fluent, chainable way to build and run an image-generation request.
///
/// Get one of these from [`Registry::images`](crate::Registry::images),
/// optionally chain `.with_*()` calls, then [`generate`](Self::generate).
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
///     .images("openai", "dall-e-3", "A watercolor painting of a lighthouse.")?
///     .with_size("1024x1024")
///     .generate()
///     .await?;
///
/// println!("{:?}", response.images[0].data);
/// # Ok(())
/// # }
/// ```
pub struct PendingImagesRequest {
    provider: Arc<dyn Provider>,
    request: ImagesRequest,
}

impl PendingImagesRequest {
    /// Starts a new builder for `provider`, targeting `model` with the given
    /// `prompt`. You'll normally get one of these from
    /// [`Registry::images`](crate::Registry::images) rather than calling
    /// this directly.
    pub fn new(
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            request: ImagesRequest::new(model, prompt),
        }
    }

    /// Requests `n` images instead of the provider's default.
    pub fn with_count(mut self, n: u32) -> Self {
        self.request.n = Some(n);
        self
    }

    /// Requests a specific size (e.g. `"1024x1024"`) instead of the
    /// provider's default. Valid values depend on the model.
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.request.size = Some(size.into());
        self
    }

    /// Freezes the builder's current state into an [`ImagesRequest`] without
    /// sending it.
    pub fn to_request(&self) -> ImagesRequest {
        self.request.clone()
    }

    /// Sends the request and returns the generated image(s).
    pub async fn generate(self) -> Result<ImagesResponse, Error> {
        self.provider.images(self.request).await
    }
}
