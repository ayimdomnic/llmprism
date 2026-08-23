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
    /// A provider-specific quality string (e.g. OpenAI's `dall-e-3` accepts
    /// `"standard"`/`"hd"`, while `gpt-image-1` accepts
    /// `"low"`/`"medium"`/`"high"`/`"auto"`). Kept as a plain string for the
    /// same reason as `size`; `None` leaves this up to the provider's own
    /// default.
    pub quality: Option<String>,
    /// A provider-specific style string (OpenAI's `dall-e-3` accepts
    /// `"vivid"`/`"natural"`; most other models have no equivalent concept
    /// at all). `None` leaves this up to the provider's own default.
    pub style: Option<String>,
    /// Extra provider-specific fields to send alongside this request, for
    /// options this crate doesn't model as a typed field yet. Must be a JSON
    /// object to have any effect: each of its top-level keys is merged into
    /// (and, if it collides with one of this crate's own fields, overrides)
    /// the request body actually sent to the provider. The default,
    /// `Value::Null`, sends nothing extra.
    pub provider_options: serde_json::Value,
}

impl ImagesRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            n: None,
            size: None,
            quality: None,
            style: None,
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

    /// Requests a specific quality (e.g. `"hd"`) instead of the provider's
    /// default. Valid values depend on the model.
    pub fn with_quality(mut self, quality: impl Into<String>) -> Self {
        self.request.quality = Some(quality.into());
        self
    }

    /// Requests a specific style (e.g. `"vivid"`) instead of the provider's
    /// default. Valid values depend on the model; most models have no
    /// equivalent concept at all.
    pub fn with_style(mut self, style: impl Into<String>) -> Self {
        self.request.style = Some(style.into());
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
