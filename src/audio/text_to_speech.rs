use std::sync::Arc;

use crate::error::Error;
use crate::provider::Provider;
use crate::value_objects::Meta;

/// The immutable, provider-agnostic shape of one text-to-speech call.
#[derive(Clone)]
pub struct TextToSpeechRequest {
    pub model: String,
    pub input: String,
    /// The voice to speak with, in whatever names the provider defines (e.g.
    /// OpenAI's `"alloy"`, `"echo"`, ...). `None` leaves this up to the
    /// provider's own default.
    pub voice: Option<String>,
    /// Escape hatch for provider-specific options this crate doesn't model
    /// directly yet. Interpretation is entirely up to the provider.
    pub provider_options: serde_json::Value,
}

impl TextToSpeechRequest {
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            voice: None,
            provider_options: serde_json::Value::Null,
        }
    }
}

/// Generated audio: raw bytes plus the MIME type they're encoded in (e.g.
/// `"audio/mpeg"`), so you know how to write, play, or re-encode them.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    pub data: Vec<u8>,
    pub mime_type: String,
}

/// The result of a text-to-speech call.
#[derive(Debug, Clone)]
pub struct AudioResponse {
    pub audio: AudioOutput,
    pub meta: Meta,
}

/// The fluent, chainable way to build and run a text-to-speech request.
///
/// Get one of these from
/// [`Registry::text_to_speech`](crate::Registry::text_to_speech), optionally
/// chain [`with_voice`](Self::with_voice), then [`generate`](Self::generate).
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "openai")]
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use llmprism::Registry;
///
/// let registry = Registry::from_env();
/// let response = registry
///     .text_to_speech("openai", "tts-1", "Hello, world!")?
///     .with_voice("alloy")
///     .generate()
///     .await?;
///
/// std::fs::write("hello.mp3", response.audio.data)?;
/// # Ok(())
/// # }
/// ```
pub struct PendingTextToSpeechRequest {
    provider: Arc<dyn Provider>,
    request: TextToSpeechRequest,
}

impl PendingTextToSpeechRequest {
    /// Starts a new builder for `provider`, targeting `model`, speaking
    /// `input`. You'll normally get one of these from
    /// [`Registry::text_to_speech`](crate::Registry::text_to_speech) rather
    /// than calling this directly.
    pub fn new(
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
        input: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            request: TextToSpeechRequest::new(model, input),
        }
    }

    /// Requests a specific voice instead of the provider's default.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.request.voice = Some(voice.into());
        self
    }

    /// Freezes the builder's current state into a [`TextToSpeechRequest`]
    /// without sending it.
    pub fn to_request(&self) -> TextToSpeechRequest {
        self.request.clone()
    }

    /// Sends the request and returns the generated audio.
    pub async fn generate(self) -> Result<AudioResponse, Error> {
        self.provider.text_to_speech(self.request).await
    }
}
