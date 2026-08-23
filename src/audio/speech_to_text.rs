use std::sync::Arc;

use crate::error::Error;
use crate::provider::Provider;
use crate::value_objects::Meta;

/// Audio to transcribe: raw bytes, the MIME type they're encoded in (e.g.
/// `"audio/mpeg"`, `"audio/wav"`), and a filename -- providers that accept
/// audio as a file upload (rather than embedded in a JSON body) need a name
/// to attach to the upload, even though the content of the name doesn't
/// usually matter.
#[derive(Debug, Clone)]
pub struct AudioInput {
    pub data: Vec<u8>,
    pub mime_type: String,
    pub filename: String,
}

impl AudioInput {
    /// Creates an input from raw audio bytes and their MIME type, with a
    /// generic default filename -- override it with
    /// [`with_filename`](Self::with_filename) if a provider cares about the
    /// file extension specifically.
    pub fn new(data: Vec<u8>, mime_type: impl Into<String>) -> Self {
        Self {
            data,
            mime_type: mime_type.into(),
            filename: "audio".to_string(),
        }
    }

    /// Sets the filename attached to the upload.
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = filename.into();
        self
    }
}

/// The immutable, provider-agnostic shape of one speech-to-text call.
#[derive(Clone)]
pub struct SpeechToTextRequest {
    pub model: String,
    pub audio: AudioInput,
    /// Escape hatch for provider-specific options this crate doesn't model
    /// directly yet (a language hint, a prompt to bias transcription, and so
    /// on). Interpretation is entirely up to the provider.
    pub provider_options: serde_json::Value,
}

impl SpeechToTextRequest {
    pub fn new(model: impl Into<String>, audio: AudioInput) -> Self {
        Self {
            model: model.into(),
            audio,
            provider_options: serde_json::Value::Null,
        }
    }
}

/// The result of a speech-to-text call.
#[derive(Debug, Clone)]
pub struct TranscriptionResponse {
    pub text: String,
    pub meta: Meta,
}

/// The fluent, chainable way to build and run a speech-to-text request.
///
/// Get one of these from
/// [`Registry::speech_to_text`](crate::Registry::speech_to_text), then
/// [`generate`](Self::generate).
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "openai")]
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use llmprism::audio::AudioInput;
/// use llmprism::Registry;
///
/// let bytes = std::fs::read("recording.mp3")?;
/// let audio = AudioInput::new(bytes, "audio/mpeg").with_filename("recording.mp3");
///
/// let registry = Registry::from_env();
/// let response = registry
///     .speech_to_text("openai", "whisper-1", audio)?
///     .generate()
///     .await?;
///
/// println!("{}", response.text);
/// # Ok(())
/// # }
/// ```
pub struct PendingSpeechToTextRequest {
    provider: Arc<dyn Provider>,
    request: SpeechToTextRequest,
}

impl PendingSpeechToTextRequest {
    /// Starts a new builder for `provider`, targeting `model`, transcribing
    /// `audio`. You'll normally get one of these from
    /// [`Registry::speech_to_text`](crate::Registry::speech_to_text) rather
    /// than calling this directly.
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>, audio: AudioInput) -> Self {
        Self {
            provider,
            request: SpeechToTextRequest::new(model, audio),
        }
    }

    /// Freezes the builder's current state into a [`SpeechToTextRequest`]
    /// without sending it.
    pub fn to_request(&self) -> SpeechToTextRequest {
        self.request.clone()
    }

    /// Sends the request and returns the transcribed text.
    pub async fn generate(self) -> Result<TranscriptionResponse, Error> {
        self.provider.speech_to_text(self.request).await
    }
}
