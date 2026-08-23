//! The ElevenLabs provider, talking to ElevenLabs' Audio APIs
//! (`api.elevenlabs.io/v1/text-to-speech/{voice_id}` and
//! `/v1/speech-to-text`). ElevenLabs is an audio specialist -- there's no
//! text-generation endpoint to speak of, so
//! [`text_to_speech`](Provider::text_to_speech) and
//! [`speech_to_text`](Provider::speech_to_text) are the only two
//! capabilities this provider implements. Enable with the `elevenlabs`
//! Cargo feature.

mod audio;

use async_trait::async_trait;
use reqwest_middleware::ClientWithMiddleware;

use crate::audio::{
    AudioResponse, SpeechToTextRequest, TextToSpeechRequest, TranscriptionResponse,
};
use crate::client::{
    build_http_client, merge_provider_options, merge_provider_options_into_form, ErrorMapper,
};
use crate::error::Error;
use crate::provider::Provider;

const DEFAULT_BASE_URL: &str = "https://api.elevenlabs.io/v1";

/// A [`Provider`] backed by ElevenLabs' Audio APIs.
///
/// # Example
///
/// ```no_run
/// use llmprism::providers::elevenlabs::ElevenLabsProvider;
/// use llmprism::Registry;
///
/// let mut registry = Registry::new();
/// registry.register(
///     "elevenlabs",
///     ElevenLabsProvider::new(std::env::var("ELEVENLABS_API_KEY").unwrap()),
/// );
/// ```
///
/// If you're happy reading the API key from `ELEVENLABS_API_KEY` yourself,
/// you likely don't need to construct this directly -- see
/// [`Registry::from_env`](crate::Registry::from_env).
pub struct ElevenLabsProvider {
    api_key: String,
    base_url: String,
    client: ClientWithMiddleware,
}

impl ElevenLabsProvider {
    /// Creates a provider that talks to the real ElevenLabs API using
    /// `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Creates a provider pointed at a different base URL.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: build_http_client(),
        }
    }
}

#[async_trait]
impl Provider for ElevenLabsProvider {
    fn name(&self) -> &str {
        "elevenlabs"
    }

    async fn text_to_speech(&self, request: TextToSpeechRequest) -> Result<AudioResponse, Error> {
        let path = audio::speech_endpoint_path(&request);
        let wire_request = audio::build_speech_request(&request);
        let body = merge_provider_options(&wire_request, &request.provider_options)?;

        let http_response = self
            .client
            .post(format!("{}/{path}", self.base_url))
            .header("xi-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = http_response.status();
        if !status.is_success() {
            let headers = http_response.headers().clone();
            let body_text = http_response.text().await?;
            let mapper = ErrorMapper {
                provider: self.name(),
            };
            return Err(mapper.map_error_response(status, &headers, &body_text));
        }

        let mime_type = http_response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.to_string());
        let data = http_response.bytes().await?;

        Ok(audio::parse_speech_response(data.to_vec(), mime_type))
    }

    async fn speech_to_text(
        &self,
        request: SpeechToTextRequest,
    ) -> Result<TranscriptionResponse, Error> {
        let part = reqwest::multipart::Part::bytes(request.audio.data.clone())
            .file_name(request.audio.filename.clone())
            .mime_str(&request.audio.mime_type)?;
        let form = reqwest::multipart::Form::new()
            .text("model_id", request.model.clone())
            .part("file", part);
        let form = merge_provider_options_into_form(form, &request.provider_options);

        let http_response = self
            .client
            .post(format!("{}/speech-to-text", self.base_url))
            .header("xi-api-key", &self.api_key)
            .multipart(form)
            .send()
            .await?;

        let status = http_response.status();
        let headers = http_response.headers().clone();
        let body_text = http_response.text().await?;

        if !status.is_success() {
            let mapper = ErrorMapper {
                provider: self.name(),
            };
            return Err(mapper.map_error_response(status, &headers, &body_text));
        }

        let wire_response: audio::TranscriptionApiResponse = serde_json::from_str(&body_text)
            .map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        Ok(audio::parse_transcription_response(wire_response))
    }
}
