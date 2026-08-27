//! `POST /v1/audio/speech` and `POST /v1/audio/transcriptions` --
//! text-to-speech and speech-to-text.
//!
//! Audio bytes travel as base64-encoded strings in the JSON body, matching
//! how this crate's own core already represents embedded binary media
//! elsewhere ([`llmprism::value_objects::MediaData::Base64`]) -- keeps
//! every route in this crate JSON-in/JSON-out, with no multipart special
//! case for just these two. This was the deliberate design call
//! `ROADMAP.md`'s Phase 1 deferred rather than folding in without thinking
//! it through.
//!
//! # `POST /v1/audio/speech`
//!
//! Request body: [`SpeechRequestBody`]. Response: [`SpeechResponseBody`] --
//! the same shape as [`llmprism::audio::AudioResponse`], except `audio.data`
//! is base64 text rather than a raw byte array (which `serde_json` would
//! otherwise render as an enormous, awkward-to-consume JSON array of
//! numbers).
//!
//! ```json
//! { "provider": "openai", "model": "tts-1", "input": "Hello, world!", "voice": "alloy" }
//! ```
//!
//! # `POST /v1/audio/transcriptions`
//!
//! Request body: [`TranscriptionRequestBody`] -- `audio.data` is base64,
//! the same convention the response above uses. Response:
//! [`llmprism::audio::TranscriptionResponse`], reused as-is (it's already
//! wire-safe: just text and metadata, no binary payload of its own).
//!
//! ```json
//! {
//!   "provider": "openai",
//!   "model": "whisper-1",
//!   "audio": { "data": "<base64 audio bytes>", "mime_type": "audio/mpeg", "filename": "clip.mp3" }
//! }
//! ```

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use llmprism::audio::{
    AudioInput, AudioOutput, PendingSpeechToTextRequest, PendingTextToSpeechRequest,
    TranscriptionResponse,
};
use llmprism::tenancy::TenantRegistry;
use llmprism::value_objects::Meta;
use llmprism::Registry;
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ErrorBody, ErrorDetail};
use crate::tenant::TenantContext;

/// The JSON body for `POST /v1/audio/speech`.
#[derive(Deserialize)]
pub struct SpeechRequestBody {
    /// Name of the provider registered in the `Registry` this router was
    /// built from.
    pub provider: String,
    /// The model to target.
    pub model: String,
    /// The text to speak.
    pub input: String,
    /// The voice to speak with, in whatever names the provider defines
    /// (e.g. OpenAI's `"alloy"`, `"echo"`). Leaves this up to the
    /// provider's own default if omitted.
    pub voice: Option<String>,
}

/// The JSON response for `POST /v1/audio/speech`. See the [module
/// docs](self) for why `audio.data` is base64 text rather than a raw byte
/// array.
#[derive(Serialize)]
pub struct SpeechResponseBody {
    /// The generated audio.
    pub audio: AudioOutputBody,
    /// Provider-reported metadata (response id, model name).
    pub meta: Meta,
}

/// Generated audio: base64-encoded bytes plus the MIME type they're encoded
/// in (e.g. `"audio/mpeg"`), so a client knows how to decode and play them.
#[derive(Serialize)]
pub struct AudioOutputBody {
    /// The audio bytes, base64-encoded.
    pub data: String,
    /// The audio's MIME type, e.g. `"audio/mpeg"`.
    pub mime_type: String,
}

impl From<AudioOutput> for AudioOutputBody {
    fn from(output: AudioOutput) -> Self {
        Self {
            data: BASE64.encode(output.data),
            mime_type: output.mime_type,
        }
    }
}

pub(crate) async fn speech(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<SpeechRequestBody>,
) -> Result<Json<SpeechResponseBody>, ApiError> {
    let response = build_speech_request(&registry, body)?.generate().await?;
    Ok(Json(SpeechResponseBody {
        audio: response.audio.into(),
        meta: response.meta,
    }))
}

pub(crate) async fn speech_multi_tenant(
    State(tenants): State<Arc<dyn TenantRegistry>>,
    TenantContext(context): TenantContext,
    Json(body): Json<SpeechRequestBody>,
) -> Result<Json<SpeechResponseBody>, ApiError> {
    let registry = tenants.resolve(&context).await?;
    let response = build_speech_request(&registry, body)?.generate().await?;
    Ok(Json(SpeechResponseBody {
        audio: response.audio.into(),
        meta: response.meta,
    }))
}

fn build_speech_request(
    registry: &Registry,
    body: SpeechRequestBody,
) -> Result<PendingTextToSpeechRequest, ApiError> {
    let mut request = registry.text_to_speech(&body.provider, body.model, body.input)?;
    if let Some(voice) = body.voice {
        request = request.with_voice(voice);
    }
    Ok(request)
}

/// The JSON body for `POST /v1/audio/transcriptions`.
#[derive(Deserialize)]
pub struct TranscriptionRequestBody {
    /// Name of the provider registered in the `Registry` this router was
    /// built from.
    pub provider: String,
    /// The model to target.
    pub model: String,
    /// The audio to transcribe.
    pub audio: AudioInputBody,
}

/// Audio to transcribe: base64-encoded bytes, the MIME type they're
/// encoded in, and an optional filename (some providers accept audio as a
/// file upload under the hood and want a name attached, even though the
/// name's content usually doesn't matter). `None` gets a generic default --
/// see [`AudioInput::new`].
#[derive(Deserialize)]
pub struct AudioInputBody {
    /// The audio bytes, base64-encoded.
    pub data: String,
    /// The audio's MIME type, e.g. `"audio/mpeg"`, `"audio/wav"`.
    pub mime_type: String,
    /// A filename to attach to the upload, if the provider cares about the
    /// file extension specifically.
    pub filename: Option<String>,
}

pub(crate) async fn transcriptions(
    State(registry): State<Arc<Registry>>,
    Json(body): Json<TranscriptionRequestBody>,
) -> Result<Json<TranscriptionResponse>, AudioError> {
    let response = build_transcription_request(&registry, body)?
        .generate()
        .await?;
    Ok(Json(response))
}

pub(crate) async fn transcriptions_multi_tenant(
    State(tenants): State<Arc<dyn TenantRegistry>>,
    TenantContext(context): TenantContext,
    Json(body): Json<TranscriptionRequestBody>,
) -> Result<Json<TranscriptionResponse>, AudioError> {
    let registry = tenants.resolve(&context).await?;
    let response = build_transcription_request(&registry, body)?
        .generate()
        .await?;
    Ok(Json(response))
}

fn build_transcription_request(
    registry: &Registry,
    body: TranscriptionRequestBody,
) -> Result<PendingSpeechToTextRequest, AudioError> {
    let data = BASE64.decode(&body.audio.data)?;
    let mut audio = AudioInput::new(data, body.audio.mime_type);
    if let Some(filename) = body.audio.filename {
        audio = audio.with_filename(filename);
    }
    Ok(registry.speech_to_text(&body.provider, body.model, audio)?)
}

/// The error type `POST /v1/audio/transcriptions` returns -- like
/// [`ApiError`] for every other route, except this one has a failure mode
/// no other route does: the client's own `audio.data` wasn't valid base64,
/// which never reaches a provider and isn't a `llmprism::Error` at all, so
/// it needs its own `400 Bad Request` mapping rather than falling into
/// `ApiError`'s "no more specific variant -> 502" default.
pub enum AudioError {
    /// A `llmprism::Error` from actually running the request, mapped the
    /// same way [`ApiError`] maps every other route's.
    Provider(ApiError),
    /// `audio.data` in the request body wasn't valid base64.
    InvalidBase64(base64::DecodeError),
}

impl From<llmprism::Error> for AudioError {
    fn from(error: llmprism::Error) -> Self {
        Self::Provider(ApiError(error))
    }
}

impl From<base64::DecodeError> for AudioError {
    fn from(error: base64::DecodeError) -> Self {
        Self::InvalidBase64(error)
    }
}

impl IntoResponse for AudioError {
    fn into_response(self) -> Response {
        match self {
            Self::Provider(error) => error.into_response(),
            Self::InvalidBase64(error) => {
                let body = ErrorBody {
                    error: ErrorDetail {
                        message: format!("invalid base64 in `audio.data`: {error}"),
                    },
                };
                (StatusCode::BAD_REQUEST, Json(body)).into_response()
            }
        }
    }
}
