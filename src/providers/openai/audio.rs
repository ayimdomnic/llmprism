//! Wire format and mapping for OpenAI's two Audio endpoints:
//! text-to-speech (`POST /v1/audio/speech`, JSON in, raw audio bytes out)
//! and speech-to-text (`POST /v1/audio/transcriptions`, a multipart file
//! upload in, JSON out). The two don't share a wire shape at all -- one
//! isn't even JSON on the way out, the other isn't JSON on the way in -- but
//! both belong to this crate's single `audio` capability, so they're kept
//! together in one file rather than split the way distinct capabilities
//! (moderation, embeddings, images) are.

use serde::{Deserialize, Serialize};

use crate::audio::{AudioOutput, AudioResponse, TextToSpeechRequest, TranscriptionResponse};
use crate::value_objects::Meta;

/// OpenAI requires a voice on every text-to-speech request; this crate's
/// [`TextToSpeechRequest::voice`] is optional (matching every other
/// capability's "let the provider default it" convention), so this is what
/// fills in when the caller didn't pick one.
const DEFAULT_VOICE: &str = "alloy";

/// Used when a text-to-speech response is missing (or has an unreadable)
/// `Content-Type` header -- OpenAI's default audio format is MP3.
const DEFAULT_AUDIO_MIME_TYPE: &str = "audio/mpeg";

#[derive(Debug, Serialize)]
pub struct SpeechApiRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
}

pub fn build_speech_request(request: &TextToSpeechRequest) -> SpeechApiRequest {
    SpeechApiRequest {
        model: request.model.clone(),
        input: request.input.clone(),
        voice: request
            .voice
            .clone()
            .unwrap_or_else(|| DEFAULT_VOICE.to_string()),
    }
}

pub fn parse_speech_response(data: Vec<u8>, mime_type: Option<String>) -> AudioResponse {
    AudioResponse {
        audio: AudioOutput {
            data,
            mime_type: mime_type.unwrap_or_else(|| DEFAULT_AUDIO_MIME_TYPE.to_string()),
        },
        meta: Meta::default(),
    }
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionApiResponse {
    pub text: String,
}

pub fn parse_transcription_response(response: TranscriptionApiResponse) -> TranscriptionResponse {
    TranscriptionResponse {
        text: response.text,
        meta: Meta::default(),
    }
}
