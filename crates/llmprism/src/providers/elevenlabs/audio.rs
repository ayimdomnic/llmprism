//! Wire format and mapping for ElevenLabs' two Audio endpoints:
//! text-to-speech (`POST /v1/text-to-speech/{voice_id}`, JSON in, raw audio
//! bytes out) and speech-to-text (`POST /v1/speech-to-text`, multipart file
//! upload in, JSON out). Structurally similar to `providers::openai::audio`
//! for the same reason: neither direction here is JSON on both sides, so
//! both live in one file rather than being split like Text/structured
//! output are.
//!
//! One shape ElevenLabs doesn't share with OpenAI's audio endpoints: the
//! voice isn't a body field, it's part of the URL path itself
//! (`/text-to-speech/{voice_id}`) -- see [`speech_endpoint_path`].

use serde::{Deserialize, Serialize};

use crate::audio::{AudioOutput, AudioResponse, TextToSpeechRequest, TranscriptionResponse};
use crate::value_objects::Meta;

/// ElevenLabs requires a voice, selected via the URL path rather than a body
/// field; this crate's [`TextToSpeechRequest::voice`] is optional (matching
/// every other capability's "let the provider default it" convention), so
/// this is what fills in when the caller didn't pick one. It's "Rachel", the
/// voice ElevenLabs' own quickstart docs default to.
const DEFAULT_VOICE_ID: &str = "21m00Tcm4TlvDq8ikWAM";

/// Used when a text-to-speech response is missing (or has an unreadable)
/// `Content-Type` header -- ElevenLabs' default audio format is MP3.
const DEFAULT_AUDIO_MIME_TYPE: &str = "audio/mpeg";

#[derive(Debug, Serialize)]
pub struct SpeechApiRequest {
    pub text: String,
    pub model_id: String,
}

/// Builds the URL path segment (after the base URL) for a text-to-speech
/// request -- `text-to-speech/{voice_id}`, using [`DEFAULT_VOICE_ID`] if the
/// request didn't specify one.
pub fn speech_endpoint_path(request: &TextToSpeechRequest) -> String {
    let voice_id = request.voice.as_deref().unwrap_or(DEFAULT_VOICE_ID);
    format!("text-to-speech/{voice_id}")
}

pub fn build_speech_request(request: &TextToSpeechRequest) -> SpeechApiRequest {
    SpeechApiRequest {
        text: request.input.clone(),
        model_id: request.model.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_endpoint_path_uses_the_default_voice_when_none_is_set() {
        let request = TextToSpeechRequest::new("eleven_flash_v2_5", "hello");
        assert_eq!(
            speech_endpoint_path(&request),
            format!("text-to-speech/{DEFAULT_VOICE_ID}")
        );
    }

    #[test]
    fn speech_endpoint_path_uses_the_requested_voice_when_set() {
        let mut request = TextToSpeechRequest::new("eleven_flash_v2_5", "hello");
        request.voice = Some("custom-voice-id".to_string());
        assert_eq!(
            speech_endpoint_path(&request),
            "text-to-speech/custom-voice-id"
        );
    }
}
