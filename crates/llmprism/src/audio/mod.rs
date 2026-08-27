//! Audio -- text-to-speech (turn text into spoken audio) and speech-to-text
//! (transcribe spoken audio into text). Start with
//! [`crate::Registry::text_to_speech`] or
//! [`crate::Registry::speech_to_text`].
//!
//! Unlike every other capability in this crate, these two are a matched
//! pair covering opposite directions of the same medium rather than one
//! request/response shape -- which is why this module (and the
//! [`Provider`](crate::Provider) trait) has two methods here instead of
//! one, mirroring how Prism's PHP `Audio` module works.

pub mod speech_to_text;
pub mod text_to_speech;

pub use speech_to_text::{
    AudioInput, PendingSpeechToTextRequest, SpeechToTextRequest, TranscriptionResponse,
};
pub use text_to_speech::{
    AudioOutput, AudioResponse, PendingTextToSpeechRequest, TextToSpeechRequest,
};
