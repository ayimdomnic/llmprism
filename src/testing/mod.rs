//! Test helpers for code that uses this crate, so you can test your own
//! application logic without making real network calls or spending real API
//! credits. See [`FakeProvider`] for the main entry point.

pub mod fake_provider;
pub mod fixtures;

pub use fake_provider::FakeProvider;
pub use fixtures::{
    FakeAudioResponse, FakeEmbeddingsResponse, FakeImagesResponse, FakeModerationResponse,
    FakeRerankResponse, FakeStructuredResponse, FakeTextResponse, FakeTranscriptionResponse,
};
