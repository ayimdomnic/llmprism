//! `llmprism` is a Rust library for talking to Large Language Model (LLM) providers
//! through one consistent, fluent API. This crate is under active construction --
//! modules are being added incrementally as each capability lands.

pub mod error;
pub mod provider;
pub mod registry;
pub mod schema;
pub mod stream_event;
pub mod testing;
pub mod text;
pub mod tool;
pub mod tool_loop;
pub mod value_objects;

#[cfg(feature = "http")]
pub mod client;

#[cfg(any(feature = "openai", feature = "anthropic"))]
pub mod providers;

pub use error::Error;
pub use provider::Provider;
pub use registry::Registry;
pub use stream_event::StreamEvent;
pub use tool::Tool;
