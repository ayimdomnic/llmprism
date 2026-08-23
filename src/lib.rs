//! `llmprism` is a Rust library for talking to Large Language Model (LLM) providers
//! -- OpenAI, Anthropic, and more to come -- through one consistent, fluent API,
//! instead of learning each provider's own SDK and JSON shape. It's a Rust port of
//! the PHP/Laravel library [Prism](https://prismphp.com).
//!
//! # A first example
//!
//! ```no_run
//! # #[cfg(feature = "openai")]
//! # async fn example() -> Result<(), llmprism::Error> {
//! use llmprism::Registry;
//!
//! // Reads OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment and registers
//! // whichever providers have a key set (gated by the crate features you enabled).
//! let registry = Registry::from_env();
//!
//! let response = registry
//!     .text("openai", "gpt-4o-mini")?
//!     .with_prompt("Say hello in one word.")
//!     .generate()
//!     .await?;
//!
//! println!("{}", response.text.unwrap_or_default());
//! # Ok(())
//! # }
//! ```
//!
//! Swap `"openai"` for `"anthropic"` and nothing else about this code needs to
//! change -- that consistency across providers is the reason this crate exists.
//!
//! # How the pieces fit together
//!
//! - [`Registry`] is where you register providers -- or let [`Registry::from_env`]
//!   do it for you -- and it's the one place your application code asks for a
//!   provider by name. In tests, you register a [`testing::FakeProvider`] under
//!   that same name instead, so no other code has to know it's being tested.
//! - Every capability (today, that's just [`text`] generation; streaming,
//!   structured output, embeddings, images, audio, and moderation are on the
//!   roadmap) follows the same shape: ask the registry for a fluent request
//!   builder, chain `.with_*()` calls to describe what you want, then call an
//!   async method like [`text::PendingTextRequest::generate`] to actually run it.
//! - [`Tool`] is how you give the model something it can call on your behalf --
//!   a weather lookup, a database query, anything. Hand a list of tools to a text
//!   request and `llmprism` handles the entire back-and-forth: if the model asks to
//!   call a tool, the tool runs, its result goes back to the model, and this
//!   repeats until the model gives a final answer (see [`tool_loop`]).
//! - [`schema`] describes the *shape* of data -- a tool's arguments today,
//!   structured output later -- in a way every provider understands, without you
//!   hand-writing each provider's own JSON Schema dialect.
//! - [`value_objects`] holds the plain data types shared by every provider and
//!   every capability: conversation messages, token usage, finish reasons, and so
//!   on.
//!
//! # Feature flags
//!
//! Nothing is enabled by default. Each provider lives behind its own Cargo feature
//! flag (e.g. `openai`, `anthropic`), so a binary that only uses one provider isn't
//! forced to compile in HTTP client code for the rest. Enable `full` to turn on
//! every provider at once.

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
