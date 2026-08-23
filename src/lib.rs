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
//! # Streaming
//!
//! Call [`stream`](text::PendingTextRequest::stream) instead of `generate` to get
//! the reply incrementally, as a stream of [`StreamEvent`]s, instead of waiting
//! for the whole thing:
//!
//! ```no_run
//! # #[cfg(feature = "openai")]
//! # async fn example() -> Result<(), llmprism::Error> {
//! use futures::StreamExt;
//! use llmprism::{Registry, StreamEvent};
//!
//! let registry = Registry::from_env();
//! let mut stream = registry
//!     .text("openai", "gpt-4o-mini")?
//!     .with_prompt("Count to three.")
//!     .stream();
//!
//! while let Some(event) = stream.next().await {
//!     if let StreamEvent::TextDelta { text } = event? {
//!         print!("{text}");
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Tool calling works the same way here as it does for `generate` -- if the
//! model asks to call a tool, `llmprism` runs it, reports the result as a
//! [`StreamEvent::ToolResult`], and keeps streaming the model's next round trip
//! automatically. See [`stream_loop`] for how that's implemented.
//!
//! # Structured output
//!
//! Use [`Registry::structured`] instead of [`Registry::text`] to get a reply
//! guaranteed to match a JSON shape you describe with [`schema`], rather than
//! free-form text:
//!
//! ```no_run
//! # #[cfg(feature = "openai")]
//! # async fn example() -> Result<(), llmprism::Error> {
//! use llmprism::schema::{BooleanSchema, ObjectSchema, Schema, StringSchema};
//! use llmprism::Registry;
//!
//! let schema = ObjectSchema::new("review")
//!     .with_property(Schema::String(StringSchema::new("summary")), true)
//!     .with_property(Schema::Boolean(BooleanSchema::new("recommended")), true);
//!
//! let registry = Registry::from_env();
//! let response = registry
//!     .structured("openai", "gpt-4o-mini", schema)?
//!     .with_prompt("Review this crate's ergonomics in one sentence.")
//!     .generate()
//!     .await?;
//!
//! println!("{}", response.data);
//! # Ok(())
//! # }
//! ```
//!
//! `response.data` is a `serde_json::Value` you can deserialize into your own
//! type with `serde_json::from_value`. See [`structured`] for how different
//! providers get a model to comply with the schema under the hood.
//!
//! # How the pieces fit together
//!
//! - [`Registry`] is where you register providers -- or let [`Registry::from_env`]
//!   do it for you -- and it's the one place your application code asks for a
//!   provider by name. In tests, you register a [`testing::FakeProvider`] under
//!   that same name instead, so no other code has to know it's being tested.
//! - Every capability (today: [`text`] generation, streaming or not,
//!   [`structured`] output, [`moderation`], and [`embeddings`]; images and
//!   audio are on the roadmap) follows the same shape: ask the registry for a
//!   fluent request builder, chain `.with_*()` calls to describe what you
//!   want, then call an async method like
//!   [`text::PendingTextRequest::generate`] (or
//!   [`text::PendingTextRequest::stream`] for incremental output) to run it.
//!   Not every provider implements every capability -- Anthropic, for
//!   instance, has no moderation endpoint -- calling one it doesn't returns
//!   [`Error::Unsupported`] rather than failing to compile.
//! - [`Tool`] is how you give the model something it can call on your behalf --
//!   a weather lookup, a database query, anything. Hand a list of tools to a text
//!   request and `llmprism` handles the entire back-and-forth: if the model asks to
//!   call a tool, the tool runs, its result goes back to the model, and this
//!   repeats until the model gives a final answer (see [`tool_loop`]).
//! - [`schema`] describes the *shape* of data -- a tool's arguments, or a
//!   [`structured`]-output request's required shape -- in a way every provider
//!   understands, without you hand-writing each provider's own JSON Schema
//!   dialect.
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

pub mod embeddings;
pub mod error;
pub mod moderation;
pub mod provider;
pub mod registry;
pub mod schema;
pub mod stream_event;
pub mod stream_loop;
pub mod structured;
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
