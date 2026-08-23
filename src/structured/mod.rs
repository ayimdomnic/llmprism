//! Structured output -- ask the model for a reply matching a specific JSON
//! shape (described with [`crate::schema`]) instead of free-form text. Start
//! with [`crate::Registry::structured`].
//!
//! Providers don't agree on how to make a model comply with a schema, and this
//! crate doesn't hide that behind one "the" strategy: OpenAI has a native
//! structured-output response format it enforces server-side, while Anthropic
//! has no equivalent, so this crate gets the same result there by forcing a
//! single call to a synthetic tool shaped like the schema and reading its
//! arguments back out. Either way, you get the same [`StructuredResponse`] back
//! -- which strategy a given provider uses is an implementation detail of that
//! provider's `Provider` impl, not something you need to choose yourself.

pub mod request;
pub mod response;

pub use request::{PendingStructuredRequest, StructuredRequest};
pub use response::StructuredResponse;
