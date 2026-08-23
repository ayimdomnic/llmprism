//! Moderation -- check text against a provider's content-safety classifier.
//! Start with [`crate::Registry::moderation`].
//!
//! Not every provider has a moderation endpoint (Anthropic, notably, doesn't);
//! calling this against one that doesn't returns
//! [`Error::Unsupported`](crate::Error::Unsupported) rather than failing to
//! compile, the same way every other optional capability on
//! [`crate::Provider`] works.

pub mod request;
pub mod response;

pub use request::{ModerationRequest, PendingModerationRequest};
pub use response::{ModerationResponse, ModerationResult};
