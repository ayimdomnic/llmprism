//! Concrete [`Provider`](crate::Provider) implementations. Each one lives behind
//! its own Cargo feature flag (see the crate root docs for the full list), so
//! enabling `openai` is what makes [`openai::OpenAiProvider`] available, and so
//! on. Most applications don't construct these directly -- prefer
//! [`Registry::from_env`](crate::Registry::from_env), which does it for you based
//! on which environment variables are set.

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;
