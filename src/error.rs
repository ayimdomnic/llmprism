//! The error types this crate returns. Almost everything that can fail -- a
//! provider call, a tool execution, decoding a response -- ends up as an
//! [`Error`], so you generally only need to match on one type regardless of which
//! capability or provider you're calling.

use std::time::Duration;

use crate::value_objects::RateLimit;

/// Something went wrong while a [`Tool`](crate::Tool) was resolved or executed.
///
/// You don't usually need to construct or match on this directly: when a tool
/// fails, the multi-step tool loop turns it into a message sent back to the model
/// (so the model can see the failure and try something else) rather than aborting
/// the whole request. It only reaches your code as [`Error::Tool`] if something
/// fails outside of that loop.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolError {
    /// The model asked to call a tool by a name that wasn't registered on the
    /// request.
    #[error("tool '{name}' not found")]
    NotFound { name: String },
    /// More than one tool with the same name was registered on the request, so it
    /// wasn't possible to tell which one the model meant.
    #[error("multiple tools named '{name}' are registered")]
    MultipleFound { name: String },
    /// The arguments the model sent didn't match what the tool expected.
    #[error("invalid arguments for tool '{name}': {message}")]
    InvalidArguments { name: String, message: String },
    /// The tool's own code returned an error while running.
    #[error("tool '{name}' failed: {message}")]
    Runtime { name: String, message: String },
}

/// The single error type returned by every fallible operation in this crate.
///
/// The intent is that you can usually write one `match` (or just check for the
/// variants you care about, like [`Error::RateLimited`]) regardless of which
/// provider or capability produced the error, instead of learning a different error
/// type per provider.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The provider rejected the request because you're sending too many requests
    /// too quickly. `retry_after`, when the provider sends it, tells you how long
    /// to wait before trying again; `limits` carries whatever rate-limit
    /// bookkeeping the provider reported (current usage against your quota, when
    /// it resets, and so on).
    #[error("{provider} rate limited{}", .retry_after.map(|d| format!(", retry after {d:?}")).unwrap_or_default())]
    RateLimited {
        provider: String,
        retry_after: Option<Duration>,
        limits: Vec<RateLimit>,
    },

    /// The provider is temporarily unable to handle requests (for example,
    /// experiencing high demand). This is usually worth retrying after a short
    /// delay.
    #[error("{provider} is overloaded")]
    Overloaded { provider: String },

    /// The request was rejected for being too large -- typically too much text or
    /// too many attached files for the provider to accept in one call.
    #[error("{provider} request too large: {details}")]
    RequestTooLarge { provider: String, details: String },

    /// While reading a streamed response, a chunk from the provider couldn't be
    /// parsed. (Reserved for the streaming capability, which isn't implemented
    /// yet.)
    #[error("failed to decode stream event from {provider}: {message}")]
    StreamDecode { provider: String, message: String },

    /// The provider's structured-output response didn't match the schema you
    /// asked for. (Reserved for the structured-output capability, which isn't
    /// implemented yet.)
    #[error("failed to decode structured output from {provider}: {message}")]
    StructuredDecode { provider: String, message: String },

    /// The provider's response body couldn't be parsed as the JSON shape this
    /// crate expected. This usually means the provider changed its API, or you're
    /// pointed at an incompatible endpoint via a custom base URL.
    #[error("failed to decode response from {provider}: {message}")]
    Decode { provider: String, message: String },

    /// You asked a provider to do something it doesn't support -- for example,
    /// calling `.moderation()` on a provider that has no moderation endpoint.
    /// Every capability method on [`Provider`](crate::Provider) returns this by
    /// default; a provider only avoids it for the capabilities it actually
    /// implements.
    #[error("{provider} does not support {capability}")]
    Unsupported {
        provider: String,
        capability: &'static str,
    },

    /// You asked the [`Registry`](crate::Registry) for a provider name that was
    /// never registered. Double-check the name you passed to `.text(...)` (or
    /// similar) matches what you passed to `.register(...)`.
    #[error("unknown provider '{0}'")]
    UnknownProvider(String),

    /// The provider's API returned an error response that doesn't fit one of the
    /// more specific variants above -- `status` is the HTTP status code, and
    /// `message` (with `kind`, when the provider sends one) is what the provider
    /// said went wrong.
    #[error("{provider} error ({status}): {message}")]
    Provider {
        provider: String,
        status: u16,
        kind: Option<String>,
        message: String,
    },

    /// A provider is missing configuration it needs to run (for example, no API
    /// key was found).
    #[error("missing configuration: {0}")]
    Config(String),

    /// A [`Tool`](crate::Tool) call failed outside of the normal tool-loop
    /// handling. See [`ToolError`] for the specific cause.
    #[error(transparent)]
    Tool(#[from] ToolError),

    /// A value couldn't be serialized to or deserialized from JSON.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// A lower-level HTTP failure (connection refused, DNS failure, timeout, and
    /// so on) that happened before the provider ever sent a response to
    /// interpret. Only present when an HTTP-based provider feature is enabled.
    #[cfg(feature = "http")]
    #[error(transparent)]
    Http(#[from] reqwest::Error),

    /// An error raised by the HTTP retry/middleware layer itself, rather than by
    /// the underlying HTTP request. Only present when an HTTP-based provider
    /// feature is enabled.
    #[cfg(feature = "http")]
    #[error("http middleware error: {0}")]
    Middleware(#[from] reqwest_middleware::Error),
}

impl Error {
    /// Builds an [`Error::Unsupported`] for the given provider and capability name.
    /// Used internally by [`Provider`](crate::Provider)'s default method bodies.
    pub fn unsupported(provider: impl Into<String>, capability: &'static str) -> Self {
        Error::Unsupported {
            provider: provider.into(),
            capability,
        }
    }
}
