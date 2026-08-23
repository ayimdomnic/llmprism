use serde::{Deserialize, Serialize};

/// Provider-reported metadata about one response, alongside the actual content.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Meta {
    /// The provider's own identifier for this response, if it sent one --
    /// useful for support requests or correlating logs with the provider's
    /// dashboard.
    pub id: Option<String>,
    /// The specific model that actually generated the response. Providers
    /// sometimes route a request to a slightly different model version than the
    /// one you asked for; this tells you which one actually ran.
    pub model: Option<String>,
    /// Rate-limit bookkeeping the provider included with this response, if any.
    #[serde(default)]
    pub rate_limits: Vec<RateLimit>,
}

/// A snapshot of your standing against one of the provider's rate limits (for
/// example, requests per minute).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// What this limit applies to (for example, `"requests"`), as reported by
    /// the provider.
    pub name: String,
    /// The maximum allowed within the current window.
    pub limit: u64,
    /// How much of that maximum is left in the current window.
    pub remaining: u64,
    /// When this limit resets, in whatever format the provider sent it (this
    /// crate passes it through as-is rather than parsing it, since providers
    /// don't agree on one format).
    pub reset_at: Option<String>,
}
