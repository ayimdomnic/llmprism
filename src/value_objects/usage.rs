use serde::{Deserialize, Serialize};

/// How many tokens a request/response round trip cost, in whatever detail the
/// provider reports.
///
/// `prompt_tokens` and `completion_tokens` are supported by every provider;
/// the rest are `None` unless the provider you're using reports them.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens consumed by the input (the conversation and any system prompt sent
    /// to the model).
    pub prompt_tokens: u32,
    /// Tokens the model generated in its reply.
    pub completion_tokens: u32,
    /// Tokens written to a prompt cache, for providers that support one (this
    /// usually costs a bit more than an ordinary prompt token, in exchange for
    /// making a later request with the same prefix cheaper).
    #[serde(default)]
    pub cache_write_tokens: Option<u32>,
    /// Tokens served from a prompt cache instead of being reprocessed -- these
    /// are typically billed at a reduced rate.
    #[serde(default)]
    pub cache_read_tokens: Option<u32>,
    /// Tokens spent on internal "thinking"/reasoning the model did before
    /// producing its visible reply, for providers that report this separately.
    #[serde(default)]
    pub thought_tokens: Option<u32>,
}
