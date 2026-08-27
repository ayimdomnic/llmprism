//! [`RepairStrategy`] -- a hook that can salvage a structured-output request
//! whose reply didn't decode, instead of just failing.

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Error;

/// Given a failed structured-output attempt, tries to produce a JSON value
/// matching the requested schema anyway.
///
/// Attach one with
/// [`PendingStructuredRequest::with_repair`](super::PendingStructuredRequest::with_repair).
/// When [`Error::StructuredDecode`] happens, `error` is passed to
/// [`repair`](Self::repair) along with the exact text the provider sent
/// (`error`'s own `raw` field, pulled out here for convenience) -- return
/// `Some(value)` to use it as the result instead of failing, or `None` to
/// give up and return the original error. Only tried once per request: a
/// repair that itself produces something that still doesn't match the
/// schema is returned to the caller as-is rather than looping indefinitely.
///
/// A simple repair strips a Markdown code fence some models wrap JSON in
/// even when explicitly asked not to:
///
/// ```
/// use async_trait::async_trait;
/// use llmprism::error::Error;
/// use llmprism::structured::RepairStrategy;
/// use serde_json::Value;
///
/// struct StripCodeFence;
///
/// #[async_trait]
/// impl RepairStrategy for StripCodeFence {
///     async fn repair(&self, raw: &str, _error: &Error) -> Option<Value> {
///         let stripped = raw.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```");
///         serde_json::from_str(stripped.trim()).ok()
///     }
/// }
/// ```
///
/// A repair can also be worth reaching for the model itself to fix -- since
/// `repair` is async, it's free to make its own provider call (for example,
/// resending the conversation with the malformed reply and an instruction to
/// correct it) rather than only doing local text surgery.
#[async_trait]
pub trait RepairStrategy: Send + Sync {
    /// `raw` is the exact text the provider sent that didn't decode (empty if
    /// the provider sent no relevant content at all -- see
    /// [`Error::StructuredDecode`]'s field docs for when that happens per
    /// provider). `error` is the full error that would otherwise be
    /// returned, in case its `message` or provider name matters to the
    /// repair logic.
    async fn repair(&self, raw: &str, error: &Error) -> Option<Value>;
}

#[async_trait]
impl<F> RepairStrategy for F
where
    F: Fn(&str, &Error) -> Option<Value> + Send + Sync,
{
    async fn repair(&self, raw: &str, error: &Error) -> Option<Value> {
        self(raw, error)
    }
}
