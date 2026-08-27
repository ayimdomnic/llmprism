use serde::{Deserialize, Serialize};

/// Why the model stopped generating a reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model reached a natural stopping point and finished its reply
    /// normally.
    Stop,
    /// The model was cut off by the `max_tokens` limit before it finished.
    /// Consider raising the limit if you need the full reply.
    Length,
    /// The model wants to call one or more tools before continuing. If you're
    /// using [`PendingTextRequest::generate`](crate::text::PendingTextRequest::generate),
    /// you won't normally see this as the *final* finish reason -- the tool loop
    /// keeps going until the model produces one of the other reasons (or
    /// `max_steps` is reached).
    ToolCalls,
    /// The provider's content filter stopped or altered the reply.
    ContentFilter,
    /// The provider reported an error partway through generating.
    Error,
    /// Any other reason not covered above.
    Other,
}
