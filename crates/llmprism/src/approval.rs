//! [`ApprovalHandler`] -- a hook that gates a specific tool call on an
//! external decision, instead of running it automatically.

use async_trait::async_trait;

use crate::value_objects::ToolCall;

/// Decides whether a tool call that opted into
/// [`Tool::needs_approval`](crate::tool::Tool::needs_approval) is allowed to
/// run.
///
/// Attach one with
/// [`PendingTextRequest::with_approval_handler`](crate::text::PendingTextRequest::with_approval_handler).
/// Being `async` is what makes this useful without any pause/resume
/// machinery: an implementation can await a database poll, a channel, a
/// webhook callback -- anything -- for as long as the request is willing to
/// stay open, entirely in your own code. A denied call (or a
/// [`needs_approval`](crate::tool::Tool::needs_approval)-marked call made
/// with no handler attached at all) never reaches
/// [`Tool::call`](crate::tool::Tool::call) -- the model sees a normal
/// tool-error result instead, the same as any other tool failure, so it can
/// react (tell the user permission was needed, try something else).
///
/// This deliberately doesn't attempt cross-process resumption -- suspending
/// the whole request now and having an approval decision arrive later
/// through an entirely separate request (e.g. via a signed token that
/// round-trips through an untrusted client). That's a real, harder problem
/// this crate doesn't try to solve for you; an `ApprovalHandler` can still
/// build it, by polling whatever store such a token round-tripped through
/// from inside [`approve`](Self::approve).
///
/// ```
/// use async_trait::async_trait;
/// use llmprism::approval::ApprovalHandler;
/// use llmprism::value_objects::ToolCall;
///
/// /// Approves anything except a tool named "delete_account".
/// struct DenyDangerousTools;
///
/// #[async_trait]
/// impl ApprovalHandler for DenyDangerousTools {
///     async fn approve(&self, call: &ToolCall) -> bool {
///         call.name != "delete_account"
///     }
/// }
/// ```
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    /// Returns `true` to let `call` run, `false` to deny it.
    async fn approve(&self, call: &ToolCall) -> bool;
}

#[async_trait]
impl<F> ApprovalHandler for F
where
    F: Fn(&ToolCall) -> bool + Send + Sync,
{
    async fn approve(&self, call: &ToolCall) -> bool {
        self(call)
    }
}
