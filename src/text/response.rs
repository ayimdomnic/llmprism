//! The Text generation response: [`Step`] (one round trip) and [`TextResponse`]
//! (the final, aggregated result).

use serde::{Deserialize, Serialize};

use crate::value_objects::{FinishReason, Meta, ToolCall, Usage};

/// The result of exactly one request/response round trip with a provider.
///
/// If a request never calls a tool, its [`TextResponse`] contains exactly one
/// `Step`. If the model calls tools along the way, each round trip through
/// [`crate::tool_loop::run_text`] produces one more `Step`, and `TextResponse`
/// holds the full sequence of them -- handy if you want to show a user what the
/// model did along the way, not just its final answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// The text the model generated in this round trip, if any. Often `None`
    /// when the model's entire reply was a tool call with no accompanying text.
    pub text: Option<String>,
    /// Any tools the model asked to call in this round trip.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Why the model stopped generating on this round trip.
    pub finish_reason: FinishReason,
    /// Token usage for this round trip specifically (see [`TextResponse::usage`]
    /// for the total across every round trip).
    pub usage: Usage,
    /// Provider-reported metadata for this round trip (response id, model name,
    /// rate-limit info).
    pub meta: Meta,
}

/// The final result of a Text generation call, once the tool-calling loop has
/// finished.
///
/// # Example
///
/// ```
/// # use llmprism::text::{Step, TextResponse};
/// # use llmprism::value_objects::{FinishReason, Meta, Usage};
/// # let steps = vec![Step {
/// #     text: Some("Hello!".to_string()),
/// #     tool_calls: vec![],
/// #     finish_reason: FinishReason::Stop,
/// #     usage: Usage::default(),
/// #     meta: Meta::default(),
/// # }];
/// let response = TextResponse::from_steps(steps);
/// println!("{}", response.text.unwrap_or_default());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextResponse {
    /// The model's final reply text, taken from the last [`Step`]. `None` if the
    /// last round trip produced no text (unusual, but possible if the model
    /// stopped for another reason).
    pub text: Option<String>,
    /// Why the model stopped generating on the *last* round trip -- this is the
    /// finish reason that actually ended the conversation.
    pub finish_reason: FinishReason,
    /// Token usage summed across every round trip the tool-calling loop ran.
    pub usage: Usage,
    /// Every round trip that happened, oldest first, including the final one.
    /// Has one entry unless tool calls extended the conversation.
    pub steps: Vec<Step>,
}

impl TextResponse {
    /// Builds the final response by folding a sequence of round trips together:
    /// the reply text and finish reason come from the last step, and token usage
    /// is summed across all of them.
    ///
    /// # Panics
    ///
    /// Panics if `steps` is empty -- the tool-calling loop always produces at
    /// least one step before returning, so this should never happen in practice
    /// unless you're constructing a `TextResponse` by hand.
    pub fn from_steps(steps: Vec<Step>) -> Self {
        let last = steps
            .last()
            .expect("tool loop always produces at least one step");
        let text = last.text.clone();
        let finish_reason = last.finish_reason;

        let usage = steps.iter().fold(Usage::default(), |mut acc, step| {
            acc.prompt_tokens += step.usage.prompt_tokens;
            acc.completion_tokens += step.usage.completion_tokens;
            acc
        });

        Self {
            text,
            finish_reason,
            usage,
            steps,
        }
    }
}
