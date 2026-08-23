use serde_json::Value;

use crate::text::Step;
use crate::value_objects::{FinishReason, Meta, ToolCall, Usage};

/// A fluent builder for a canned [`Step`], for scripting what a
/// [`FakeProvider`](super::FakeProvider) should return in a test.
///
/// # Example
///
/// ```
/// use llmprism::testing::FakeTextResponse;
///
/// let step = FakeTextResponse::new("Hello!").build();
/// assert_eq!(step.text.as_deref(), Some("Hello!"));
/// ```
pub struct FakeTextResponse {
    step: Step,
}

impl FakeTextResponse {
    /// Starts a canned response with the given reply text and a `Stop` finish
    /// reason. Pass an empty string if you only want a tool call and no visible
    /// text (see [`with_tool_call`](Self::with_tool_call)).
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            step: Step {
                text: if text.is_empty() { None } else { Some(text) },
                tool_calls: Vec::new(),
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                meta: Meta::default(),
            },
        }
    }

    /// Scripts this response as also requesting a tool call, and switches the
    /// finish reason to [`FinishReason::ToolCalls`] to match. Call this more than
    /// once to script several tool calls in one round trip.
    pub fn with_tool_call(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: Value,
    ) -> Self {
        self.step.tool_calls.push(ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        });
        self.step.finish_reason = FinishReason::ToolCalls;
        self
    }

    /// Overrides the token usage reported for this canned response, if your test
    /// needs to check usage accounting.
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.step.usage = usage;
        self
    }

    /// Overrides the finish reason, for testing how your application handles a
    /// specific one (e.g. [`FinishReason::Length`]).
    pub fn with_finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.step.finish_reason = finish_reason;
        self
    }

    /// Finishes building and returns the underlying [`Step`].
    pub fn build(self) -> Step {
        self.step
    }
}

impl From<FakeTextResponse> for Step {
    fn from(fixture: FakeTextResponse) -> Self {
        fixture.build()
    }
}
