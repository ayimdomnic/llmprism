//! The multi-step tool-calling loop shared by every provider.
//!
//! This is the piece that makes tool calling feel automatic: a provider only
//! knows how to do one request/response round trip
//! ([`Provider::text_step`]); everything about *looping* -- running tools,
//! feeding their results back to the model, and deciding when to stop -- lives
//! here, once, so every provider gets the same behavior for free.

use std::sync::Arc;

use futures::future::join_all;

use crate::error::{Error, ToolError};
use crate::provider::Provider;
use crate::text::request::{TextRequest, ToolChoice};
use crate::text::response::{Step, TextResponse};
use crate::tool::Tool;
use crate::value_objects::{
    AssistantMessage, FinishReason, Message, ToolCall, ToolOutcome, ToolResult, ToolResultMessage,
};

/// Drives a text request to completion, looping through tool calls as needed.
///
/// In plain terms, this does the following, repeatedly:
///
/// 1. Send the current conversation to the provider and get back one reply
///    (a [`Step`]).
/// 2. If that reply doesn't ask to call any tools, we're done -- return the
///    accumulated result.
/// 3. Otherwise, run the requested tools (concurrently or one at a time,
///    depending on each [`Tool::concurrent`]), append the model's request and
///    the tools' results to the conversation as new messages, and go back to
///    step 1.
///
/// The loop also stops once `request.max_steps` round trips have happened, even
/// if the model is still asking for more tool calls -- this is a safety valve
/// against a model (or a misbehaving tool) looping forever.
///
/// This is what
/// [`PendingTextRequest::generate`](crate::text::PendingTextRequest::generate)
/// calls internally; you normally reach this indirectly through that builder
/// rather than calling it yourself.
pub async fn run_text(
    provider: Arc<dyn Provider>,
    mut request: TextRequest,
) -> Result<TextResponse, Error> {
    let mut steps: Vec<Step> = Vec::new();

    loop {
        let step = provider.text_step(request.clone()).await?;
        let has_tool_calls =
            step.finish_reason == FinishReason::ToolCalls && !step.tool_calls.is_empty();
        let tool_calls = step.tool_calls.clone();
        let text = step.text.clone();

        steps.push(step);

        if !has_tool_calls || steps.len() as u32 >= request.max_steps {
            return Ok(TextResponse::from_steps(steps));
        }

        let results = execute_tools(&request.tools, &tool_calls).await;

        request.messages.push(Message::Assistant(AssistantMessage {
            content: text,
            tool_calls,
        }));
        request
            .messages
            .push(Message::ToolResult(ToolResultMessage {
                tool_results: results,
            }));
        // If the caller forced a specific tool via `ToolChoice::Any`/`Tool(name)`,
        // keep honoring that for exactly one round -- forcing it again here would
        // make the model call a tool forever with no way to ever answer in plain
        // text.
        request.tool_choice = ToolChoice::Auto;
    }
}

/// Runs every requested tool call and collects the results, in the same order the
/// calls were requested in.
///
/// Tools marked [`Tool::concurrent`] run at the same time as each other (useful
/// for independent, read-only lookups); every other tool call runs one at a time,
/// in order. Either way, the results come back in the original request order, so
/// callers never need to worry about which bucket a given call ended up in.
///
/// Crate-visible (rather than private) because [`crate::stream_loop`] drives the
/// same tool-execution behavior for streaming requests and reuses this directly,
/// rather than duplicating it.
pub(crate) async fn execute_tools(tools: &[Arc<dyn Tool>], calls: &[ToolCall]) -> Vec<ToolResult> {
    let mut concurrent_idx = Vec::new();
    let mut sequential_idx = Vec::new();

    for (i, call) in calls.iter().enumerate() {
        match tools.iter().find(|t| t.name() == call.name) {
            Some(tool) if tool.concurrent() => concurrent_idx.push(i),
            _ => sequential_idx.push(i),
        }
    }

    let mut results: Vec<Option<ToolResult>> = (0..calls.len()).map(|_| None).collect();

    let concurrent_futures = concurrent_idx
        .iter()
        .map(|&i| execute_one(tools, &calls[i]));
    for (i, result) in concurrent_idx
        .iter()
        .zip(join_all(concurrent_futures).await)
    {
        results[*i] = Some(result);
    }

    for i in sequential_idx {
        results[i] = Some(execute_one(tools, &calls[i]).await);
    }

    results
        .into_iter()
        .map(|r| {
            r.expect("every call index is populated by either the concurrent or sequential pass")
        })
        .collect()
}

/// Resolves and runs a single tool call, turning any failure (tool not found, more
/// than one tool with that name, or the tool's own error) into a [`ToolOutcome`]
/// that gets reported back to the model, rather than propagated as a hard error.
async fn execute_one(tools: &[Arc<dyn Tool>], call: &ToolCall) -> ToolResult {
    let matches: Vec<&Arc<dyn Tool>> = tools.iter().filter(|t| t.name() == call.name).collect();

    let outcome = match matches.as_slice() {
        [] => ToolOutcome::Error(
            ToolError::NotFound {
                name: call.name.clone(),
            }
            .to_string(),
        ),
        [tool] => match tool.call(call.arguments.clone()).await {
            Ok(output) => ToolOutcome::Output(output),
            Err(err) => ToolOutcome::Error(err.to_string()),
        },
        _ => ToolOutcome::Error(
            ToolError::MultipleFound {
                name: call.name.clone(),
            }
            .to_string(),
        ),
    };

    ToolResult {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        args: call.arguments.clone(),
        result: outcome,
    }
}
