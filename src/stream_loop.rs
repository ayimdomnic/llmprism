//! The multi-step tool-calling loop for *streaming* Text generation -- the
//! streaming counterpart to [`crate::tool_loop`].

use std::sync::Arc;

use async_stream::try_stream;
use futures::stream::{BoxStream, StreamExt};

use crate::error::Error;
use crate::provider::Provider;
use crate::stream_event::StreamEvent;
use crate::text::request::{TextRequest, ToolChoice};
use crate::text::response::{Step, TextResponse};
use crate::tool_loop::execute_tools;
use crate::value_objects::{
    AssistantMessage, FinishReason, Message, Meta, ToolCall, ToolResultMessage, Usage,
};

/// Drives a text request to completion the same way
/// [`crate::tool_loop::run_text`] does, but yields [`StreamEvent`]s as they
/// arrive instead of waiting for the whole thing to finish.
///
/// This lives once, centrally, for the same reason the non-streaming loop does:
/// each provider only implements
/// [`Provider::stream_text_once`] -- a single
/// round trip -- and this function is what turns that into a full, potentially
/// multi-step conversation. The whole loop runs inside one
/// [`async_stream::try_stream!`] generator rather than a function that calls
/// itself for each round trip: that keeps it a single, boundedly-sized state
/// machine instead of an ever-growing chain of nested futures.
///
/// This is what
/// [`PendingTextRequest::stream`](crate::text::PendingTextRequest::stream) calls
/// internally; you normally reach this indirectly through that builder rather
/// than calling it yourself.
///
/// Returns a [`BoxStream`] (a boxed, already-pinned stream) rather than `impl
/// Stream` -- `try_stream!`'s generated type isn't [`Unpin`], which would force
/// every caller to pin it themselves (with `futures::pin_mut!` or similar)
/// before calling `.next()`. Boxing it here, once, means callers just get a
/// stream that works directly.
pub fn stream_text(
    provider: Arc<dyn Provider>,
    mut request: TextRequest,
) -> BoxStream<'static, Result<StreamEvent, Error>> {
    let stream = try_stream! {
        let mut steps: Vec<Step> = Vec::new();

        loop {
            let mut inner = provider.stream_text_once(request.clone()).await?;

            let mut text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut usage = Usage::default();
            let mut finish_reason = FinishReason::Stop;
            let mut meta = Meta::default();

            while let Some(event) = inner.next().await {
                let event = event?;

                match &event {
                    StreamEvent::StreamStart { meta: step_meta } => meta = step_meta.clone(),
                    StreamEvent::TextDelta { text: delta } => text.push_str(delta),
                    StreamEvent::ToolCall(call) => tool_calls.push(call.clone()),
                    StreamEvent::StepFinish {
                        usage: step_usage,
                        finish_reason: step_finish_reason,
                    } => {
                        usage = *step_usage;
                        finish_reason = *step_finish_reason;
                    }
                    // ToolCallDelta / ToolResult carry no state this loop needs to
                    // track -- they're forwarded to the caller as-is, below.
                    StreamEvent::ToolCallDelta { .. } | StreamEvent::ToolResult(_) => {}
                    // A provider's single round trip never emits its own
                    // StreamEnd -- only this loop does, once the whole
                    // conversation (every round trip) is done.
                    StreamEvent::StreamEnd { .. } => {}
                }

                yield event;
            }

            let has_tool_calls = finish_reason == FinishReason::ToolCalls && !tool_calls.is_empty();

            steps.push(Step {
                text: if text.is_empty() { None } else { Some(text) },
                tool_calls: tool_calls.clone(),
                finish_reason,
                usage,
                meta,
            });

            if !has_tool_calls || steps.len() as u32 >= request.max_steps {
                yield StreamEvent::StreamEnd {
                    response: TextResponse::from_steps(steps),
                };
                break;
            }

            let results = execute_tools(&request.tools, &tool_calls).await;
            for result in &results {
                yield StreamEvent::ToolResult(result.clone());
            }

            let last_text = steps.last().and_then(|step| step.text.clone());
            request.messages.push(Message::Assistant(AssistantMessage {
                content: last_text,
                tool_calls,
            }));
            request.messages.push(Message::ToolResult(ToolResultMessage {
                tool_results: results,
            }));
            // See the matching comment in `tool_loop::run_text` -- a forced tool
            // choice only applies for one round, or the model could be made to
            // call a tool forever with no way to ever answer in plain text.
            request.tool_choice = ToolChoice::Auto;
        }
    };

    stream.boxed()
}
