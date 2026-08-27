//! Exercises `stream_loop::stream_text` (via `PendingTextRequest::stream`)
//! generically against `FakeProvider` -- no network access, no feature flags
//! required.

use async_trait::async_trait;
use futures::StreamExt;
use llmprism::error::ToolError;
use llmprism::schema::{ObjectSchema, Schema, StringSchema};
use llmprism::stream_event::StreamEvent;
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::tool::Tool;
use llmprism::value_objects::{FinishReason, ToolOutcome, ToolOutput};
use llmprism::Registry;
use serde_json::{json, Value};

struct EchoTool {
    parameters: ObjectSchema,
}

impl EchoTool {
    fn new() -> Self {
        Self {
            parameters: ObjectSchema::new("parameters").with_property(
                Schema::String(
                    StringSchema::new("message").with_description("The message to echo back"),
                ),
                true,
            ),
        }
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes the given message back"
    }

    fn parameters(&self) -> &ObjectSchema {
        &self.parameters
    }

    async fn call(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        Ok(ToolOutput::from(format!("echo: {message}")))
    }
}

#[tokio::test]
async fn streaming_reassembles_text_deltas_and_ends_with_the_final_response() {
    let provider = FakeProvider::new("fake").respond_with(FakeTextResponse::new("hello there"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let events: Vec<StreamEvent> = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("hi")
        .stream()
        .map(|event| event.unwrap())
        .collect()
        .await;

    let deltas: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TextDelta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, "hello there");

    match events.last() {
        Some(StreamEvent::StreamEnd { response }) => {
            assert_eq!(response.text.as_deref(), Some("hello there"));
            assert_eq!(response.finish_reason, FinishReason::Stop);
        }
        other => panic!("expected the stream to end with StreamEnd, got {other:?}"),
    }
}

#[tokio::test]
async fn streaming_runs_a_tool_call_and_continues_into_a_second_round_trip() {
    let provider = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call(
            "call_1",
            "echo",
            json!({"message": "hi"}),
        ))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let events: Vec<StreamEvent> = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(EchoTool::new())
        .with_max_steps(5)
        .stream()
        .map(|event| event.unwrap())
        .collect()
        .await;

    // Two StreamStart events -- one per round trip.
    let stream_starts = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::StreamStart { .. }))
        .count();
    assert_eq!(stream_starts, 2);

    let tool_result = events
        .iter()
        .find_map(|e| match e {
            StreamEvent::ToolResult(result) => Some(result),
            _ => None,
        })
        .expect("expected a ToolResult event before the second round trip");
    assert!(
        matches!(&tool_result.result, ToolOutcome::Output(output) if output.content == "echo: hi")
    );

    match events.last() {
        Some(StreamEvent::StreamEnd { response }) => {
            assert_eq!(response.text.as_deref(), Some("done"));
            assert_eq!(response.steps.len(), 2);
        }
        other => panic!("expected the stream to end with StreamEnd, got {other:?}"),
    }
}

#[tokio::test]
async fn a_denied_tool_call_never_runs_during_streaming() {
    struct ApprovalRequiredEchoTool {
        parameters: ObjectSchema,
        call_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
    }

    #[async_trait]
    impl Tool for ApprovalRequiredEchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes the given message back, but only once approved"
        }

        fn parameters(&self) -> &ObjectSchema {
            &self.parameters
        }

        fn needs_approval(&self) -> bool {
            true
        }

        async fn call(&self, _args: Value) -> Result<ToolOutput, ToolError> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ToolOutput::from("echoed"))
        }
    }

    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

    let provider = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call("call_1", "echo", json!({})))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let _events: Vec<StreamEvent> = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(ApprovalRequiredEchoTool {
            parameters: ObjectSchema::new("parameters"),
            call_count: call_count.clone(),
        })
        .with_max_steps(5)
        .with_approval_handler(|_call: &llmprism::value_objects::ToolCall| false)
        .stream()
        .map(|event| event.unwrap())
        .collect()
        .await;

    assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn with_stop_when_ends_the_stream_before_running_the_pending_tool_call() {
    use llmprism::text::Step;
    use llmprism::tool_loop::StopCondition;

    struct ToolWasRequested(&'static str);

    impl StopCondition for ToolWasRequested {
        fn should_stop(&self, steps: &[Step]) -> bool {
            steps
                .last()
                .is_some_and(|step| step.tool_calls.iter().any(|call| call.name == self.0))
        }
    }

    // Same shape as `streaming_runs_a_tool_call_and_continues_into_a_second_round_trip`,
    // but `stop_when` should end the stream after the first round trip
    // instead of continuing into the second queued response.
    let provider = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call(
            "call_1",
            "echo",
            json!({"message": "hi"}),
        ))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let events: Vec<StreamEvent> = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(EchoTool::new())
        .with_max_steps(5)
        .with_stop_when(ToolWasRequested("echo"))
        .stream()
        .map(|event| event.unwrap())
        .collect()
        .await;

    let stream_starts = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::StreamStart { .. }))
        .count();
    assert_eq!(stream_starts, 1);

    match events.last() {
        Some(StreamEvent::StreamEnd { response }) => {
            assert_eq!(response.finish_reason, FinishReason::ToolCalls);
            assert_eq!(response.steps.len(), 1);
        }
        other => panic!("expected the stream to end with StreamEnd, got {other:?}"),
    }
}
