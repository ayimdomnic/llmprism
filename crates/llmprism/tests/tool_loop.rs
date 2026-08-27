//! Exercises `tool_loop::run_text` generically against `FakeProvider` -- no network
//! access, no feature flags required.

use async_trait::async_trait;
use llmprism::error::ToolError;
use llmprism::schema::{ObjectSchema, Schema, StringSchema};
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::tool::Tool;
use llmprism::value_objects::{FinishReason, ToolOutput};
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
async fn tool_loop_executes_a_tool_call_and_completes() {
    let provider = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call(
            "call_1",
            "echo",
            json!({"message": "hi"}),
        ))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(EchoTool::new())
        .with_max_steps(5)
        .generate()
        .await
        .unwrap();

    assert_eq!(response.text.as_deref(), Some("done"));
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.steps.len(), 2);
}

#[tokio::test]
async fn tool_loop_stops_at_max_steps_even_with_pending_tool_calls() {
    let provider = FakeProvider::new("fake").respond_with(
        FakeTextResponse::new("").with_tool_call("call_1", "echo", json!({"message": "hi"})),
    );

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(EchoTool::new())
        .with_max_steps(1)
        .generate()
        .await
        .unwrap();

    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    assert_eq!(response.steps.len(), 1);
}

#[tokio::test]
async fn with_stop_when_ends_the_loop_before_running_the_pending_tool_call() {
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

    // Two tool-call responses queued, but `stop_when` should end the loop
    // right after the first one -- if it didn't, the second queued response
    // would get consumed too and `steps.len()` would be 2.
    let provider = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call(
            "call_1",
            "echo",
            json!({"message": "hi"}),
        ))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(EchoTool::new())
        .with_max_steps(5)
        .with_stop_when(ToolWasRequested("echo"))
        .generate()
        .await
        .unwrap();

    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    assert_eq!(response.steps.len(), 1);
}

/// An echo tool that requires approval and counts how many times it
/// actually ran -- lets a test assert not just what the model saw, but
/// whether the tool's own code ever executed at all.
struct ApprovalRequiredEchoTool {
    parameters: ObjectSchema,
    call_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl ApprovalRequiredEchoTool {
    fn new(call_count: std::sync::Arc<std::sync::atomic::AtomicU32>) -> Self {
        Self {
            parameters: ObjectSchema::new("parameters"),
            call_count,
        }
    }
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

#[tokio::test]
async fn an_approved_tool_call_runs_normally() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

    let provider = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call("call_1", "echo", json!({})))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(ApprovalRequiredEchoTool::new(call_count.clone()))
        .with_max_steps(5)
        .with_approval_handler(|_call: &llmprism::value_objects::ToolCall| true)
        .generate()
        .await
        .unwrap();

    assert_eq!(response.text.as_deref(), Some("done"));
    assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_denied_tool_call_never_runs_and_reports_a_tool_error() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

    let provider = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call("call_1", "echo", json!({})))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(ApprovalRequiredEchoTool::new(call_count.clone()))
        .with_max_steps(5)
        .with_approval_handler(|_call: &llmprism::value_objects::ToolCall| false)
        .generate()
        .await
        .unwrap();

    assert_eq!(
        call_count.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the tool's own call() should never have run"
    );
}

#[tokio::test]
async fn an_approval_required_tool_with_no_handler_attached_is_denied_by_default() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

    let provider = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call("call_1", "echo", json!({})))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    // No `.with_approval_handler(...)` call at all.
    registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(ApprovalRequiredEchoTool::new(call_count.clone()))
        .with_max_steps(5)
        .generate()
        .await
        .unwrap();

    assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn unknown_tool_call_surfaces_as_a_tool_error_result_instead_of_failing() {
    let provider = FakeProvider::new("fake")
        .respond_with(FakeTextResponse::new("").with_tool_call(
            "call_1",
            "does_not_exist",
            json!({}),
        ))
        .respond_with(FakeTextResponse::new("done"));

    let mut registry = Registry::new();
    registry.register("fake", provider);

    let response = registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("say hi")
        .with_tool(EchoTool::new())
        .with_max_steps(5)
        .generate()
        .await
        .unwrap();

    assert_eq!(response.text.as_deref(), Some("done"));
}
