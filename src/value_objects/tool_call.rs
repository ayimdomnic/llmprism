use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A request from the model to call one specific tool with specific arguments.
///
/// You'll typically only read these; the tool loop
/// ([`crate::tool_loop::run_text`]) matches `name` against your registered
/// [`Tool`](crate::Tool)s and runs `arguments` through it automatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// An identifier for this specific call, assigned by the provider. Used to
    /// match this call up with its eventual [`ToolResult`].
    pub id: String,
    /// The name of the tool the model wants to call.
    pub name: String,
    /// The arguments the model filled in, as raw JSON matching the tool's
    /// [`parameters`](crate::Tool::parameters) schema.
    pub arguments: Value,
}

/// The outcome of running one [`ToolCall`], sent back to the model so it can
/// continue the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Matches the [`ToolCall::id`] this is the result of.
    pub tool_call_id: String,
    /// The name of the tool that was called.
    pub tool_name: String,
    /// The arguments the tool was called with (repeated here for convenience,
    /// e.g. when logging or displaying tool activity).
    pub args: Value,
    /// What actually happened -- either the tool's output, or a description of
    /// why it failed. See [`ToolOutcome`].
    pub result: ToolOutcome,
}

/// Whether a tool call succeeded or failed.
///
/// Both cases are sent back to the model the same way (as text it can read) --
/// a failed tool call doesn't abort the request, it just tells the model what
/// went wrong so it can adapt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutcome {
    /// The tool ran successfully.
    Output(ToolOutput),
    /// The tool failed; this holds a plain-language description of what went
    /// wrong.
    Error(String),
}

/// What a tool returns after running successfully. Build one from a plain
/// string with `.into()` for the common case, or construct it directly if you
/// need to attach [`artifacts`](Self::artifacts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// The text sent back to the model as the tool's result.
    pub content: String,
    /// Extra structured data produced alongside `content` -- for example, an
    /// image or file a tool generated -- that your application can use even
    /// though the model itself only sees `content`. Empty for most tools.
    #[serde(default)]
    pub artifacts: Vec<Value>,
}

impl From<String> for ToolOutput {
    fn from(content: String) -> Self {
        Self {
            content,
            artifacts: Vec::new(),
        }
    }
}

impl From<&str> for ToolOutput {
    fn from(content: &str) -> Self {
        content.to_string().into()
    }
}
