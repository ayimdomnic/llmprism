//! The [`Tool`] trait -- how you give a model something it can call.

use async_trait::async_trait;
use serde_json::Value;

use crate::error::ToolError;
use crate::schema::ObjectSchema;
use crate::value_objects::ToolOutput;

/// Something a model can ask to have run on its behalf -- a weather lookup, a
/// database query, a calculator, anything your application can do that you want to
/// expose to the model.
///
/// Attach one or more tools to a text request (see
/// [`PendingTextRequest::with_tool`](crate::text::PendingTextRequest::with_tool)),
/// and if the model decides to call one, `llmprism` runs it for you, sends the
/// result back, and continues the conversation automatically -- you don't need to
/// write any of that back-and-forth yourself. See [`crate::tool_loop`] if you want
/// to understand exactly how that works.
///
/// Arguments come in and results go out as [`serde_json::Value`] rather than a
/// typed struct. That's not a limitation carried over from somewhere else -- a
/// tool call always arrives from the model as a JSON blob picked out by name at
/// runtime, so JSON in, JSON out is simply the natural shape here. If you'd rather
/// work with a typed Rust struct inside `call`, just deserialize `args` yourself
/// with `serde_json::from_value` as the first thing you do.
///
/// # Example
///
/// ```
/// use async_trait::async_trait;
/// use llmprism::error::ToolError;
/// use llmprism::schema::{ObjectSchema, Schema, StringSchema};
/// use llmprism::tool::Tool;
/// use llmprism::value_objects::ToolOutput;
/// use serde_json::Value;
///
/// struct Weather {
///     parameters: ObjectSchema,
/// }
///
/// impl Weather {
///     fn new() -> Self {
///         Self {
///             parameters: ObjectSchema::new("parameters").with_property(
///                 Schema::String(StringSchema::new("city").with_description("City name")),
///                 true, // required
///             ),
///         }
///     }
/// }
///
/// #[async_trait]
/// impl Tool for Weather {
///     fn name(&self) -> &str { "get_weather" }
///     fn description(&self) -> &str { "Looks up the current weather for a city" }
///     fn parameters(&self) -> &ObjectSchema { &self.parameters }
///
///     async fn call(&self, args: Value) -> Result<ToolOutput, ToolError> {
///         let city = args.get("city").and_then(|v| v.as_str()).unwrap_or("unknown");
///         Ok(ToolOutput::from(format!("It's sunny in {city}.")))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// The name the model uses to call this tool. Must be unique among the tools
    /// attached to a single request.
    fn name(&self) -> &str;

    /// A plain-language description of what this tool does and when to use it.
    /// The model reads this to decide whether calling the tool is appropriate --
    /// the clearer this is, the better the model's decisions will be.
    fn description(&self) -> &str;

    /// Describes the shape of the arguments this tool expects, so the model knows
    /// what to send and providers can validate/generate matching calls. See
    /// [`crate::schema`] for how to build one.
    fn parameters(&self) -> &ObjectSchema;

    /// Whether this tool is safe to run at the same time as other concurrent
    /// tools when the model requests several tool calls in a single turn.
    /// Defaults to `false` (run one at a time, in order) -- override this and
    /// return `true` for tools with no side effects on each other, like read-only
    /// lookups, to let them run in parallel.
    fn concurrent(&self) -> bool {
        false
    }

    /// Runs the tool with the arguments the model provided and returns the result
    /// to send back to the model.
    ///
    /// If this returns `Err`, the tool loop doesn't abort the request -- the error
    /// message is sent back to the model as the tool's result, so the model can
    /// see what went wrong and decide how to proceed (try different arguments,
    /// give up on that approach, tell the user, etc.).
    async fn call(&self, args: Value) -> Result<ToolOutput, ToolError>;
}
