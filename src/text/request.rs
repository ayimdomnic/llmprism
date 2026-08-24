//! The Text generation request: [`TextRequest`] (the data) and
//! [`PendingTextRequest`] (the fluent builder you actually use).

use std::sync::Arc;

use futures::stream::BoxStream;

use crate::error::Error;
use crate::provider::Provider;
use crate::stream_event::StreamEvent;
use crate::tool::Tool;
use crate::value_objects::{Message, UserMessage};

use super::response::TextResponse;

/// Controls how much freedom the model has to decide whether (and which) tool to
/// call.
///
/// This has no effect unless the request also has tools attached via
/// [`PendingTextRequest::with_tool`] / [`with_tools`](PendingTextRequest::with_tools).
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    /// Let the model decide freely whether to call a tool, and which one. This is
    /// almost always what you want, and is the default.
    #[default]
    Auto,
    /// Don't allow the model to call any tool on this turn, even though tools are
    /// attached to the request.
    None,
    /// Require the model to call *some* tool, but let it pick which one.
    Any,
    /// Require the model to call this specific tool (by name) on this turn.
    Tool(String),
}

/// The frozen, ready-to-send description of one Text generation call.
///
/// You'll rarely construct this directly -- build one with
/// [`PendingTextRequest`] instead, which offers a nicer, chainable API and
/// produces a `TextRequest` for you. This type mainly exists so providers have a
/// concrete, provider-agnostic value to translate into their own wire format.
///
/// One detail worth knowing if you're implementing a [`Provider`]: while a
/// multi-step tool-calling conversation is in progress,
/// [`crate::tool_loop::run_text`] appends new messages to `messages` and sends the
/// request again for each round trip -- so a `Provider` always sees the full
/// conversation so far, not just the newest message.
#[derive(Clone)]
pub struct TextRequest {
    /// The model identifier to use, in whatever format the target provider
    /// expects (e.g. `"gpt-4o-mini"`, `"claude-3-5-haiku-20241022"`).
    pub model: String,
    /// System-level instructions for the model, set once up front (see
    /// [`PendingTextRequest::with_system_prompt`]).
    pub system_prompts: Vec<String>,
    /// The conversation so far, oldest first.
    pub messages: Vec<Message>,
    /// The maximum number of tokens the model may generate in its reply. `None`
    /// leaves this up to the provider's own default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature -- higher values make output more random, lower
    /// values make it more focused and deterministic. `None` leaves this up to
    /// the provider's own default.
    pub temperature: Option<f32>,
    /// Nucleus-sampling cutoff. `None` leaves this up to the provider's own
    /// default.
    pub top_p: Option<f32>,
    /// Strings that stop generation the moment the model produces one of
    /// them (the stop string itself isn't included in the reply). Supported
    /// by OpenAI (`stop`), Anthropic (`stop_sequences`), and Gemini
    /// (`stopSequences`); ignored elsewhere. `None`/empty leaves this up to
    /// the provider's own default (usually "no extra stop strings").
    pub stop_sequences: Vec<String>,
    /// A seed for providers whose backend can honor one (OpenAI's `seed`;
    /// most other providers have no equivalent and ignore this). Requesting
    /// a seed makes output *more* reproducible across identical requests,
    /// not guaranteed-deterministic -- treat it as a best-effort hint, the
    /// same way the provider itself does. `None` leaves this up to the
    /// provider's own default (unseeded).
    pub seed: Option<u64>,
    /// The maximum number of request/response round trips the tool-calling loop
    /// will run before stopping, even if the model keeps asking for more tool
    /// calls. Defaults to `1` (a single round trip, i.e. no tool-calling loop) via
    /// [`TextRequest::new`] -- raise it with
    /// [`PendingTextRequest::with_max_steps`] if you're attaching tools.
    pub max_steps: u32,
    /// Tools the model is allowed to call during this request. See [`Tool`].
    pub tools: Vec<Arc<dyn Tool>>,
    /// Provider-native tools -- built-in, server-side capabilities like web
    /// search or code execution that the provider itself runs, as opposed to
    /// a [`Tool`] this crate runs on your behalf. Unlike [`Tool`], there's no
    /// common shape to give these a typed Rust API: each provider defines
    /// its own set with its own JSON shape (Anthropic's
    /// `{"type": "web_search_20250305", "name": "web_search", ...}`,
    /// Gemini's `{"googleSearch": {}}`, and so on), so each entry here is
    /// sent through mostly as-is. Supported on Anthropic and Gemini; ignored
    /// on OpenAI, whose Chat Completions API (the one this crate talks to --
    /// see the crate root docs) has no equivalent concept, only its newer
    /// Responses API. See
    /// [`with_provider_tool`](PendingTextRequest::with_provider_tool).
    pub provider_tools: Vec<serde_json::Value>,
    /// How much freedom the model has to decide whether to call a tool. See
    /// [`ToolChoice`].
    pub tool_choice: ToolChoice,
    /// Marks the system prompt as a cache breakpoint on providers that
    /// support explicit prompt caching (currently just Anthropic; ignored
    /// elsewhere). Worth setting whenever `system_prompts` is long and
    /// reused across many requests (a large, mostly-static persona or style
    /// guide, for example) -- a cached prompt costs more on the request that
    /// writes the cache but substantially less on every later request that
    /// hits it, within the provider's cache lifetime. Defaults to `false`;
    /// see [`with_prompt_caching`](PendingTextRequest::with_prompt_caching).
    pub cache_system_prompt: bool,
    /// Enables Claude's extended thinking, capping how many tokens it may
    /// spend on internal reasoning before writing its visible reply
    /// (Anthropic only; ignored elsewhere -- and, on Anthropic, only for
    /// this method: extended thinking is incompatible with the forced tool
    /// call [`Registry::structured`](crate::Registry::structured) uses
    /// under the hood, so this has no equivalent on
    /// [`StructuredRequest`](crate::structured::StructuredRequest)).
    /// `None` leaves thinking off, matching every other provider's default
    /// behavior. See
    /// [`with_thinking_budget`](PendingTextRequest::with_thinking_budget).
    pub thinking_budget: Option<u32>,
    /// Sets how much effort a reasoning model spends thinking before
    /// replying (OpenAI's o-series/GPT-5-class models; ignored elsewhere,
    /// and most non-reasoning models reject it outright rather than
    /// ignoring it). Valid values are model-dependent (`"minimal"`,
    /// `"low"`, `"medium"`, `"high"`, and newer values like `"xhigh"` on
    /// some models) -- kept as a plain string for the same reason
    /// `size`/`quality` on an images request are. `None` leaves this up to
    /// the model's own default.
    pub reasoning_effort: Option<String>,
    /// Extra provider-specific fields to send alongside this request, for
    /// options this crate doesn't model as a typed field yet. Must be a JSON
    /// object to have any effect: each of its top-level keys is merged into
    /// (and, if it collides with one of this crate's own fields, overrides)
    /// the request body actually sent to the provider. The default,
    /// `Value::Null`, sends nothing extra.
    pub provider_options: serde_json::Value,
}

impl TextRequest {
    /// Creates a bare request targeting `model`, with no messages, no tools, and
    /// every optional setting left at the provider's default.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system_prompts: Vec::new(),
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: Vec::new(),
            seed: None,
            max_steps: 1,
            tools: Vec::new(),
            provider_tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            cache_system_prompt: false,
            thinking_budget: None,
            reasoning_effort: None,
            provider_options: serde_json::Value::Null,
        }
    }
}

/// The fluent, chainable way to build and run a Text generation request.
///
/// Get one of these from [`Registry::text`](crate::Registry::text), chain
/// `.with_*()` calls to describe what you want, then call [`generate`](Self::generate)
/// to actually send the request (running the full multi-step tool-calling loop if
/// you attached tools).
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "openai")]
/// # async fn example() -> Result<(), llmprism::Error> {
/// use llmprism::Registry;
///
/// let registry = Registry::from_env();
///
/// let response = registry
///     .text("openai", "gpt-4o-mini")?
///     .with_system_prompt("You are a concise assistant.")
///     .with_prompt("What's the capital of France?")
///     .with_temperature(0.2)
///     .with_max_tokens(100)
///     .generate()
///     .await?;
///
/// println!("{}", response.text.unwrap_or_default());
/// # Ok(())
/// # }
/// ```
pub struct PendingTextRequest {
    provider: Arc<dyn Provider>,
    request: TextRequest,
}

impl PendingTextRequest {
    /// Starts a new builder for `provider`, targeting `model`. You'll normally get
    /// one of these from [`Registry::text`](crate::Registry::text) rather than
    /// calling this directly.
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            request: TextRequest::new(model),
        }
    }

    /// Adds a system-level instruction -- guidance for how the model should
    /// behave that isn't part of the visible conversation (a persona, a style
    /// guide, ground rules). Call this more than once to add several; they're
    /// combined in the order you added them.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.request.system_prompts.push(prompt.into());
        self
    }

    /// Adds a plain-text user message to the conversation. This is the shortcut
    /// for the common case of a single-turn question; for a multi-turn
    /// conversation, build it up with repeated calls to this method and/or
    /// [`with_message`](Self::with_message).
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.request
            .messages
            .push(Message::User(UserMessage::text(prompt)));
        self
    }

    /// Appends one message (of any role) to the conversation. Use this when you
    /// need more control than [`with_prompt`](Self::with_prompt) gives you -- for
    /// example, replaying a stored conversation history.
    pub fn with_message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    /// Appends several messages to the conversation at once, in order.
    pub fn with_messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.request.messages.extend(messages);
        self
    }

    /// Caps how many tokens the model may generate in its reply.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.request.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the sampling temperature. Typical values range from `0.0`
    /// (deterministic, focused) to around `1.0`+ (more varied, creative) --
    /// check your chosen provider's docs for its exact accepted range.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.request.temperature = Some(temperature);
        self
    }

    /// Sets the nucleus-sampling cutoff (`top_p`). Most applications only need to
    /// tune one of `temperature` or `top_p`, not both.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.request.top_p = Some(top_p);
        self
    }

    /// Adds a string that stops generation the moment the model produces it.
    /// Call this more than once to add several. See
    /// [`TextRequest::stop_sequences`] for which providers support this.
    pub fn with_stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.request.stop_sequences.push(stop.into());
        self
    }

    /// Sets a seed for providers whose backend can honor one. See
    /// [`TextRequest::seed`] for what "honor" means here.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.request.seed = Some(seed);
        self
    }

    /// Sets how many request/response round trips the tool-calling loop is
    /// allowed to run before it stops on its own. If you're attaching tools with
    /// [`with_tool`](Self::with_tool), you'll usually want this higher than the
    /// default of `1` (which only allows a single round trip, with no room for a
    /// tool call and a follow-up reply).
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.request.max_steps = max_steps;
        self
    }

    /// Attaches one tool the model may call during this request. Call this once
    /// per tool. See [`Tool`] for how to define one, and remember to raise
    /// [`with_max_steps`](Self::with_max_steps) above `1` so there's room for the
    /// model to use the tool's result in a follow-up reply.
    pub fn with_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.request.tools.push(Arc::new(tool));
        self
    }

    /// Attaches several already-constructed tools at once -- handy when you keep
    /// a shared `Vec<Arc<dyn Tool>>` of tools available to your application and
    /// want to reuse it across requests.
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.request.tools.extend(tools);
        self
    }

    /// Attaches one provider-native tool -- a built-in, server-side
    /// capability like web search, in whatever JSON shape the target
    /// provider expects for it. See [`TextRequest::provider_tools`] for
    /// which providers support this at all.
    ///
    /// ```
    /// # #[cfg(feature = "anthropic")]
    /// # {
    /// use llmprism::providers::anthropic::AnthropicProvider;
    /// use llmprism::Registry;
    /// use serde_json::json;
    ///
    /// let mut registry = Registry::new();
    /// registry.register("anthropic", AnthropicProvider::new("sk-ant-..."));
    ///
    /// let request = registry
    ///     .text("anthropic", "claude-sonnet-4-5")
    ///     .unwrap()
    ///     .with_prompt("What's the latest Rust release?")
    ///     .with_provider_tool(json!({
    ///         "type": "web_search_20250305",
    ///         "name": "web_search",
    ///     }))
    ///     .to_request();
    /// # assert_eq!(request.provider_tools.len(), 1);
    /// # }
    /// ```
    pub fn with_provider_tool(mut self, tool: serde_json::Value) -> Self {
        self.request.provider_tools.push(tool);
        self
    }

    /// Controls how free the model is to decide whether to call a tool. See
    /// [`ToolChoice`]. Has no effect unless tools are attached.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.request.tool_choice = choice;
        self
    }

    /// Marks the system prompt as a cache breakpoint on providers that
    /// support explicit prompt caching (currently just Anthropic; a no-op
    /// elsewhere). See [`TextRequest::cache_system_prompt`] for when this is
    /// worth turning on.
    pub fn with_prompt_caching(mut self) -> Self {
        self.request.cache_system_prompt = true;
        self
    }

    /// Enables Claude's extended thinking, capping its internal reasoning at
    /// `budget_tokens`. Anthropic requires `max_tokens` to be strictly
    /// greater than `budget_tokens`; if
    /// [`with_max_tokens`](Self::with_max_tokens) wasn't called (or was
    /// called with too small a value), this crate raises the value actually
    /// sent so the request doesn't fail for a reason that isn't obvious from
    /// the error alone. See [`TextRequest::thinking_budget`] for the
    /// `structured`-output caveat.
    pub fn with_thinking_budget(mut self, budget_tokens: u32) -> Self {
        self.request.thinking_budget = Some(budget_tokens);
        self
    }

    /// Sets how much effort a reasoning model spends thinking before
    /// replying. See [`TextRequest::reasoning_effort`] for which providers
    /// and models this applies to.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.request.reasoning_effort = Some(effort.into());
        self
    }

    /// Freezes the builder's current state into a [`TextRequest`] without sending
    /// it. Mainly useful for inspecting or logging what would be sent, or for
    /// advanced cases where you want to drive [`crate::tool_loop::run_text`]
    /// yourself.
    pub fn to_request(&self) -> TextRequest {
        self.request.clone()
    }

    /// Sends the request and returns the model's final answer.
    ///
    /// If tools are attached and the model asks to call one, this handles the
    /// entire back-and-forth automatically -- running the tool, sending its
    /// result back, and continuing the conversation -- until the model produces a
    /// final answer or [`with_max_steps`](Self::with_max_steps) is reached. See
    /// [`crate::tool_loop`] for the details.
    pub async fn generate(self) -> Result<TextResponse, Error> {
        crate::tool_loop::run_text(self.provider, self.request).await
    }

    /// Sends the request and returns a stream of [`StreamEvent`]s, instead of
    /// waiting for the model's entire reply the way [`generate`](Self::generate)
    /// does. Useful for showing a reply as it's generated rather than all at
    /// once.
    ///
    /// Tool calling works the same way it does for `generate`: if the model asks
    /// to call a tool, this handles running it, reports the result as a
    /// [`StreamEvent::ToolResult`], and continues streaming the model's next
    /// round trip automatically. The stream ends with exactly one
    /// [`StreamEvent::StreamEnd`], carrying the same final result `generate`
    /// would have returned.
    pub fn stream(self) -> BoxStream<'static, Result<StreamEvent, Error>> {
        crate::stream_loop::stream_text(self.provider, self.request)
    }
}
