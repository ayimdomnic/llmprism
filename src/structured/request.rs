use std::sync::Arc;

use crate::error::Error;
use crate::provider::Provider;
use crate::schema::ObjectSchema;
use crate::value_objects::{Message, UserMessage};

use super::response::StructuredResponse;

/// The immutable, provider-agnostic shape of one structured-output call. You'll
/// normally build one with [`PendingStructuredRequest`] rather than
/// constructing this directly.
#[derive(Clone)]
pub struct StructuredRequest {
    pub model: String,
    pub system_prompts: Vec<String>,
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// The shape the model's reply must match. Required up front (via
    /// [`PendingStructuredRequest::new`]) rather than optional the way tools are
    /// for a text request, since a structured request without a schema isn't a
    /// meaningful thing to send.
    pub schema: ObjectSchema,
    /// Extra provider-specific fields to send alongside this request, for
    /// options this crate doesn't model as a typed field yet. Must be a JSON
    /// object to have any effect: each of its top-level keys is merged into
    /// (and, if it collides with one of this crate's own fields, overrides)
    /// the request body actually sent to the provider. The default,
    /// `Value::Null`, sends nothing extra.
    pub provider_options: serde_json::Value,
}

impl StructuredRequest {
    pub fn new(model: impl Into<String>, schema: ObjectSchema) -> Self {
        Self {
            model: model.into(),
            system_prompts: Vec::new(),
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            schema,
            provider_options: serde_json::Value::Null,
        }
    }
}

/// The fluent, chainable way to build and run a structured-output request.
///
/// Get one of these from
/// [`Registry::structured`](crate::Registry::structured), chain `.with_*()`
/// calls to describe the conversation, then call [`generate`](Self::generate).
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "openai")]
/// # async fn example() -> Result<(), llmprism::Error> {
/// use llmprism::schema::{NumberSchema, ObjectSchema, Schema, StringSchema};
/// use llmprism::Registry;
///
/// let schema = ObjectSchema::new("recipe")
///     .with_property(Schema::String(StringSchema::new("title")), true)
///     .with_property(Schema::Number(NumberSchema::new("minutes")), true);
///
/// let registry = Registry::from_env();
/// let response = registry
///     .structured("openai", "gpt-4o-mini", schema)?
///     .with_prompt("A quick pasta recipe.")
///     .generate()
///     .await?;
///
/// println!("{}", response.data);
/// # Ok(())
/// # }
/// ```
pub struct PendingStructuredRequest {
    provider: Arc<dyn Provider>,
    request: StructuredRequest,
}

impl PendingStructuredRequest {
    /// Starts a new builder for `provider`, targeting `model` and requiring the
    /// reply to match `schema`. You'll normally get one of these from
    /// [`Registry::structured`](crate::Registry::structured) rather than calling
    /// this directly.
    pub fn new(
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
        schema: ObjectSchema,
    ) -> Self {
        Self {
            provider,
            request: StructuredRequest::new(model, schema),
        }
    }

    /// Adds a system-level instruction. See
    /// [`PendingTextRequest::with_system_prompt`](crate::text::PendingTextRequest::with_system_prompt)
    /// for the details -- this behaves identically.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.request.system_prompts.push(prompt.into());
        self
    }

    /// Adds a plain-text user message to the conversation.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.request
            .messages
            .push(Message::User(UserMessage::text(prompt)));
        self
    }

    /// Appends one message (of any role) to the conversation.
    pub fn with_message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    /// Appends several messages to the conversation at once, in order.
    pub fn with_messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.request.messages.extend(messages);
        self
    }

    /// Caps how many tokens the model may generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.request.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.request.temperature = Some(temperature);
        self
    }

    /// Sets the nucleus-sampling cutoff (`top_p`).
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.request.top_p = Some(top_p);
        self
    }

    /// Freezes the builder's current state into a [`StructuredRequest`] without
    /// sending it.
    pub fn to_request(&self) -> StructuredRequest {
        self.request.clone()
    }

    /// Sends the request and returns the model's reply, parsed as JSON matching
    /// the schema you provided.
    pub async fn generate(self) -> Result<StructuredResponse, Error> {
        self.provider.structured(self.request).await
    }
}
