use std::sync::Arc;

use async_stream::try_stream;
use futures::stream::{BoxStream, StreamExt};

use crate::error::Error;
use crate::provider::Provider;
use crate::schema::ObjectSchema;
use crate::value_objects::{Message, UserMessage};

use super::repair::RepairStrategy;
use super::response::{StructuredResponse, StructuredStreamEvent};

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
    /// Marks the system prompt as a cache breakpoint on providers that
    /// support explicit prompt caching (currently just Anthropic; ignored
    /// elsewhere). See
    /// [`TextRequest::cache_system_prompt`](crate::text::TextRequest::cache_system_prompt)
    /// for when this is worth turning on. Defaults to `false`.
    pub cache_system_prompt: bool,
    /// Sets how much effort a reasoning model spends thinking before
    /// replying. See
    /// [`TextRequest::reasoning_effort`](crate::text::TextRequest::reasoning_effort)
    /// for which providers and models this applies to.
    pub reasoning_effort: Option<String>,
    /// A seed for providers whose backend can honor one. See
    /// [`TextRequest::seed`](crate::text::TextRequest::seed) for what
    /// "honor" means here. (No `stop_sequences` equivalent here, unlike
    /// [`TextRequest`](crate::text::TextRequest): a stop string can truncate
    /// otherwise-valid JSON output, which cuts against the whole point of a
    /// structured request.)
    pub seed: Option<u64>,
    /// A hook that gets one chance to salvage a reply that failed to decode,
    /// instead of the request just failing. `None` (the default) means a
    /// decode failure always returns [`Error::StructuredDecode`] as-is. See
    /// [`RepairStrategy`] and [`with_repair`](PendingStructuredRequest::with_repair).
    pub repair: Option<Arc<dyn RepairStrategy>>,
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
            cache_system_prompt: false,
            reasoning_effort: None,
            seed: None,
            repair: None,
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

    /// Marks the system prompt as a cache breakpoint on providers that
    /// support explicit prompt caching (currently just Anthropic; a no-op
    /// elsewhere). See [`StructuredRequest::cache_system_prompt`] for when
    /// this is worth turning on.
    pub fn with_prompt_caching(mut self) -> Self {
        self.request.cache_system_prompt = true;
        self
    }

    /// Sets how much effort a reasoning model spends thinking before
    /// replying. See
    /// [`StructuredRequest::reasoning_effort`] for which providers and
    /// models this applies to.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.request.reasoning_effort = Some(effort.into());
        self
    }

    /// Sets a seed for providers whose backend can honor one. See
    /// [`StructuredRequest::seed`] for what "honor" means here.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.request.seed = Some(seed);
        self
    }

    /// Attaches a hook that gets one chance to salvage a reply that fails to
    /// decode, instead of [`generate`](Self::generate) returning
    /// [`Error::StructuredDecode`] outright. See [`RepairStrategy`].
    pub fn with_repair(mut self, repair: impl RepairStrategy + 'static) -> Self {
        self.request.repair = Some(Arc::new(repair));
        self
    }

    /// Freezes the builder's current state into a [`StructuredRequest`] without
    /// sending it.
    pub fn to_request(&self) -> StructuredRequest {
        self.request.clone()
    }

    /// Sends the request and returns the model's reply, parsed as JSON matching
    /// the schema you provided.
    ///
    /// If the reply fails to decode and [`with_repair`](Self::with_repair)
    /// attached a [`RepairStrategy`], that hook gets one chance to salvage a
    /// result before this returns [`Error::StructuredDecode`].
    pub async fn generate(self) -> Result<StructuredResponse, Error> {
        let repair = self.request.repair.clone();

        let error = match self.provider.structured(self.request).await {
            Ok(response) => return Ok(response),
            Err(error) => error,
        };

        let Some(repair) = repair else {
            return Err(error);
        };

        let Error::StructuredDecode { context, .. } = &error else {
            return Err(error);
        };

        match repair.repair(&context.raw, &error).await {
            Some(data) => Ok(StructuredResponse {
                data,
                finish_reason: context.finish_reason,
                usage: context.usage,
                meta: context.meta.clone(),
            }),
            None => Err(error),
        }
    }

    /// Sends the request and returns a stream of [`StructuredStreamEvent`]s
    /// instead of waiting for the whole reply the way
    /// [`generate`](Self::generate) does -- each
    /// [`PartialObject`](StructuredStreamEvent::PartialObject) is a
    /// best-effort parse of everything generated so far, ending with exactly
    /// one [`End`](StructuredStreamEvent::End) carrying the same final
    /// result `generate` would have returned.
    ///
    /// Not every provider that supports [`generate`](Self::generate)
    /// necessarily supports this too -- see [`crate::structured`] for which
    /// ones do; calling this on one that doesn't returns a stream whose
    /// first (and only) item is `Err(Error::Unsupported)`.
    ///
    /// Unlike [`generate`](Self::generate), no [`RepairStrategy`] is applied
    /// here: a malformed final reply surfaces as an error on the stream
    /// itself, the same as any other mid-stream failure.
    pub fn stream(self) -> BoxStream<'static, Result<StructuredStreamEvent, Error>> {
        let PendingStructuredRequest { provider, request } = self;

        let stream = try_stream! {
            let mut inner = provider.stream_structured_once(&request).await?;
            while let Some(event) = inner.next().await {
                yield event?;
            }
        };

        stream.boxed()
    }
}
