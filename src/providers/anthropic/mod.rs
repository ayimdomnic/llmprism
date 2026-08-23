//! The Anthropic provider, talking to the Messages API
//! (`api.anthropic.com/v1/messages`). Enable with the `anthropic` Cargo feature.

mod config;
mod maps;
mod wire;

use async_trait::async_trait;
use reqwest_middleware::ClientWithMiddleware;

use crate::client::{build_http_client, ErrorMapper};
use crate::error::Error;
use crate::provider::Provider;
use crate::text::{Step, TextRequest};

use self::config::{ANTHROPIC_VERSION, DEFAULT_BASE_URL};
use self::wire::MessagesResponse;

/// A [`Provider`] backed by Anthropic's Messages API.
///
/// # Example
///
/// ```no_run
/// use llmprism::providers::anthropic::AnthropicProvider;
/// use llmprism::Registry;
///
/// let mut registry = Registry::new();
/// registry.register(
///     "anthropic",
///     AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY").unwrap()),
/// );
/// ```
///
/// If you're happy reading the API key from `ANTHROPIC_API_KEY` yourself, you
/// likely don't need to construct this directly -- see
/// [`Registry::from_env`](crate::Registry::from_env).
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    version: String,
    client: ClientWithMiddleware,
}

impl AnthropicProvider {
    /// Creates a provider that talks to the real Anthropic API using `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            version: ANTHROPIC_VERSION.to_string(),
            client: build_http_client(),
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
        let wire_request = maps::build_request(&request);

        let http_response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version)
            .json(&wire_request)
            .send()
            .await?;

        let status = http_response.status();
        let headers = http_response.headers().clone();
        let body_text = http_response.text().await?;

        if !status.is_success() {
            let mapper = ErrorMapper {
                provider: self.name(),
            };
            return Err(mapper.map_error_response(status, &headers, &body_text));
        }

        let wire_response: MessagesResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        Ok(maps::parse_response(wire_response))
    }
}
