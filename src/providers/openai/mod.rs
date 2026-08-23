//! The OpenAI provider, talking to the Chat Completions API
//! (`api.openai.com/v1/chat/completions`). Enable with the `openai` Cargo
//! feature.

mod config;
mod maps;
mod wire;

use async_trait::async_trait;
use reqwest_middleware::ClientWithMiddleware;

use crate::client::{build_http_client, ErrorMapper};
use crate::error::Error;
use crate::provider::Provider;
use crate::text::{Step, TextRequest};

use self::config::DEFAULT_BASE_URL;
use self::wire::ChatResponse;

/// A [`Provider`] backed by OpenAI's Chat Completions API.
///
/// # Example
///
/// ```no_run
/// use llmprism::providers::openai::OpenAiProvider;
/// use llmprism::Registry;
///
/// let mut registry = Registry::new();
/// registry.register("openai", OpenAiProvider::new(std::env::var("OPENAI_API_KEY").unwrap()));
/// ```
///
/// If you're happy reading the API key from `OPENAI_API_KEY` yourself, you
/// likely don't need to construct this directly -- see
/// [`Registry::from_env`](crate::Registry::from_env).
pub struct OpenAiProvider {
    api_key: String,
    base_url: String,
    client: ClientWithMiddleware,
}

impl OpenAiProvider {
    /// Creates a provider that talks to the real OpenAI API using `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Creates a provider pointed at a different base URL -- useful for an
    /// OpenAI-compatible endpoint (a local model server, a proxy, an
    /// alternative provider that mirrors OpenAI's API shape) rather than
    /// `api.openai.com` itself. `base_url` should not include a trailing
    /// `/chat/completions`; that's appended automatically.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: build_http_client(),
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn text_step(&self, request: TextRequest) -> Result<Step, Error> {
        let wire_request = maps::build_request(&request);

        let http_response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
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

        let wire_response: ChatResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        Ok(maps::parse_response(wire_response))
    }
}
