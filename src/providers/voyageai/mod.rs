//! The VoyageAI provider, talking to VoyageAI's Embeddings API
//! (`api.voyageai.com/v1/embeddings`). VoyageAI is an embeddings-only
//! service -- there's no chat/completions endpoint to speak of, so this is
//! the one and only capability this provider implements. Enable with the
//! `voyageai` Cargo feature.

mod embeddings;

use async_trait::async_trait;
use reqwest_middleware::ClientWithMiddleware;

use crate::client::{build_http_client, merge_provider_options, ErrorMapper};
use crate::embeddings::{EmbeddingsRequest, EmbeddingsResponse};
use crate::error::Error;
use crate::provider::Provider;

const DEFAULT_BASE_URL: &str = "https://api.voyageai.com/v1";

/// A [`Provider`] backed by VoyageAI's Embeddings API.
///
/// # Example
///
/// ```no_run
/// use llmprism::providers::voyageai::VoyageAiProvider;
/// use llmprism::Registry;
///
/// let mut registry = Registry::new();
/// registry.register(
///     "voyageai",
///     VoyageAiProvider::new(std::env::var("VOYAGEAI_API_KEY").unwrap()),
/// );
/// ```
///
/// If you're happy reading the API key from `VOYAGEAI_API_KEY` yourself, you
/// likely don't need to construct this directly -- see
/// [`Registry::from_env`](crate::Registry::from_env).
pub struct VoyageAiProvider {
    api_key: String,
    base_url: String,
    client: ClientWithMiddleware,
}

impl VoyageAiProvider {
    /// Creates a provider that talks to the real VoyageAI API using
    /// `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Creates a provider pointed at a different base URL.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: build_http_client(),
        }
    }

    /// Replaces the underlying HTTP client -- an escape hatch for anything
    /// [`build_http_client`] doesn't cover
    /// (a request timeout, a proxy, a custom retry policy). See
    /// `OpenAiProvider::with_client`'s docs (not linked here since that type
    /// only exists with the `openai` feature enabled) for a full example;
    /// this works the same way.
    pub fn with_client(mut self, client: ClientWithMiddleware) -> Self {
        self.client = client;
        self
    }
}

#[async_trait]
impl Provider for VoyageAiProvider {
    fn name(&self) -> &str {
        "voyageai"
    }

    async fn embeddings(&self, request: EmbeddingsRequest) -> Result<EmbeddingsResponse, Error> {
        let wire_request = embeddings::build_request(&request);
        let body = merge_provider_options(&wire_request, &request.provider_options)?;

        let http_response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
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

        let wire_response: embeddings::ApiResponse =
            serde_json::from_str(&body_text).map_err(|e| Error::Decode {
                provider: self.name().to_string(),
                message: e.to_string(),
            })?;

        Ok(embeddings::parse_response(wire_response))
    }
}
