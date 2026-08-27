use std::sync::Arc;

use crate::error::Error;
use crate::provider::Provider;

use super::response::RerankResponse;

/// The immutable, provider-agnostic shape of one rerank call: score and sort
/// a list of documents by relevance to a query.
#[derive(Clone)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    /// The documents to score, in their original order. [`RankedDocument`]
    /// results reference these by [`RankedDocument::index`], not by
    /// re-sending the text, unless [`return_documents`](Self::return_documents)
    /// is set.
    ///
    /// [`RankedDocument`]: super::response::RankedDocument
    /// [`RankedDocument::index`]: super::response::RankedDocument::index
    pub documents: Vec<String>,
    /// Keep only this many of the most relevant documents. `None` returns
    /// every document, just sorted.
    pub top_k: Option<u32>,
    /// Asks the provider to echo each document's own text back in the
    /// response instead of just its index. Defaults to `false` -- you
    /// already have `documents` locally, so this mostly matters if you'd
    /// rather not keep your own copy around to match indices back up.
    pub return_documents: bool,
    /// Extra provider-specific fields to send alongside this request, for
    /// options this crate doesn't model as a typed field yet. Must be a JSON
    /// object to have any effect: each of its top-level keys is merged into
    /// (and, if it collides with one of this crate's own fields, overrides)
    /// the request body actually sent to the provider. The default,
    /// `Value::Null`, sends nothing extra.
    pub provider_options: serde_json::Value,
}

impl RerankRequest {
    pub fn new(model: impl Into<String>, query: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            query: query.into(),
            documents: Vec::new(),
            top_k: None,
            return_documents: false,
            provider_options: serde_json::Value::Null,
        }
    }
}

/// The fluent, chainable way to build and run a rerank request.
///
/// Get one of these from [`Registry::rerank`](crate::Registry::rerank), add
/// one or more [`with_document`](Self::with_document) calls, then
/// [`generate`](Self::generate).
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "voyageai")]
/// # async fn example() -> Result<(), llmprism::Error> {
/// use llmprism::Registry;
///
/// let registry = Registry::from_env();
/// let response = registry
///     .rerank("voyageai", "rerank-2.5", "What's the capital of France?")?
///     .with_document("Paris is the capital of France.")
///     .with_document("Berlin is the capital of Germany.")
///     .generate()
///     .await?;
///
/// // Sorted most-relevant first -- `results[0]` is the best match.
/// println!("best match was document {}", response.results[0].index);
/// # Ok(())
/// # }
/// ```
pub struct PendingRerankRequest {
    provider: Arc<dyn Provider>,
    request: RerankRequest,
}

impl PendingRerankRequest {
    /// Starts a new builder for `provider`, targeting `model` with the given
    /// `query`. You'll normally get one of these from
    /// [`Registry::rerank`](crate::Registry::rerank) rather than calling
    /// this directly.
    pub fn new(
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
        query: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            request: RerankRequest::new(model, query),
        }
    }

    /// Adds one document to score. Call this more than once to rerank
    /// several documents in a single request.
    pub fn with_document(mut self, document: impl Into<String>) -> Self {
        self.request.documents.push(document.into());
        self
    }

    /// Adds several documents to score at once.
    pub fn with_documents(
        mut self,
        documents: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.request
            .documents
            .extend(documents.into_iter().map(Into::into));
        self
    }

    /// Keeps only the `top_k` most relevant documents in the response,
    /// instead of every document just sorted.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.request.top_k = Some(top_k);
        self
    }

    /// Asks the provider to echo each document's own text back in the
    /// response. See [`RerankRequest::return_documents`] for why you'd
    /// bother, given you already have the text locally.
    pub fn with_return_documents(mut self, return_documents: bool) -> Self {
        self.request.return_documents = return_documents;
        self
    }

    /// Freezes the builder's current state into a [`RerankRequest`] without
    /// sending it.
    pub fn to_request(&self) -> RerankRequest {
        self.request.clone()
    }

    /// Sends the request and returns every document, scored and sorted by
    /// relevance (most relevant first).
    pub async fn generate(self) -> Result<RerankResponse, Error> {
        self.provider.rerank(self.request).await
    }
}
