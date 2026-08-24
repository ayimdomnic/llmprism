//! Reranking -- given a query and a list of documents, score and sort them
//! by relevance. Start with [`crate::Registry::rerank`], which hands you a
//! [`PendingRerankRequest`] to configure and run.
//!
//! Distinct from [`crate::embeddings`]: embedding turns text into a vector
//! you compare yourself (cosine similarity, a vector database, ...);
//! reranking sends the query and candidate documents to the provider
//! together and gets back a relevance-sorted list directly. The two are
//! often used as a pair in retrieval pipelines -- embeddings for a fast
//! first-pass candidate search, reranking to refine the top results.

pub mod request;
pub mod response;

pub use request::{PendingRerankRequest, RerankRequest};
pub use response::{RankedDocument, RerankResponse};
