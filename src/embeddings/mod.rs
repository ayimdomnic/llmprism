//! Embeddings -- turn text into numeric vectors for similarity search,
//! clustering, retrieval, and so on. Start with
//! [`crate::Registry::embeddings`].

pub mod request;
pub mod response;

pub use request::{EmbeddingsRequest, PendingEmbeddingsRequest};
pub use response::{Embedding, EmbeddingsResponse};
