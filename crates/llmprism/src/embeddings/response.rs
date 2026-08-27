use serde::{Deserialize, Serialize};

use crate::value_objects::{EmbeddingsUsage, Meta};

/// The result of an embeddings call: one [`Embedding`] per input string, in
/// the same order the inputs were given.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub embeddings: Vec<Embedding>,
    pub usage: EmbeddingsUsage,
    pub meta: Meta,
}

/// A single embedding vector -- a numeric representation of one input's
/// meaning, suitable for similarity search, clustering, and so on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub vector: Vec<f32>,
}
