use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::value_objects::Meta;

/// The result of a moderation call: one [`ModerationResult`] per input string,
/// in the same order the inputs were given.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResponse {
    pub results: Vec<ModerationResult>,
    pub meta: Meta,
}

/// Whether one piece of input content was flagged, and why.
///
/// `categories`/`category_scores` are kept as maps rather than a fixed struct
/// with one field per category: providers don't agree on the exact category
/// set (and OpenAI itself has changed it across moderation model versions), so
/// a map degrades gracefully as that set changes instead of silently dropping
/// fields an older version of this crate didn't know about.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResult {
    /// `true` if the provider considers this input to violate its usage
    /// policies in any category.
    pub flagged: bool,
    /// Which specific categories were flagged, keyed by the provider's own
    /// category name (e.g. `"violence"`, `"hate/threatening"`).
    pub categories: HashMap<String, bool>,
    /// A confidence score per category, generally in `0.0..=1.0` (higher means
    /// more confident that category applies), keyed the same way as
    /// `categories`.
    pub category_scores: HashMap<String, f64>,
}
