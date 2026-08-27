use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An escape hatch for when you already have a JSON Schema document (or need one
/// this crate's other [`Schema`](super::Schema) variants can't express) and just
/// want to use it as-is, rather than rebuilding it field by field.
///
/// # Example
///
/// ```
/// use llmprism::schema::RawSchema;
/// use serde_json::json;
///
/// let coordinates = RawSchema::new(
///     "coordinates",
///     json!({
///         "type": "object",
///         "properties": {
///             "lat": {"type": "number"},
///             "lon": {"type": "number"},
///         },
///         "required": ["lat", "lon"],
///     }),
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSchema {
    /// The field/parameter name this schema describes.
    pub name: String,
    /// A plain-language explanation of what this value represents. Optional here
    /// since `json_schema` may already carry its own `"description"` key.
    pub description: Option<String>,
    /// The JSON Schema document to send to the provider, unmodified.
    pub json_schema: Value,
}

impl RawSchema {
    /// Creates a schema named `name` that's sent to the provider exactly as
    /// `json_schema`, with no further processing.
    pub fn new(name: impl Into<String>, json_schema: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            json_schema,
        }
    }

    /// Attaches a plain-language description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}
