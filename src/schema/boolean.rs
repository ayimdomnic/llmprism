use serde::{Deserialize, Serialize};

/// A schema for a true/false value.
///
/// # Example
///
/// ```
/// use llmprism::schema::BooleanSchema;
///
/// let urgent = BooleanSchema::new("urgent").with_description("Whether this request is time-sensitive");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BooleanSchema {
    /// The field/parameter name this schema describes.
    pub name: String,
    /// A plain-language explanation of what this value represents, shown to the
    /// model so it can fill it in correctly.
    pub description: Option<String>,
}

impl BooleanSchema {
    /// Creates a schema for a boolean named `name`, with no description yet.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
        }
    }

    /// Attaches a plain-language description, shown to the model.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}
