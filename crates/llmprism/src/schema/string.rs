use serde::{Deserialize, Serialize};

/// A schema for a plain text value.
///
/// # Example
///
/// ```
/// use llmprism::schema::StringSchema;
///
/// let city = StringSchema::new("city").with_description("The city to look up");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringSchema {
    /// The field/parameter name this schema describes.
    pub name: String,
    /// A plain-language explanation of what this value represents, shown to the
    /// model so it can fill it in correctly.
    pub description: Option<String>,
}

impl StringSchema {
    /// Creates a schema for a string named `name`, with no description yet.
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
