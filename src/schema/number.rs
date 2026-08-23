use serde::{Deserialize, Serialize};

/// A schema for a numeric value (integer or floating-point).
///
/// # Example
///
/// ```
/// use llmprism::schema::NumberSchema;
///
/// let quantity = NumberSchema::new("quantity").with_description("How many items to order");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberSchema {
    /// The field/parameter name this schema describes.
    pub name: String,
    /// A plain-language explanation of what this value represents, shown to the
    /// model so it can fill it in correctly.
    pub description: Option<String>,
}

impl NumberSchema {
    /// Creates a schema for a number named `name`, with no description yet.
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
