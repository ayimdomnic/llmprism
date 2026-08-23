use serde::{Deserialize, Serialize};

use super::Schema;

/// A schema for a list of values, all sharing the same shape.
///
/// # Example
///
/// ```
/// use llmprism::schema::{ArraySchema, Schema, StringSchema};
///
/// // A list of ingredient names.
/// let ingredients = ArraySchema::new("ingredients", Schema::String(StringSchema::new("ingredient")));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArraySchema {
    /// The field/parameter name this schema describes.
    pub name: String,
    /// A plain-language explanation of what this list represents, shown to the
    /// model so it can fill it in correctly.
    pub description: Option<String>,
    /// The schema every element of the list must match.
    pub items: Box<Schema>,
}

impl ArraySchema {
    /// Creates a schema for a list named `name`, where every element matches
    /// `items`.
    pub fn new(name: impl Into<String>, items: Schema) -> Self {
        Self {
            name: name.into(),
            description: None,
            items: Box::new(items),
        }
    }

    /// Attaches a plain-language description, shown to the model.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}
