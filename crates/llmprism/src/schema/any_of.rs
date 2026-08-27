use serde::{Deserialize, Serialize};

use super::Schema;

/// A schema for a value that may take any one of several different shapes.
///
/// Use this when a field can legitimately be, say, either a string or a number,
/// and you want the model to pick whichever fits.
///
/// # Example
///
/// ```
/// use llmprism::schema::{AnyOfSchema, NumberSchema, Schema, StringSchema};
///
/// // Accepts either a numeric id or a string slug.
/// let identifier = AnyOfSchema::new(
///     "identifier",
///     [Schema::Number(NumberSchema::new("id")), Schema::String(StringSchema::new("slug"))],
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnyOfSchema {
    /// The field/parameter name this schema describes.
    pub name: String,
    /// A plain-language explanation of what this value represents, shown to the
    /// model so it can fill it in correctly.
    pub description: Option<String>,
    /// The set of shapes this value may take -- any one of these is valid.
    pub schemas: Vec<Schema>,
}

impl AnyOfSchema {
    /// Creates a schema for a value named `name` that may match any one of
    /// `schemas`.
    pub fn new(name: impl Into<String>, schemas: impl IntoIterator<Item = Schema>) -> Self {
        Self {
            name: name.into(),
            description: None,
            schemas: schemas.into_iter().collect(),
        }
    }

    /// Attaches a plain-language description, shown to the model.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}
