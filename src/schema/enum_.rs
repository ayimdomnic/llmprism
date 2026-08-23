use serde::{Deserialize, Serialize};

/// A schema for a value that must be one of a fixed set of strings.
///
/// Use this instead of [`StringSchema`](super::StringSchema) whenever there's a
/// closed set of valid answers -- it steers the model toward a value you can
/// safely match on, instead of free-form text you'd have to parse and validate
/// yourself.
///
/// # Example
///
/// ```
/// use llmprism::schema::EnumSchema;
///
/// let unit = EnumSchema::new("unit", ["celsius", "fahrenheit"])
///     .with_description("Temperature unit to report in");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumSchema {
    /// The field/parameter name this schema describes.
    pub name: String,
    /// A plain-language explanation of what this value represents, shown to the
    /// model so it can fill it in correctly.
    pub description: Option<String>,
    /// The complete set of values the model may choose from.
    pub options: Vec<String>,
}

impl EnumSchema {
    /// Creates a schema for a closed-choice value named `name`, allowing exactly
    /// the values in `options`.
    pub fn new(
        name: impl Into<String>,
        options: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            options: options.into_iter().map(Into::into).collect(),
        }
    }

    /// Attaches a plain-language description, shown to the model.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}
