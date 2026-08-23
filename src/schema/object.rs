use serde::{Deserialize, Serialize};

use super::Schema;

/// A schema for a JSON object made up of named fields -- this is what
/// [`Tool::parameters`](crate::Tool::parameters) returns, and it's how you
/// describe a whole tool's argument list.
///
/// Each entry in `properties` carries its own name (every [`Schema`] variant has
/// one), so `properties` is a plain list rather than a name-keyed map -- you never
/// have to keep a field's key and its schema's name in sync separately.
///
/// # Example
///
/// ```
/// use llmprism::schema::{NumberSchema, ObjectSchema, Schema, StringSchema};
///
/// let parameters = ObjectSchema::new("parameters")
///     .with_property(Schema::String(StringSchema::new("city")), true) // required
///     .with_property(Schema::Number(NumberSchema::new("days_ahead")), false); // optional
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectSchema {
    /// The field/parameter name this schema describes (for a tool's top-level
    /// parameter list, this is usually just a placeholder like `"parameters"`,
    /// since it isn't itself nested inside another object).
    pub name: String,
    /// A plain-language explanation of what this object represents, shown to the
    /// model so it can fill it in correctly.
    pub description: Option<String>,
    /// The object's fields. Each entry's own `name()` is its key.
    pub properties: Vec<Schema>,
    /// The names of fields the model must always provide (as opposed to fields
    /// it may omit).
    pub required: Vec<String>,
    /// Whether the model is allowed to include fields beyond the ones listed in
    /// `properties`. Defaults to `false` -- most providers expect a closed set of
    /// fields.
    pub allow_additional_properties: bool,
}

impl ObjectSchema {
    /// Creates an empty object schema named `name`, with no fields yet. Add
    /// fields with [`with_property`](Self::with_property).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            properties: Vec::new(),
            required: Vec::new(),
            allow_additional_properties: false,
        }
    }

    /// Attaches a plain-language description, shown to the model.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Adds one field to the object. `schema`'s own name becomes the field's key.
    /// Set `required` to `true` if the model must always provide this field.
    pub fn with_property(mut self, schema: Schema, required: bool) -> Self {
        if required {
            self.required.push(schema.name().to_string());
        }
        self.properties.push(schema);
        self
    }

    /// Sets whether the model may include fields beyond the ones you've listed.
    pub fn with_additional_properties(mut self, allow: bool) -> Self {
        self.allow_additional_properties = allow;
        self
    }
}
