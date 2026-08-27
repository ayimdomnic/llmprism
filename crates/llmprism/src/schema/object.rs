use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    /// An already-formed JSON Schema document to send exactly as-is, bypassing
    /// `properties`/`required`/`allow_additional_properties` entirely. `None`
    /// (the default) builds the schema from those fields normally, the same
    /// way it always has.
    ///
    /// This is [`RawSchema`](super::RawSchema)'s escape hatch, made available
    /// here too: `RawSchema` works anywhere a general [`Schema`] fits, but
    /// [`Tool::parameters`](crate::Tool::parameters) and
    /// [`StructuredRequest::schema`](crate::structured::StructuredRequest::schema)
    /// are both typed as `ObjectSchema` specifically, so `Schema::Raw` can't be
    /// plugged in at either point. Set via
    /// [`from_raw_json_schema`](Self::from_raw_json_schema) instead of directly,
    /// for a schema you already have in hand -- from an MCP server's tool
    /// listing, or a user-supplied schema file -- rather than one you're
    /// building field by field.
    pub json_schema: Option<Value>,
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
            json_schema: None,
        }
    }

    /// Creates an object schema named `name` that sends `json_schema` exactly
    /// as-is, instead of being built from [`with_property`](Self::with_property)
    /// calls. See [`json_schema`](Self::json_schema) for when this is worth
    /// reaching for.
    ///
    /// One caveat worth knowing if you're targeting OpenAI: its
    /// structured-output request sets `"strict": true`, which requires
    /// `additionalProperties: false` and every property listed in `required`,
    /// recursively, throughout the whole schema. A raw schema that doesn't
    /// already satisfy that shape will be rejected by OpenAI at request time --
    /// this crate can't fix that up for you, since it sends `json_schema`
    /// unmodified by design.
    pub fn from_raw_json_schema(name: impl Into<String>, json_schema: Value) -> Self {
        Self {
            json_schema: Some(json_schema),
            ..Self::new(name)
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
