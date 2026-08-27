//! Describing the *shape* of data in a way every provider understands.
//!
//! The same [`Schema`] type describes both a [`Tool`](crate::Tool)'s
//! arguments and a [`structured`](crate::structured)-output request's
//! required shape. Build one from the pieces in this module (start with
//! [`ObjectSchema`] for a tool's whole argument list, or a structured
//! request's top-level shape), and each provider takes care of translating it
//! into its own particular JSON Schema dialect -- you never write raw JSON
//! Schema by hand unless you want to, via [`RawSchema`].

mod any_of;
mod array;
mod boolean;
mod enum_;
mod number;
mod object;
mod raw;
mod string;

pub use any_of::AnyOfSchema;
pub use array::ArraySchema;
pub use boolean::BooleanSchema;
pub use enum_::EnumSchema;
pub use number::NumberSchema;
pub use object::ObjectSchema;
pub use raw::RawSchema;
pub use string::StringSchema;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Describes the shape one value must have -- a string, a number, a nested
/// object, and so on.
///
/// Each variant wraps a small struct with the details (a name, an optional
/// description, and whatever's specific to that shape). Every variant's inner
/// struct has its own `::new(...)` constructor and `with_description(...)`
/// builder method, so building a schema reads naturally, for example:
///
/// ```
/// use llmprism::schema::{Schema, StringSchema};
///
/// let city = Schema::String(StringSchema::new("city").with_description("City name"));
/// ```
///
/// Most of the time you'll build a [`Schema::Object`] to describe a whole set of
/// named fields (see [`ObjectSchema`]) rather than a single bare value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Schema {
    String(StringSchema),
    Number(NumberSchema),
    Boolean(BooleanSchema),
    Array(ArraySchema),
    Object(ObjectSchema),
    Enum(EnumSchema),
    AnyOf(AnyOfSchema),
    /// A hand-written JSON Schema document, for cases the other variants can't
    /// express. See [`RawSchema`].
    Raw(RawSchema),
}

impl Schema {
    /// The name of the field or parameter this schema describes, regardless of
    /// which variant it is.
    pub fn name(&self) -> &str {
        match self {
            Schema::String(s) => &s.name,
            Schema::Number(s) => &s.name,
            Schema::Boolean(s) => &s.name,
            Schema::Array(s) => &s.name,
            Schema::Object(s) => &s.name,
            Schema::Enum(s) => &s.name,
            Schema::AnyOf(s) => &s.name,
            Schema::Raw(s) => &s.name,
        }
    }
}

/// Converts a [`Schema`] into a standard JSON Schema document.
///
/// This is the shape both OpenAI's and Anthropic's tool-parameter payloads
/// accept directly, so their provider implementations call this as-is. A
/// provider with a genuinely different dialect (for example, one based on
/// OpenAPI rather than JSON Schema) will get its own mapping function when that
/// provider is implemented, instead of using this one.
pub fn to_json_schema(schema: &Schema) -> Value {
    match schema {
        Schema::String(s) => with_description(json!({"type": "string"}), s.description.as_deref()),
        Schema::Number(s) => with_description(json!({"type": "number"}), s.description.as_deref()),
        Schema::Boolean(s) => {
            with_description(json!({"type": "boolean"}), s.description.as_deref())
        }
        Schema::Array(s) => {
            let mut value = with_description(json!({"type": "array"}), s.description.as_deref());
            value["items"] = to_json_schema(&s.items);
            value
        }
        Schema::Object(s) if s.json_schema.is_some() => s
            .json_schema
            .clone()
            .expect("just checked is_some via the match guard"),
        Schema::Object(s) => {
            let mut properties = serde_json::Map::new();
            for prop in &s.properties {
                properties.insert(prop.name().to_string(), to_json_schema(prop));
            }
            let mut value = with_description(
                json!({
                    "type": "object",
                    "properties": properties,
                    "required": s.required,
                }),
                s.description.as_deref(),
            );
            value["additionalProperties"] = json!(s.allow_additional_properties);
            value
        }
        Schema::Enum(s) => with_description(
            json!({"type": "string", "enum": s.options}),
            s.description.as_deref(),
        ),
        Schema::AnyOf(s) => with_description(
            json!({"anyOf": s.schemas.iter().map(to_json_schema).collect::<Vec<_>>()}),
            s.description.as_deref(),
        ),
        Schema::Raw(s) => s.json_schema.clone(),
    }
}

fn with_description(mut value: Value, description: Option<&str>) -> Value {
    if let Some(description) = description {
        value["description"] = json!(description);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_json_schema_returns_a_raw_object_schema_unchanged() {
        let raw = json!({
            "type": "object",
            "properties": {"anything": {"type": "string", "const": "not a Schema variant"}},
            "oneOf": [{"required": ["a"]}, {"required": ["b"]}],
        });
        let schema = Schema::Object(ObjectSchema::from_raw_json_schema("weird", raw.clone()));

        assert_eq!(to_json_schema(&schema), raw);
    }

    #[test]
    fn from_raw_json_schema_still_sets_the_ordinary_name_field() {
        let schema = ObjectSchema::from_raw_json_schema("my_tool", json!({"type": "object"}));

        assert_eq!(schema.name, "my_tool");
    }
}
