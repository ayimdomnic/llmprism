//! Benchmarks `schema::to_json_schema` -- the function every tool call and
//! structured-output request runs at least once (to build the schema sent to
//! the provider), so its cost scales directly with how complex an
//! application's tool/output schemas are.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use llmprism::schema::{
    AnyOfSchema, ArraySchema, BooleanSchema, EnumSchema, NumberSchema, ObjectSchema, Schema,
    StringSchema,
};

/// A realistic tool-argument schema: half a dozen flat fields of mixed types,
/// the common case for most tools.
fn flat_schema() -> ObjectSchema {
    ObjectSchema::new("parameters")
        .with_property(
            Schema::String(StringSchema::new("city").with_description("City name")),
            true,
        )
        .with_property(
            Schema::String(StringSchema::new("country").with_description("Country code")),
            false,
        )
        .with_property(
            Schema::Number(NumberSchema::new("days_ahead").with_description("Forecast horizon")),
            false,
        )
        .with_property(Schema::Boolean(BooleanSchema::new("include_hourly")), false)
        .with_property(
            Schema::Enum(EnumSchema::new("unit", ["celsius", "fahrenheit"])),
            true,
        )
        .with_property(
            Schema::AnyOf(AnyOfSchema::new(
                "identifier",
                [
                    Schema::Number(NumberSchema::new("id")),
                    Schema::String(StringSchema::new("slug")),
                ],
            )),
            false,
        )
}

/// A structured-output shape with real nesting: an object containing an array
/// of objects, each with their own nested array -- closer to what a
/// real-world extraction/reporting schema looks like than a flat argument
/// list, and exercises `to_json_schema`'s recursion.
fn nested_schema() -> ObjectSchema {
    let line_item = Schema::Object(
        ObjectSchema::new("line_item")
            .with_property(Schema::String(StringSchema::new("sku")), true)
            .with_property(Schema::Number(NumberSchema::new("quantity")), true)
            .with_property(Schema::Number(NumberSchema::new("unit_price")), true)
            .with_property(
                Schema::Array(ArraySchema::new(
                    "tags",
                    Schema::String(StringSchema::new("tag")),
                )),
                false,
            ),
    );

    let order = Schema::Object(
        ObjectSchema::new("order")
            .with_property(Schema::String(StringSchema::new("order_id")), true)
            .with_property(Schema::String(StringSchema::new("customer_email")), true)
            .with_property(
                Schema::Array(ArraySchema::new("line_items", line_item)),
                true,
            ),
    );

    ObjectSchema::new("report")
        .with_property(Schema::String(StringSchema::new("generated_at")), true)
        .with_property(Schema::Array(ArraySchema::new("orders", order)), true)
}

fn bench_to_json_schema(c: &mut Criterion) {
    let flat = Schema::Object(flat_schema());
    let nested = Schema::Object(nested_schema());

    c.bench_function("to_json_schema/flat_object", |b| {
        b.iter(|| llmprism::schema::to_json_schema(black_box(&flat)))
    });

    c.bench_function("to_json_schema/nested_object", |b| {
        b.iter(|| llmprism::schema::to_json_schema(black_box(&nested)))
    });
}

criterion_group!(benches, bench_to_json_schema);
criterion_main!(benches);
