//! Confirms `GenAiTracingMiddleware` actually records the `gen_ai.*` fields
//! it claims to, against `FakeProvider` -- no network access, no provider
//! feature required.

#![cfg(feature = "tracing")]

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::tracing_middleware::GenAiTracingMiddleware;
use llmprism::value_objects::Usage;
use llmprism::Registry;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Captures every field recorded on any span (at creation, via
/// `tracing::info_span!`'s initial fields, or later via `Span::record`) as a
/// debug-formatted string, keyed by field name -- enough to assert on
/// without depending on `tracing`'s exact internal `Visit` dispatch rules for
/// each field type.
#[derive(Default, Clone)]
struct CapturedFields(Arc<Mutex<BTreeMap<String, String>>>);

impl Visit for CapturedFields {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0
            .lock()
            .unwrap()
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

struct CaptureLayer(CapturedFields);

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for CaptureLayer {
    fn on_new_span(&self, attrs: &Attributes<'_>, _id: &Id, _ctx: Context<'_, S>) {
        attrs.record(&mut self.0.clone());
    }

    fn on_record(&self, _id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        values.record(&mut self.0.clone());
    }
}

#[tokio::test]
async fn records_gen_ai_attributes_for_a_text_step() {
    let captured = CapturedFields::default();
    let subscriber = tracing_subscriber::registry().with(CaptureLayer(captured.clone()));
    let _guard = tracing::subscriber::set_default(subscriber);

    let mut registry = Registry::new();
    registry.register(
        "fake",
        FakeProvider::new("fake").respond_with(FakeTextResponse::new("hi").with_usage(Usage {
            prompt_tokens: 3,
            completion_tokens: 5,
            ..Usage::default()
        })),
    );
    registry.wrap("fake", GenAiTracingMiddleware).unwrap();

    registry
        .text("fake", "test-model")
        .unwrap()
        .with_prompt("hello")
        .generate()
        .await
        .unwrap();

    let fields = captured.0.lock().unwrap();
    let get = |name: &str| {
        fields
            .get(name)
            .unwrap_or_else(|| panic!("no {name} field recorded"))
    };

    assert!(get("gen_ai.operation.name").contains("chat"));
    assert!(get("gen_ai.provider.name").contains("fake"));
    assert!(get("gen_ai.request.model").contains("test-model"));
    assert!(get("gen_ai.usage.input_tokens").contains('3'));
    assert!(get("gen_ai.usage.output_tokens").contains('5'));
}
