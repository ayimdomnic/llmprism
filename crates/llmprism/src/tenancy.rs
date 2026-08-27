//! [`RequestContext`], [`TenantRegistry`], and [`UsageTrackingMiddleware`] --
//! the hooks an application builds per-tenant provider resolution and usage
//! tracking on top of, without this crate knowing anything about how
//! identity is actually established.
//!
//! This crate never verifies a token, session, or API key itself -- that's
//! squarely application code (or, over HTTP, whatever auth middleware/layer
//! the application already has). What lives here is what comes *after*
//! identity is established: given a [`RequestContext`], which [`Registry`]
//! (which API keys, which default models, which allowed providers) should
//! handle this request, and how is usage attributed back to whoever it
//! belongs to.
//!
//! ```
//! use llmprism::tenancy::{RequestContext, StaticTenantRegistry, TenantRegistry};
//! use llmprism::testing::{FakeProvider, FakeTextResponse};
//! use llmprism::Registry;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let mut acme = Registry::new();
//! acme.register(
//!     "openai",
//!     FakeProvider::new("openai").respond_with(FakeTextResponse::new("Hi, Acme.")),
//! );
//!
//! let mut globex = Registry::new();
//! globex.register(
//!     "openai",
//!     FakeProvider::new("openai").respond_with(FakeTextResponse::new("Hi, Globex.")),
//! );
//!
//! let tenants = StaticTenantRegistry::new()
//!     .with_tenant("acme", acme)
//!     .with_tenant("globex", globex);
//!
//! let registry = tenants.resolve(&RequestContext::new("acme")).await.unwrap();
//! let response = registry
//!     .text("openai", "gpt-4o-mini")
//!     .unwrap()
//!     .with_prompt("hi")
//!     .generate()
//!     .await
//!     .unwrap();
//! assert_eq!(response.text.as_deref(), Some("Hi, Acme."));
//! # }
//! ```
//!
//! `llmprism-axum` builds a `TenantContext` Axum extractor and a
//! `routes_multi_tenant` entry point on top of this -- see that crate's own
//! docs. This crate's own reference [`StaticTenantRegistry`] resolves a
//! fixed, startup-time set of tenants; an application whose tenants come
//! from a database instead implements [`TenantRegistry`] directly against
//! it.

use std::collections::HashMap;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};

use crate::error::Error;
use crate::middleware::ProviderMiddleware;
use crate::provider::Provider;
use crate::registry::Registry;
use crate::stream_event::StreamEvent;
use crate::text::{Step, TextRequest};
use crate::value_objects::Usage;

/// The identity a request was made under, established by application code
/// (or, over HTTP, the application's own auth middleware) before this
/// crate is ever involved.
///
/// `claims` is the same kind of raw-JSON escape hatch
/// [`TextRequest::provider_options`] already is elsewhere in this crate --
/// whatever extra identity data an application's own auth needs to carry
/// through (roles, scopes, an email) without this crate having to model it
/// as typed fields.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Which tenant this request belongs to -- the key [`TenantRegistry::resolve`]
    /// looks up.
    pub tenant_id: String,
    /// The individual user within that tenant, if the application tracks
    /// one (a tenant might be a whole company, with many users inside it).
    pub user_id: Option<String>,
    /// Whatever else the application's own auth wants to carry through.
    /// `Value::Null` (the default) carries nothing extra.
    pub claims: serde_json::Value,
}

impl RequestContext {
    /// Creates a context for `tenant_id`, with no user id and no extra claims.
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            user_id: None,
            claims: serde_json::Value::Null,
        }
    }

    /// Attaches the individual user within the tenant this request belongs to.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Attaches extra application-defined claims.
    pub fn with_claims(mut self, claims: serde_json::Value) -> Self {
        self.claims = claims;
        self
    }
}

/// Resolves a [`RequestContext`] to the [`Registry`] that tenant's requests
/// should be handled by -- different API keys, different default models,
/// different allowed providers per tenant, instead of one process-wide
/// [`Registry::from_env`].
#[async_trait]
pub trait TenantRegistry: Send + Sync {
    /// Looks up which [`Registry`] should handle a request from `context`.
    /// An unrecognized `context.tenant_id` is a normal, expected failure
    /// mode -- return [`Error::Store`], not a panic.
    async fn resolve(&self, context: &RequestContext) -> Result<Arc<Registry>, Error>;
}

#[async_trait]
impl<T: TenantRegistry + ?Sized> TenantRegistry for Arc<T> {
    async fn resolve(&self, context: &RequestContext) -> Result<Arc<Registry>, Error> {
        T::resolve(self, context).await
    }
}

/// A [`TenantRegistry`] backed by a fixed, startup-time set of tenants --
/// useful when the set of tenants and their credentials come from
/// application config/environment rather than changing at runtime. An
/// application whose tenants are added/removed dynamically (from a
/// database, say) implements [`TenantRegistry`] directly instead.
#[derive(Default)]
pub struct StaticTenantRegistry {
    registries: HashMap<String, Arc<Registry>>,
}

impl StaticTenantRegistry {
    /// Creates an empty tenant registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `registry` as the one `tenant_id` resolves to.
    pub fn with_tenant(mut self, tenant_id: impl Into<String>, registry: Registry) -> Self {
        self.registries.insert(tenant_id.into(), Arc::new(registry));
        self
    }
}

#[async_trait]
impl TenantRegistry for StaticTenantRegistry {
    async fn resolve(&self, context: &RequestContext) -> Result<Arc<Registry>, Error> {
        self.registries
            .get(&context.tenant_id)
            .cloned()
            .ok_or_else(|| Error::Store {
                message: format!("unknown tenant '{}'", context.tenant_id),
            })
    }
}

/// Records token usage somewhere -- a database, a metrics counter, a
/// billing queue -- for [`UsageTrackingMiddleware`] to call after every
/// round trip.
#[async_trait]
pub trait UsageSink: Send + Sync {
    /// Records that `tenant_id` spent `usage` on one round trip. Called
    /// once per round trip, not once per `generate()`/`stream()` call --
    /// summing every recorded [`Usage`] for a tenant reconstructs the same
    /// total [`crate::text::TextResponse::usage`] would report for a given
    /// call, but at finer granularity (useful if you care how many round
    /// trips a tool-calling loop took, not just the total).
    async fn record(&self, tenant_id: &str, usage: Usage) -> Result<(), Error>;
}

#[async_trait]
impl<T: UsageSink + ?Sized> UsageSink for Arc<T> {
    async fn record(&self, tenant_id: &str, usage: Usage) -> Result<(), Error> {
        T::record(self, tenant_id, usage).await
    }
}

/// A [`UsageSink`] that keeps every recorded `(tenant_id, Usage)` pair in
/// memory, in order -- useful for tests, and as a template for a real sink
/// (a database write, a metrics increment) that would replace [`record`](UsageSink::record)'s
/// body with something that outlives the process.
#[derive(Debug, Default)]
pub struct InMemoryUsageSink {
    recorded: std::sync::Mutex<Vec<(String, Usage)>>,
}

impl InMemoryUsageSink {
    /// Creates an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns every [`Usage`] recorded for `tenant_id`, in the order it
    /// was recorded.
    pub fn usage_for(&self, tenant_id: &str) -> Vec<Usage> {
        self.recorded
            .lock()
            .expect("usage sink mutex poisoned")
            .iter()
            .filter(|(id, _)| id == tenant_id)
            .map(|(_, usage)| *usage)
            .collect()
    }
}

#[async_trait]
impl UsageSink for InMemoryUsageSink {
    async fn record(&self, tenant_id: &str, usage: Usage) -> Result<(), Error> {
        self.recorded
            .lock()
            .expect("usage sink mutex poisoned")
            .push((tenant_id.to_string(), usage));
        Ok(())
    }
}

/// A [`ProviderMiddleware`] that records token usage to a [`UsageSink`]
/// after every round trip, attributed to one fixed tenant.
///
/// Unlike [`crate::persistence::PersistenceMiddleware`], this middleware
/// carries its own `tenant_id` set at construction time, rather than
/// reading one from the request -- attribution comes from *which*
/// `Registry`/middleware instance handled the call (one per tenant, via
/// [`TenantRegistry`]), not from data threaded through the request itself.
/// Wrap one into each tenant's own `Registry` (e.g. inside your
/// [`TenantRegistry`] implementation, before handing it out) via
/// [`Registry::wrap`].
pub struct UsageTrackingMiddleware<S: UsageSink> {
    tenant_id: String,
    sink: Arc<S>,
}

impl<S: UsageSink> UsageTrackingMiddleware<S> {
    /// Records every round trip's usage against `tenant_id`, via `sink`.
    pub fn new(tenant_id: impl Into<String>, sink: S) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            sink: Arc::new(sink),
        }
    }
}

#[async_trait]
impl<S: UsageSink + 'static> ProviderMiddleware for UsageTrackingMiddleware<S> {
    async fn text_step(&self, request: TextRequest, next: &dyn Provider) -> Result<Step, Error> {
        let step = next.text_step(&request).await?;
        self.sink.record(&self.tenant_id, step.usage).await?;
        Ok(step)
    }

    async fn stream_text_once(
        &self,
        request: TextRequest,
        next: &dyn Provider,
    ) -> Result<BoxStream<'static, Result<StreamEvent, Error>>, Error> {
        let mut inner = next.stream_text_once(&request).await?;
        let tenant_id = self.tenant_id.clone();
        let sink = Arc::clone(&self.sink);

        let stream = try_stream! {
            while let Some(event) = inner.next().await {
                let event = event?;
                if let StreamEvent::StepFinish { usage, .. } = &event {
                    sink.record(&tenant_id, *usage).await?;
                }
                yield event;
            }
        };

        Ok(stream.boxed())
    }
}
