//! [`TenantContext`] -- an Axum extractor pulling a `RequestContext` an
//! application's own auth middleware already inserted into the request,
//! for use with [`crate::routes_multi_tenant`].

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use llmprism::tenancy::RequestContext;

/// Extracts the [`RequestContext`] an application's own auth middleware
/// already inserted into the request's extensions -- the standard
/// `tower`/Axum pattern: a `tower::Layer` or `axum::middleware::from_fn`
/// verifies a JWT, session, or API key and calls
/// `request.extensions_mut().insert(RequestContext::new(...))` before
/// these routes ever run. This crate never verifies a token itself; see
/// the [crate root docs](crate) for what is and isn't in scope here.
///
/// Rejects with `401 Unauthorized` if no application middleware inserted one.
pub struct TenantContext(pub RequestContext);

impl<S> FromRequestParts<S> for TenantContext
where
    S: Send + Sync,
{
    type Rejection = MissingTenantContext;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        parts
            .extensions
            .get::<RequestContext>()
            .cloned()
            .map(TenantContext)
            .ok_or(MissingTenantContext)
    }
}

/// The rejection [`TenantContext`] returns when no [`RequestContext`] was
/// found in the request's extensions.
pub struct MissingTenantContext;

impl IntoResponse for MissingTenantContext {
    fn into_response(self) -> Response {
        (
            StatusCode::UNAUTHORIZED,
            "no RequestContext in request extensions -- attach your own auth \
             middleware (a tower::Layer or axum::middleware::from_fn) that \
             inserts one before these routes run",
        )
            .into_response()
    }
}
