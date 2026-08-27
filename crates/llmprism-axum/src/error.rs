//! [`ApiError`] -- turns an [`llmprism::Error`] into an HTTP response, so
//! every handler can just propagate `?` and get a sensible status code and
//! JSON body for free.
//!
//! | `llmprism::Error` variant | HTTP status |
//! |---|---|
//! | `RateLimited` | 429 Too Many Requests |
//! | `RequestTooLarge` | 413 Payload Too Large |
//! | `Overloaded` | 503 Service Unavailable |
//! | `UnknownProvider` | 404 Not Found |
//! | `Unsupported` | 501 Not Implemented |
//! | `Config` | 500 Internal Server Error |
//! | everything else | 502 Bad Gateway |
//!
//! The response body is always `{"error": {"message": "..."}}`. A
//! mid-stream error on `/v1/text/stream` or `/v1/structured/stream` carries
//! the same `message` text, but as an SSE `event: error` frame instead --
//! HTTP status can't change after an SSE response has already started, so
//! streaming routes report failure as data rather than as a failed
//! response.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

/// Wraps [`llmprism::Error`] with an [`IntoResponse`] impl. Every
/// non-streaming handler in this crate returns `Result<_, ApiError>`, so `?`
/// on any fallible `llmprism` call just works. See the [module docs](self)
/// for the exact status-code mapping.
pub struct ApiError(pub llmprism::Error);

impl From<llmprism::Error> for ApiError {
    fn from(error: llmprism::Error) -> Self {
        Self(error)
    }
}

/// The `{"error": {"message": "..."}}` body every error response in this
/// crate uses -- `pub(crate)` (not exposed publicly) so
/// [`audio`](crate::audio)'s own error type, which has a failure mode
/// [`ApiError`] doesn't (invalid base64 in a request body, never reaching a
/// `llmprism::Error` at all), can still produce byte-for-byte the same
/// response shape.
#[derive(Serialize)]
pub(crate) struct ErrorBody {
    pub(crate) error: ErrorDetail,
}

#[derive(Serialize)]
pub(crate) struct ErrorDetail {
    pub(crate) message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        use llmprism::Error;

        let status = match &self.0 {
            Error::RateLimited { .. } => StatusCode::TOO_MANY_REQUESTS,
            Error::RequestTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            Error::Overloaded { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Error::UnknownProvider(_) => StatusCode::NOT_FOUND,
            Error::Unsupported { .. } => StatusCode::NOT_IMPLEMENTED,
            Error::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
            // Everything else (Provider/Decode/StreamDecode/StructuredDecode/
            // Tool/Json/Http/Middleware/Mcp) is either an upstream provider
            // problem or a lower-level failure this crate can't usefully
            // subdivide further -- a generic gateway error either way.
            _ => StatusCode::BAD_GATEWAY,
        };

        let body = ErrorBody {
            error: ErrorDetail {
                message: self.0.to_string(),
            },
        };

        (status, Json(body)).into_response()
    }
}
