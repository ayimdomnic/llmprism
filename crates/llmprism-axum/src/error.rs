//! [`ApiError`] -- turns an [`llmprism::Error`] into an HTTP response, so
//! every handler can just propagate `?` and get a sensible status code and
//! JSON body for free.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

/// Wraps [`llmprism::Error`] with an [`IntoResponse`] impl. Every handler in
/// this crate returns `Result<_, ApiError>`, so `?` on any fallible
/// `llmprism` call just works.
pub struct ApiError(pub llmprism::Error);

impl From<llmprism::Error> for ApiError {
    fn from(error: llmprism::Error) -> Self {
        Self(error)
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
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
