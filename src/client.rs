//! Shared HTTP plumbing used by every network-based provider: building the
//! underlying HTTP client and turning a provider's error response into a typed
//! [`Error`]. If you're just *using* this crate rather than implementing a new
//! provider, you shouldn't need anything in this module directly.

use std::time::Duration;

use reqwest::header::HeaderMap;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};

use crate::error::Error;
use crate::value_objects::RateLimit;

/// Builds the HTTP client every provider sends requests through.
///
/// It automatically retries a small number of times on purely transient
/// failures -- a dropped connection, a timeout, a generic `5xx` -- so a brief
/// network hiccup doesn't have to become an error your application sees.
///
/// This is deliberately *not* where rate-limiting (`429`), overload (`529`), or
/// "request too large" (`413`) responses get handled: those carry information
/// (like how long to wait before retrying) that's more useful surfaced to you as
/// a specific, inspectable [`Error`] than silently retried away. See
/// [`ErrorMapper`] for that part.
pub fn build_http_client() -> ClientWithMiddleware {
    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(2);
    ClientBuilder::new(reqwest::Client::new())
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build()
}

/// Turns a provider's non-2xx HTTP response into the right [`Error`] variant.
///
/// Each provider constructs one of these (naming itself via `provider`) and
/// calls [`map_error_response`](Self::map_error_response) once it knows a
/// request failed.
pub struct ErrorMapper<'a> {
    pub provider: &'a str,
}

impl<'a> ErrorMapper<'a> {
    /// Inspects the HTTP status, headers, and raw response body of a failed
    /// request and builds the matching [`Error`]: [`Error::RateLimited`] for
    /// `429`, [`Error::RequestTooLarge`] for `413`, [`Error::Overloaded`] for
    /// `529`, and [`Error::Provider`] for anything else.
    pub fn map_error_response(
        &self,
        status: reqwest::StatusCode,
        headers: &HeaderMap,
        body: &str,
    ) -> Error {
        let (kind, message) = parse_error_body(body);

        match status.as_u16() {
            429 => Error::RateLimited {
                provider: self.provider.to_string(),
                retry_after: header_duration(headers, "retry-after"),
                limits: parse_rate_limits(headers),
            },
            413 => Error::RequestTooLarge {
                provider: self.provider.to_string(),
                details: message,
            },
            529 => Error::Overloaded {
                provider: self.provider.to_string(),
            },
            status_code => Error::Provider {
                provider: self.provider.to_string(),
                status: status_code,
                kind,
                message,
            },
        }
    }
}

/// Pulls a human-readable message (and, if present, an error "type" string) out
/// of a provider's JSON error body. Providers overwhelmingly shape their error
/// bodies as `{"error": {"message": "...", "type": "..."}}`; when a body doesn't
/// parse as JSON at all, the raw text is used as the message instead of failing.
fn parse_error_body(body: &str) -> (Option<String>, String) {
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(value) => {
            let message = value
                .pointer("/error/message")
                .and_then(|v| v.as_str())
                .unwrap_or(body)
                .to_string();
            let kind = value
                .pointer("/error/type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            (kind, message)
        }
        Err(_) => (None, body.to_string()),
    }
}

fn header_duration(headers: &HeaderMap, name: &str) -> Option<Duration> {
    headers
        .get(name)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

fn parse_rate_limits(headers: &HeaderMap) -> Vec<RateLimit> {
    let limit = header_u64(headers, "x-ratelimit-limit-requests");
    let remaining = header_u64(headers, "x-ratelimit-remaining-requests");

    match (limit, remaining) {
        (Some(limit), Some(remaining)) => vec![RateLimit {
            name: "requests".to_string(),
            limit,
            remaining,
            reset_at: headers
                .get("x-ratelimit-reset-requests")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string()),
        }],
        _ => Vec::new(),
    }
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    headers.get(name)?.to_str().ok()?.parse::<u64>().ok()
}
