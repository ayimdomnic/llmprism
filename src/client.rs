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

/// Builds the HTTP client every provider sends requests through by default.
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
///
/// There's deliberately no overall request timeout here either, even though
/// that means a hung connection can block a call indefinitely: a blanket
/// `reqwest` timeout applies to the *entire* request, including reading the
/// response body, which would incorrectly cut off a legitimate long-lived
/// streaming reply (see [`Provider::stream_text_once`](crate::Provider::stream_text_once))
/// partway through. If you want a timeout, build your own client -- most
/// likely with `.connect_timeout(...)` on the inner [`reqwest::Client`],
/// which bounds only the initial connection and is safe for streaming -- and
/// hand it to a provider's `with_client` constructor (every provider that
/// talks over HTTP has one) instead of using this function.
pub fn build_http_client() -> ClientWithMiddleware {
    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(2);
    ClientBuilder::new(reqwest::Client::new())
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build()
}

/// Serializes `wire_request` to JSON and shallow-merges `provider_options` on
/// top of the result, letting a caller add or override any field the
/// provider's real API accepts that this crate doesn't model as a typed
/// field -- the mechanism behind every request type's `provider_options`
/// escape hatch. A `provider_options` that isn't a JSON object (including the
/// default `Value::Null` every request starts with) is a no-op: nothing is
/// merged, and `wire_request`'s own fields are sent exactly as built.
///
/// Every provider that sends a JSON body calls this instead of serializing
/// `wire_request` directly, so the escape hatch behaves identically
/// everywhere rather than working for some providers and silently doing
/// nothing for others.
pub fn merge_provider_options<T: serde::Serialize>(
    wire_request: &T,
    provider_options: &serde_json::Value,
) -> Result<serde_json::Value, Error> {
    let mut body = serde_json::to_value(wire_request)?;
    if let (Some(base), Some(overrides)) = (body.as_object_mut(), provider_options.as_object()) {
        for (key, value) in overrides {
            base.insert(key.clone(), value.clone());
        }
    }
    Ok(body)
}

/// The `provider_options` escape hatch's counterpart for an endpoint that
/// sends a multipart form instead of a JSON body (the two speech-to-text
/// endpoints in this crate): adds each field of `provider_options`, if it's a
/// JSON object, to `form` as an extra text field. A string value is added
/// as-is; anything else is compacted to a JSON string, since a multipart
/// field is always text. A `provider_options` that isn't a JSON object is a
/// no-op, same as [`merge_provider_options`].
pub fn merge_provider_options_into_form(
    mut form: reqwest::multipart::Form,
    provider_options: &serde_json::Value,
) -> reqwest::multipart::Form {
    if let Some(overrides) = provider_options.as_object() {
        for (key, value) in overrides {
            let value = value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string());
            form = form.text(key.clone(), value);
        }
    }
    form
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use serde_json::json;

    #[derive(Serialize)]
    struct Example {
        model: String,
        temperature: Option<f32>,
    }

    #[test]
    fn merge_provider_options_overrides_and_adds_fields() {
        let wire_request = Example {
            model: "gpt-4o-mini".to_string(),
            temperature: Some(0.2),
        };
        let provider_options = json!({"temperature": 0.9, "seed": 42});

        let body = merge_provider_options(&wire_request, &provider_options).unwrap();

        assert_eq!(body["model"], "gpt-4o-mini");
        assert_eq!(body["temperature"], 0.9);
        assert_eq!(body["seed"], 42);
    }

    #[test]
    fn merge_provider_options_is_a_no_op_for_the_default_null_value() {
        let wire_request = Example {
            model: "gpt-4o-mini".to_string(),
            temperature: Some(0.2),
        };
        let expected = serde_json::to_value(&wire_request).unwrap();

        let body = merge_provider_options(&wire_request, &serde_json::Value::Null).unwrap();

        assert_eq!(body, expected);
    }
}
