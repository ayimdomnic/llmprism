//! Shared plumbing for turning a `BoxStream<Result<T, llmprism::Error>>` into
//! an SSE byte stream that never fails: every `Ok` becomes an `event:
//! message` frame, every `Err` becomes an `event: error` frame, and the
//! stream itself always yields `Ok(Event)` -- HTTP status can't change after
//! an SSE response has started, so a mid-stream provider failure has to be
//! reported as data, not as a failed response.

use std::convert::Infallible;

use axum::response::sse::Event;
use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use serde::Serialize;

pub(crate) fn sse_stream<T>(
    stream: BoxStream<'static, Result<T, llmprism::Error>>,
) -> impl Stream<Item = Result<Event, Infallible>>
where
    T: Serialize + Send + 'static,
{
    stream.map(|item| {
        let event = match item {
            Ok(value) => Event::default()
                .event("message")
                .json_data(&value)
                .unwrap_or_else(|error| Event::default().event("error").data(error.to_string())),
            Err(error) => Event::default().event("error").data(error.to_string()),
        };
        Ok(event)
    })
}
