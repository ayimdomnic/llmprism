# llmprism-axum

[![Crates.io](https://img.shields.io/crates/v/llmprism-axum.svg)](https://crates.io/crates/llmprism-axum)
[![docs.rs](https://img.shields.io/docsrs/llmprism-axum)](https://docs.rs/llmprism-axum)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

Mounts every [`llmprism`](https://crates.io/crates/llmprism) capability as an
HTTP API, in one line:

```rust,no_run
use llmprism::Registry;

#[tokio::main]
async fn main() {
    let registry = Registry::from_env();
    let app = llmprism_axum::routes(registry);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

That's the whole integration. `routes(registry)` returns a plain
`axum::Router` -- merge it into a bigger application with `Router::merge`,
layer your own `tower` middleware on top (auth, rate limiting, tracing,
whatever your service already uses), or serve it standalone like above.

## Routes

Every route is `POST`, takes a `provider` field in its JSON body naming
which provider registered in your `Registry` to use (there's no provider
segment in the path itself -- mirrors the `llmprism` CLI's
`--provider`/`--model` pairing), and returns exactly the response type the
matching `Registry` method would.

| Route                          | What it does                                          |
| ------------------------------ | ------------------------------------------------------ |
| `POST /v1/text`                | Text generation, with tool calling and multi-turn history |
| `POST /v1/text/stream`         | Same, as Server-Sent Events                             |
| `POST /v1/structured`          | A reply matching a JSON Schema you provide              |
| `POST /v1/structured/stream`   | Same, as Server-Sent Events (best-effort partial objects, then the final result) |
| `POST /v1/moderation`          | Content-safety classification                           |
| `POST /v1/embeddings`          | Text → vector, for similarity search / retrieval        |
| `POST /v1/rerank`              | Score and sort documents against a query                |
| `POST /v1/images`              | Image generation                                        |

Full request/response shapes, worked JSON examples, and the exact
`llmprism::Error` → HTTP status mapping are documented per route on
[docs.rs](https://docs.rs/llmprism-axum) -- every request-body struct
(`TextRequestBody`, `StructuredRequestBody`, ...) is public and documented
field-by-field, so it's the source of truth for what a client can send.

### Example: non-streaming text

```sh
curl -s http://localhost:3000/v1/text \
  -H 'content-type: application/json' \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": [{"Text": "Say hello in one word."}]}]
  }'
```

### Example: streaming text (SSE)

```sh
curl -N http://localhost:3000/v1/text/stream \
  -H 'content-type: application/json' \
  -d '{
    "provider": "openai",
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": [{"Text": "Count to five."}]}]
  }'
```

Each line is `event: message` with a JSON-encoded `StreamEvent` as its
data, ending with a `StreamEvent::StreamEnd`. If something goes wrong
partway through, you get `event: error` instead -- HTTP status can't change
after an SSE response has already started, so a mid-stream failure has to
be reported as data, not as a failed response.

## Multi-tenancy

Serving several tenants (different API keys, different allowed providers,
different default models per tenant) from one process? Use
`routes_multi_tenant` instead of `routes`:

```rust,no_run
use llmprism::tenancy::StaticTenantRegistry;
use llmprism::Registry;

#[tokio::main]
async fn main() {
    let tenants = StaticTenantRegistry::new()
        .with_tenant("acme", Registry::from_env())
        .with_tenant("globex", Registry::from_env());

    let app = llmprism_axum::routes_multi_tenant(tenants);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

Every route now resolves its `Registry` *per request*, from a
`llmprism::tenancy::RequestContext` (`tenant_id`/`user_id`/free-form
claims). This crate never establishes that context itself -- attach your
own auth middleware (a `tower::Layer` or `axum::middleware::from_fn` that
verifies a JWT/session/API key) in front of these routes, and have it call
`request.extensions_mut().insert(RequestContext::new(tenant_id))`. A
request with no context attached gets `401`; an unrecognized tenant maps
through the usual error handling.

## What's deliberately not here

- **Tool calling, approval handling, and MCP** don't have an HTTP
  representation -- a `Tool` is arbitrary server-side Rust code with a
  `call()` method, which a JSON body can't describe. Attach tools directly
  to the `Registry`/`PendingTextRequest` your server builds, the same way
  you would without this crate.
- **Audio** (`/v1/audio/speech`, `/v1/audio/transcriptions`) isn't wired up
  yet -- binary bodies need a deliberate base64-vs-multipart-vs-raw-bytes
  decision, tracked as a Phase 1 follow-up in
  [`ROADMAP.md`](../../ROADMAP.md).

## Installing

```toml
[dependencies]
llmprism = { version = "0.3", features = ["openai", "anthropic"] }
llmprism-axum = "0.1"
```

## Testing your own code against this

Register a `FakeProvider` into the `Registry` you pass to `routes`, and
drive requests through `tower::ServiceExt::oneshot` -- no real network
socket, no API key:

```rust
use axum::body::Body;
use axum::http::Request;
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::Registry;
use tower::ServiceExt;

# #[tokio::main]
# async fn main() {
let fake = FakeProvider::new("openai").respond_with(FakeTextResponse::new("Hello!"));
let mut registry = Registry::new();
registry.register("openai", fake);

let app = llmprism_axum::routes(registry);
let body = serde_json::json!({
    "provider": "openai",
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": [{"Text": "hi"}]}],
});

let response = app
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/text")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
    )
    .await
    .unwrap();

assert_eq!(response.status(), 200);
# }
```

See [`tests/routes.rs`](tests/routes.rs) in this crate for the full set of
route/error/streaming tests written this way.

## Part of a bigger picture

This crate builds on `llmprism`'s [framework-integration
roadmap](../../ROADMAP.md): Phase 1 (this crate's routes), Phase 2
(`llmprism::persistence`), and Phase 3 (`llmprism::tenancy`, and this
crate's `routes_multi_tenant`) are all built once on the
framework-agnostic `ProviderMiddleware` seam in core, rather than
reimplemented per adapter -- the next framework adapter (Actix-web,
Rocket, ...) gets the same persistence and multi-tenancy support for free.

## License

MIT -- see [LICENSE](../../LICENSE).
