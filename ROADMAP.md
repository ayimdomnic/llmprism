# Roadmap: integrating llmprism into Rust web frameworks

`llmprism` today is a framework-agnostic library: a `Provider`/`Registry`
core, a `ProviderMiddleware` wrapping layer, and a CLI on top. Nothing about
it assumes Axum, Actix-web, or any other framework -- which is deliberate,
but it also means using it inside a real API server currently means hand
-wiring routes, streaming, conversation persistence, and per-tenant
configuration yourself, every time.

This document lays out how that gets closer to "pull in a crate, minimal
config, you have API functionality" for the frameworks most Rust developers
actually reach for -- without dragging framework dependencies into the core
crate for people who aren't using them.

## Guiding principles

- **Core stays framework-agnostic.** `llmprism` itself never gains a hard
  dependency on Axum, Actix, or anything web-specific. Framework support
  lives in separate, optional crates.
- **Adapters are thin.** Each framework crate's job is wiring, not new
  behavior -- request extraction, route registration, streaming glue. The
  actual capability logic (persistence, auth, multi-tenancy) is built once,
  generically, on top of the `ProviderMiddleware` seam already in core, so
  it isn't reimplemented per framework.
- **Optional at every layer.** A consumer who wants just Axum routing,
  without persistence or multi-tenancy, should be able to take exactly
  that and nothing else.

## Phase 0 — Workspace conversion (prerequisite)

Convert this repo from a single package into a Cargo workspace:
`crates/llmprism` becomes the existing core (unchanged public API, crate
name, and version history), with room for sibling adapter crates.

- **Why first:** an adapter crate that pulls in Axum can't live in the same
  package as core without forcing Axum onto every core consumer. A
  workspace is the only way to publish `llmprism-axum` as its own
  crates.io crate while keeping `llmprism` itself dependency-free of it.
- **Size:** small, mechanical -- move `src/`, `Cargo.toml`, `benches/`,
  `examples/`, `tests/` under `crates/llmprism/`, add a workspace root
  `Cargo.toml`. No code changes.
- **Non-goal:** this phase ships no new functionality by itself.

## Phase 1 — `llmprism-axum`

An Axum router builder that exposes `Registry` capabilities as HTTP
endpoints with minimal setup:

```rust,ignore
let registry = Registry::from_env();
let app = Router::new().merge(llmprism_axum::routes(registry));
```

- **Routes:** one per capability -- `POST /v1/text`, `/v1/structured`,
  `/v1/moderation`, `/v1/embeddings`, `/v1/rerank`, `/v1/images` -- each
  taking its own request DTO with the wire-safe fields `PendingXRequest`
  builds from (the request types themselves can't derive `Deserialize`,
  since several carry trait-object fields like `Vec<Arc<dyn Tool>>` that
  have no JSON representation).
- **Streaming:** separate `/v1/text/stream` and `/v1/structured/stream`
  routes return Server-Sent Events built from the same `StreamEvent`s /
  `StructuredStreamEvent`s the non-HTTP `stream()` methods already yield --
  no new streaming logic, just an SSE encoder over an existing stream.
- **Audio deferred:** `/v1/audio/speech` and `/v1/audio/transcriptions`
  didn't ship in the first pass -- binary request/response bodies need a
  deliberate design choice (base64 in JSON vs. multipart vs. raw bytes with
  a content-type header) that's better made as its own focused follow-up
  than folded into the rest of Phase 1.
- **Tool calling, approval, and MCP stay out of scope for HTTP.** A
  `Tool` is arbitrary server-side code with a `call()` method -- there's no
  wire representation for a client to send one, so this stays something the
  server configures directly against `Registry`/`PendingTextRequest`.
- **Why Axum first:** it's the most-used async web framework in the
  ecosystem right now and composes cleanly via `tower::Service`, which
  keeps the door open for the persistence/auth middleware in later phases
  to be ordinary `tower` layers instead of Axum-specific glue.
- **Size:** medium. The router/handlers are mechanical once the request
  types are `Deserialize`; the SSE encoding is the one genuinely new piece.

## Phase 2 — Persistence

A `ConversationStore` trait -- save and load a conversation's message
history by an opaque id:

```rust,ignore
#[async_trait]
pub trait ConversationStore: Send + Sync {
    async fn load(&self, id: &str) -> Result<Vec<Message>, Error>;
    async fn save(&self, id: &str, messages: &[Message]) -> Result<(), Error>;
}
```

- **Where it plugs in:** `PersistenceMiddleware<S: ConversationStore>`
  implements `ProviderMiddleware` -- loads history before a call, merges it
  with the incoming request's messages, runs the request, saves the
  updated history after. Confirmed the middleware seam added earlier this
  year was enough on its own: no changes to `Provider`, `tool_loop`, or
  `stream_loop` were needed. The one subtlety worth calling out for anyone
  extending this: `Provider::text_step`/`stream_text_once` are called once
  per round trip of a multi-step tool-calling loop, not once per
  `generate()`/`stream()` call, so the *save* specifically has to be gated
  on the round trip that actually ends the call (via `finish_reason`) --
  otherwise a tool-calling conversation would persist once per step instead
  of once per call.
- **What shipped where:** the `ConversationStore` trait and an in-memory
  reference implementation, `InMemoryConversationStore`, shipped in
  `llmprism` core (useful on its own for tests and single-process apps).
  Real backends (Postgres, SQLite, Redis) are still meant to ship as
  separate follow-up crates, each a thin `ConversationStore` impl -- none
  have shipped yet, and which one comes first is demand-driven, not
  decided here.
- **Non-goal:** no commitment yet to which database backends ship first.

## Phase 3 — Auth context & multi-tenancy

Request-scoped identity, threaded through to per-tenant provider/key
resolution and usage tracking -- again via the middleware seam, not new
core plumbing:

- **Auth context:** `llmprism::tenancy::RequestContext` (`tenant_id`,
  `user_id`, free-form `claims`) plus an Axum extractor,
  `llmprism_axum::tenant::TenantContext`, that reads one back out of the
  request's extensions -- the same pattern `tower`/Axum middleware already
  uses for this. This crate never establishes identity itself: an
  application's own auth (a `tower::Layer` or `axum::middleware::from_fn`
  verifying a JWT, session, or API key) is what actually inserts a
  `RequestContext` before these routes run.
- **Multi-tenancy:** `llmprism::tenancy::TenantRegistry` resolves a
  `RequestContext` to the right `Registry` (different API keys, different
  default models, different allowed providers per tenant) instead of one
  process-wide `Registry::from_env()`, with a `StaticTenantRegistry`
  reference implementation for a fixed, startup-time set of tenants.
  `llmprism_axum::routes_multi_tenant` picks the right `Registry` per
  request before dispatching to the same handlers `routes` uses --
  consistent with how persistence hooks in.
- **Usage tracking:** `UsageSink`/`UsageTrackingMiddleware` record token
  usage per tenant after every round trip, for quota enforcement or billing
  -- a natural extension of the same middleware pattern, needing no new
  core plumbing. Unlike persistence, this middleware is constructed once
  per tenant (wrapped into that tenant's own `Registry`, already knowing
  its `tenant_id`) rather than reading one from the request.
- **Non-goal:** this phase doesn't include a hosted quota/billing service --
  just the hooks an application builds one on top of, and it doesn't
  include this crate verifying identity itself (no JWT/session dependency
  was added anywhere).

## Phase 4 — Additional frameworks

Actix-web, Rocket, and Poem/warp adapters, once the Axum pattern (routes +
persistence + auth middleware) is proven out and there's a template to
follow. Lower priority than phases 1-3, and reasonable candidates for
community contribution rather than needing to originate from this
project -- the design work in phases 1-3 is the part that has to happen
first and centrally; a second or third framework adapter is comparatively
mechanical once that pattern exists.

## What this roadmap doesn't promise

- No commitment to a specific release timeline for any phase.
- No commitment to a hosted/managed version of any of this -- everything
  here is a library a developer runs themselves.
- No commitment to which database backends land first in Phase 2, or which
  framework lands first in Phase 4 -- both are open to being driven by
  actual demand rather than decided speculatively here.
