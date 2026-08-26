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
  `/v1/moderation`, `/v1/embeddings`, `/v1/rerank`, `/v1/images`,
  `/v1/audio/speech`, `/v1/audio/transcriptions` -- each accepting roughly
  the same shape `PendingXRequest` builds, since every request type already
  derives (or can derive) `Deserialize`.
- **Streaming:** `/v1/text` with `"stream": true` (or a separate
  `/v1/text/stream` route) returns Server-Sent Events built from the same
  `StreamEvent`s `PendingTextRequest::stream()` already yields -- no new
  streaming logic, just an SSE encoder over an existing stream.
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

- **Where it plugs in:** a `PersistenceMiddleware<S: ConversationStore>`
  implementing `ProviderMiddleware` -- load history before a call, merge it
  with the incoming request's messages, run the request, save the updated
  history after. This is exactly what the middleware seam added earlier
  this year was for: no changes to `Provider`, `tool_loop`, or
  `stream_loop` are needed at all.
- **What ships where:** the `ConversationStore` trait and an in-memory
  reference implementation ship in `llmprism` core (useful on its own for
  tests and single-process apps); real backends (Postgres, SQLite, Redis)
  ship as separate follow-up crates (`llmprism-store-postgres`, etc.), each
  a thin `ConversationStore` impl with no reason to force that database
  driver onto everyone else.
- **Size:** small for the trait and in-memory impl; each backend crate is
  its own small, independent piece of work.
- **Non-goal:** no commitment yet to which database backends ship first --
  that's demand-driven, not decided by this roadmap.

## Phase 3 — Auth context & multi-tenancy

Request-scoped identity, threaded through to per-tenant provider/key
resolution and usage tracking -- again via the middleware seam, not new
core plumbing:

- **Auth context:** an Axum extractor (`llmprism-axum`) that pulls an
  identity (a JWT claim, a session, a header) into a typed
  `RequestContext { tenant_id, user_id, .. }`, made available to
  middleware via request extensions -- the same pattern `tower`/Axum
  middleware already uses for this, not a new mechanism.
- **Multi-tenancy:** a `TenantRegistry` resolving `tenant_id` to the right
  `Registry` (different API keys, different default models, different
  allowed providers per tenant) instead of one process-wide
  `Registry::from_env()`. A `TenantMiddleware` reads the request context and
  picks the right `Registry` before dispatching -- consistent with how
  persistence hooks in.
- **Usage tracking:** a natural extension of the same middleware -- record
  token usage per tenant after each call, for quota enforcement or billing,
  without touching core.
- **Size:** medium -- the extractor and `TenantRegistry` are
  straightforward; getting the ergonomics right (so wiring three
  middlewares together doesn't feel like framework soup) is the real work.
- **Non-goal:** this phase doesn't include a hosted quota/billing service --
  just the hooks an application would build one on top of.

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
