# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/);
`git log` is the authoritative detail behind every entry below.

## [Unreleased]

### Added

- **Tool-call approval**: `Tool::needs_approval` (default `false`) marks a
  tool as needing an external decision before it runs; `ApprovalHandler`
  (attach with `PendingTextRequest::with_approval_handler`) makes that
  decision, async, so it's free to await a database poll, a channel, or a
  webhook callback for as long as the request stays open -- no pause/resume
  machinery needed. A denied call, or one with no handler attached at all,
  never reaches `Tool::call`; the model sees a normal tool-error result
  (`ToolError::ApprovalDenied`) instead, the same as any other tool failure.
  Deliberately doesn't attempt cross-process resumption (a signed token
  round-tripping through a separate request later) -- a real, harder
  problem left to an `ApprovalHandler` implementation to build if needed,
  not something this crate takes on.
- **Streaming structured output**: `PendingStructuredRequest::stream()` and
  `Provider::stream_structured_once`, yielding `StructuredStreamEvent::PartialObject`
  as the model's JSON reply arrives (repaired into valid JSON via the
  `partial-json-fixer` crate after every chunk) and ending with exactly one
  `StructuredStreamEvent::End`. Implemented for OpenAI, whose
  `response_format: json_schema` mode streams identically to a plain chat
  completion; Anthropic support (a different streaming shape, via
  `input_json_delta` events on a forced tool call) is a documented gap, not
  built yet. `FakeProvider` also gained `stream_structured_once`, reusing the
  existing `respond_with_structured` fixture.

### Changed

- The repo is now a Cargo workspace (`ROADMAP.md`'s Phase 0): this crate
  lives at `crates/llmprism/` rather than the repo root, with room for
  framework-adapter crates (`llmprism-axum` and friends) to be published
  separately later without forcing their dependencies onto everyone. The
  published package itself (name, version history, public API) is
  unaffected -- `cargo add llmprism` and everything already written against
  it keeps working exactly as before. One small, accepted packaging change:
  the literal `LICENSE` file is no longer bundled inside the published
  `.crate` archive (Cargo's automatic license-file discovery only looks in
  the same directory as `Cargo.toml`, and adding `license-file` back
  explicitly to work around that is what Cargo itself warns against once
  `license = "MIT"` is already set) -- the SPDX `license = "MIT"` field
  crates.io reads is untouched, and the real file is still right there in
  the repository for anyone who wants to read it.

## [0.2.0] - 2026-08-26

Borrowed from a comparison against Vercel's AI SDK, scoped to what matters
for server-side Rust applications. Also adds benchmark coverage, a CLI, MCP
client support, OpenTelemetry-style tracing, and a framework-integration
roadmap, each unrelated to that comparison.

### Added

- **CLI** (`cli` feature, new `llmprism` binary): every `Registry`
  capability as a subcommand (`text`, `stream`, `structured`, `moderate`,
  `embed`, `rerank`, `image`, `speak`, `transcribe`, `providers`). A thin
  wrapper around `Registry::from_env()` -- no provider-specific logic of its
  own. `--json` for machine-readable output; every text-taking flag falls
  back to stdin when omitted. With `mcp` also enabled, `text`/`stream` gain
  `--mcp-stdio`/`--mcp-http` flags. See the README's "Command-line usage"
  section.
- **MCP client support** (`mcp` feature): `McpToolset::connect_stdio`/
  `connect_http` discover a remote MCP server's tools and hand them back as
  ordinary `Tool`s, ready for `with_tool`/`with_tools` -- no
  protocol-specific code needed in application code. Built on the official
  `rmcp` SDK; verified against a real server (the MCP "everything"
  reference server) in `tests/mcp.rs`, not just compiled.
- **OpenTelemetry-style tracing** (`tracing` feature): `GenAiTracingMiddleware`
  instruments every `Provider` call with a `tracing` span following the
  current OpenTelemetry GenAI semantic conventions (`gen_ai.*`). Depends
  only on `tracing`, not `opentelemetry` directly -- bridge to a real OTel
  backend from your own application with `tracing-opentelemetry`.
- **`ObjectSchema::from_raw_json_schema`**: send an already-formed JSON
  Schema document as-is for a tool's parameters or a structured request's
  schema, bypassing the `properties`/`required` builder entirely. Shared
  foundation for both MCP (whose tool schemas arrive this way already) and
  the CLI's `structured --schema-file`.
- `ROADMAP.md`: a staged plan for framework integration (Axum first, then
  persistence and multi-tenancy built on the existing `ProviderMiddleware`
  seam, then additional frameworks) -- linked from the README.
- `Registry::provider_names()`, for diagnosing what `from_env()` actually
  found configured.
- `Serialize`/`Deserialize` on `AudioOutput`/`AudioResponse`/
  `TranscriptionResponse` -- every other response type already had these;
  a pre-existing inconsistency the CLI's `--json` output surfaced.
- `criterion` benchmarks (`cargo bench --features full`) for the crate's own
  hot paths -- schema-to-JSON-Schema conversion, the tool-calling loop's
  bookkeeping, `Registry::wrap` middleware dispatch overhead, and the
  `provider_options` merge -- isolated from real provider network latency via
  `FakeProvider`, so they measure code this crate actually owns.

- **`ProviderMiddleware`** and `Registry::wrap`: wrap a registered provider to
  intercept its calls from outside its own implementation -- transform a
  request, transform or replace a response, or skip the call entirely (a
  cache hit, a policy rejection). Middlewares compose.
- **`StopCondition`** and `TextRequest::stop_when`: end the tool-calling loop
  early for a reason other than `max_steps`, in both the non-streaming and
  streaming loops.
- **`RepairStrategy`** and `StructuredRequest::with_repair`: salvage a
  structured-output reply that fails to decode (a stray Markdown code fence,
  a model that didn't call the forced tool) instead of failing outright.
  `Error::StructuredDecode` now carries the raw response text and what the
  response's finish reason/usage/metadata would have been, boxed into a new
  `StructuredDecodeContext` to avoid inflating every other `Result<_, Error>`
  in the crate.
- **Rerank capability** (VoyageAI): score and sort documents by relevance to
  a query, via `Registry::rerank`.
- **Configurable HTTP retry policy**: `client::build_http_client_with_max_retries`
  and `client::build_http_client_with_retry_strategy` replace the old
  fixed-at-2-retries default. Along the way, fixed a stale doc claim that
  `429`/`529` weren't retried by default -- `reqwest-retry`'s
  `DefaultRetryableStrategy` (used unmodified by every provider's default
  client all along) already retries both.
- **`TextRequest`/`StructuredRequest`** gained typed `stop_sequences`/`seed`
  fields (`stop_sequences` on `TextRequest` only -- a stop string can
  truncate otherwise-valid JSON, which cuts against the point of a
  structured request).

### Changed

- **Breaking, if you implement `Provider` yourself**: `Provider::text_step`
  and `Provider::stream_text_once` now take `&TextRequest` instead of an
  owned `TextRequest`. The non-streaming and streaming tool-calling loops
  call one of these once per round trip against a conversation that keeps
  growing (each round trip appends new messages), so taking ownership
  forced cloning the entire, ever-larger request on every single round
  trip -- no provider implementation ever actually needed to *consume* the
  request, only read it. Update a custom `Provider` by changing the
  parameter type and, if you were already just borrowing internally
  (`build_request(&request)`), removing the now-redundant `&`.

## [0.1.1] - 2026-08-24

No public API changes -- release hygiene, testing, and documentation only.

### Added

- `rust-version = "1.88"` declared in `Cargo.toml`, verified by actually
  building against that exact toolchain (the floor comes from transitive
  dependencies, not this crate's own code).
- HTTP error-path integration tests (429/413/529/generic/malformed-body
  handling) against a mocked server, closing a real gap where the full
  request-to-typed-`Error` path was only ever verified by inspection.
- CI now also checks the declared MSRV and runs a security-advisory scan on
  every push/PR.
- `CONTRIBUTING.md` and GitHub issue templates.

### Changed

- The CI badge is temporarily removed from the README (Actions runs are
  currently blocked at the account level; a red "failing" badge would
  misrepresent the actual state of the workflow).

## [0.1.0] - 2026-08-23

Initial release. `llmprism` is a Rust port of the PHP/Laravel library
[Prism](https://prismphp.com), unifying 13 LLM providers behind one
consistent, fluent API.

### Added

- **13 providers**, each behind its own Cargo feature flag: OpenAI, Anthropic,
  Gemini, Groq, DeepSeek, Mistral, xAI, OpenRouter, Perplexity, Z.ai, Ollama,
  VoyageAI, and ElevenLabs.
- **Text generation**, including multi-step tool calling, for OpenAI,
  Anthropic, Gemini, and the OpenAI-compatible provider family (Groq,
  DeepSeek, Mistral, xAI, OpenRouter, Perplexity, Z.ai, Ollama).
- **Streaming** text generation, yielding incremental `StreamEvent`s instead
  of waiting for the full reply, with the same multi-step tool-calling
  behavior as non-streaming requests.
- **Structured output**, using each provider's own strategy under the hood
  (OpenAI and Gemini's native JSON Schema enforcement, Anthropic's
  forced-tool-call approach) behind one consistent `StructuredResponse` type.
- **Moderation**, **embeddings** (with optional output-dimension control),
  **image generation** (with quality/style options), and **audio**
  (text-to-speech and speech-to-text) for the providers that support them.
- **Multimodal message content**: image and document attachments for OpenAI
  and Anthropic; image, document, audio, and video for Gemini.
- **Provider-native tools** (built-in, server-side capabilities like web
  search) for Anthropic and Gemini, alongside this crate's own user-defined
  `Tool` trait.
- **Anthropic extended thinking** and **OpenAI reasoning effort** support for
  reasoning-capable models.
- **Anthropic prompt caching** for the system prompt.
- A `provider_options` escape hatch on every request type for provider
  -specific fields this crate doesn't model directly, and a `with_client`
  escape hatch on every provider for HTTP-level configuration (timeouts,
  proxies, custom retry policies).
- `Registry` and `FakeProvider` testing infrastructure, so application code
  can be tested without real API keys or network access.
- Five runnable examples (`examples/`) and a full CI matrix covering every
  provider feature individually, formatting, linting, and documentation
  builds.

[Unreleased]: https://github.com/ayimdomnic/llmprism/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ayimdomnic/llmprism/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ayimdomnic/llmprism/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ayimdomnic/llmprism/releases/tag/v0.1.0
