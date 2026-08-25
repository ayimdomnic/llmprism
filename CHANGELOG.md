# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/);
`git log` is the authoritative detail behind every entry below.

## [Unreleased]

Borrowed from a comparison against Vercel's AI SDK, scoped to what matters
for server-side Rust applications. Also adds benchmark coverage, unrelated to
that comparison.

### Added

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

[Unreleased]: https://github.com/ayimdomnic/llmprism/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ayimdomnic/llmprism/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ayimdomnic/llmprism/releases/tag/v0.1.0
