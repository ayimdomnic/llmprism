# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/);
`git log` is the authoritative detail behind every entry below.

## [Unreleased]

Nothing yet.

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

[Unreleased]: https://github.com/ayimdomnic/llmprism/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ayimdomnic/llmprism/releases/tag/v0.1.0
