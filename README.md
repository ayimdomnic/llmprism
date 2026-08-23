# llmprism

[![Crates.io](https://img.shields.io/crates/v/llmprism.svg)](https://crates.io/crates/llmprism)
[![docs.rs](https://img.shields.io/docsrs/llmprism)](https://docs.rs/llmprism)
[![CI](https://github.com/ayimdomnic/llmprism/actions/workflows/ci.yml/badge.svg)](https://github.com/ayimdomnic/llmprism/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`llmprism` is a Rust library for talking to Large Language Model providers -- 13 of
them, from OpenAI and Anthropic to Gemini, Ollama, and more -- through one
consistent, fluent API instead of learning each provider's own SDK and JSON shape.
It's a Rust port of the PHP/Laravel library [Prism](https://prismphp.com), rebuilt
around Rust's async ecosystem and type system.

```rust,no_run
use llmprism::Registry;

#[tokio::main]
async fn main() -> Result<(), llmprism::Error> {
    // Picks up OPENAI_API_KEY / ANTHROPIC_API_KEY from the environment.
    let registry = Registry::from_env();

    let response = registry
        .text("openai", "gpt-4o-mini")?
        .with_prompt("Say hello in one word.")
        .generate()
        .await?;

    println!("{}", response.text.unwrap_or_default());
    Ok(())
}
```

Swap `"openai"` for `"anthropic"` and the rest of the code doesn't change -- that's
the whole point. Call `.stream()` instead of `.generate()` to get the reply
incrementally as a stream of `StreamEvent`s rather than waiting for the whole
thing -- tool calling works exactly the same way either way.

## Status

This crate is early (`0.1.0`, pre-1.0) and the public API may still change, but
every provider and capability in the original scope is implemented. Not every
provider supports every capability -- a provider's own API has to actually offer
it -- so here's what works where:

| Provider                                                     | Text + tools + streaming | Structured output | Moderation | Embeddings | Images | Audio |
| -------------------------------------------------------------| :-----------------------: | :----------------: | :--------: | :--------: | :----: | :---: |
| **OpenAI**                                                   | ✅                         | ✅                  | ✅          | ✅          | ✅      | ✅     |
| **Anthropic**                                                | ✅                         | ✅                  |            |            |        |       |
| **Gemini**                                                   | ✅                         | ✅                  |            | ✅          |        |       |
| **Groq**, **DeepSeek**, **Mistral**, **xAI**, **OpenRouter**, **Perplexity**, **Z.ai** | ✅ |     |            |            |        |       |
| **Ollama**                                                   | ✅                         |                     |            | ✅          |        |       |
| **VoyageAI**                                                 |                            |                     |            | ✅          |        |       |
| **ElevenLabs**                                                |                            |                     |            |            |        | ✅     |

A few notes on the gaps:

- Anthropic has no moderation, embeddings, or image endpoints to call at
  all; Gemini has no moderation or image endpoints. Calling one of those
  methods on a provider that doesn't have it returns `Error::Unsupported`
  rather than failing to compile.
- Groq, DeepSeek, Mistral, xAI, OpenRouter, Perplexity, and Z.ai are thin
  wrappers around the OpenAI provider pointed at each vendor's own
  OpenAI-compatible endpoint. Text generation (including tools and streaming)
  is the one capability guaranteed to actually be compatible across all of
  them -- see the `providers::openai_compatible` module docs (`cargo doc
  --open`) for why the rest are deliberately not wired up for this family.
- Ollama also reuses the OpenAI provider (plus embeddings), needs no API key
  by default, and -- unlike every other provider -- is registered by
  `Registry::from_env()` unconditionally rather than only when a key is set.
- VoyageAI and ElevenLabs are single-capability specialists: VoyageAI does
  only embeddings (frequently paired with Anthropic, which has none of its
  own), and ElevenLabs does only text-to-speech and speech-to-text.

## Installing

```toml
[dependencies]
llmprism = { version = "0.1", features = ["openai", "anthropic"] }
```

Nothing is enabled by default. Every provider lives behind its own Cargo feature
flag -- `openai`, `anthropic`, `groq`, `deepseek`, `mistral`, `xai`,
`openrouter`, `perplexity`, `zai`, `ollama`, `voyageai`, `elevenlabs`,
`gemini` -- so
your binary only pulls in the HTTP client and pays the compile-time cost for
the providers you actually use. Turn on everything with the `full` feature.

## How it's organized

- **`Registry`** is where you register providers (or let `Registry::from_env()` do
  it for you) and it's the one place application code asks for a provider by name.
  In tests you register a `FakeProvider` under the same name instead -- no other
  code has to change.
- Every capability -- **Text**, **structured output**, **moderation**,
  **embeddings**, **images**, and **audio** (text-to-speech and
  speech-to-text) -- follows the same shape: call a method on the registry to
  get a fluent builder, chain `.with_*()` calls to describe the request, then
  run it -- `.generate()` (all at once) or `.stream()` (incrementally, as a
  `Stream` of events) for Text, `.generate()` for the others. Not every
  provider implements every capability; calling one that doesn't returns
  `Error::Unsupported` rather than failing to compile.
- **Tools** are anything implementing the `Tool` trait. Give a request a list of
  tools and `llmprism` handles the whole back-and-forth automatically: if the model
  asks to call a tool, the tool runs, its result is sent back to the model, and this
  repeats (up to `with_max_steps`) until the model produces a final answer.
- **`Schema`** describes the shape of a tool's arguments -- or, for
  `Registry::structured`, the exact shape a reply must match -- in a way every
  provider understands, without you hand-writing each provider's particular JSON
  Schema dialect. OpenAI and Anthropic get there differently under the hood (a
  native enforced response format vs. a forced tool call); you get the same
  `StructuredResponse` back either way.

Run `cargo doc --open --all-features` for the full reference -- every public type
has a plain-language explanation of what it's for and, where it helps, a short
example.

## Examples

The [`examples/`](examples) directory has five runnable examples: text
generation, streaming, tool calling, structured output, and testing your own
code with `FakeProvider` (the only one that needs no API key -- try that one
first):

```sh
cargo run --example testing_with_fake_provider

OPENAI_API_KEY=sk-... cargo run --example text_generation --features openai
OPENAI_API_KEY=sk-... cargo run --example streaming --features openai
OPENAI_API_KEY=sk-... cargo run --example tool_calling --features openai
OPENAI_API_KEY=sk-... cargo run --example structured_output --features openai
```

The four gated examples target OpenAI specifically, but nothing about them is
OpenAI-specific -- swap the provider name and Cargo feature and they work the
same way against any other registered provider.

## Testing your own code against this crate

You don't need real API keys to test code that uses `llmprism`. Register a
`FakeProvider` with scripted responses instead of a real provider:

```rust
use llmprism::testing::{FakeProvider, FakeTextResponse};
use llmprism::Registry;

# #[tokio::main]
# async fn main() {
let fake = FakeProvider::new("openai").respond_with(FakeTextResponse::new("Hello!"));

let mut registry = Registry::new();
registry.register("openai", fake);

let response = registry.text("openai", "gpt-4o-mini").unwrap()
    .with_prompt("hi")
    .generate()
    .await
    .unwrap();

assert_eq!(response.text.as_deref(), Some("Hello!"));
# }
```

## Development

```sh
cargo test                              # core crate + tool-calling loop, no network access
cargo test --features full              # also builds every provider
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
cargo build --examples --features full
```

Every provider has its own live smoke test under `tests/` (`openai_text.rs`,
`anthropic_text.rs`, `gemini.rs`, and so on) that only runs when that
provider's API key is set in your environment; without it, the test prints a
note and passes trivially, so CI needs no secrets. `.github/workflows/ci.yml`
also builds every provider feature completely on its own
(`--no-default-features --features X`, one job per provider) -- `full` alone
isn't enough to catch a feature-gated item that only compiles because some
*other* enabled feature happens to pull in what it needs.

Before publishing a new version, `cargo publish --dry-run` packages and
compiles the crate exactly as crates.io would, without actually uploading
anything. See [`scripts/release.sh`](scripts/release.sh) for the actual
release steps, and [CHANGELOG.md](CHANGELOG.md) for what's shipped in each
version.

## License

MIT -- see [LICENSE](LICENSE).
