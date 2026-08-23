# llmprism

`llmprism` is a Rust library for talking to Large Language Model providers (OpenAI,
Anthropic, and more to come) through one consistent, fluent API instead of learning
each provider's own SDK and JSON shape. It's a Rust port of the PHP/Laravel library
[Prism](https://prismphp.com), rebuilt around Rust's async ecosystem and type system.

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

This crate is early and under active development. Text generation -- including
multi-step tool calling, streaming or not -- and structured output both work end
to end for **OpenAI** and **Anthropic**. Embeddings, images, audio, moderation,
and the remaining ten providers from the original Prism library are on the
roadmap but not implemented yet -- see the module-level docs (`cargo doc --open`)
for what exists today. The public API may still change as more capabilities land.

## Installing

```toml
[dependencies]
llmprism = { version = "0.1", features = ["openai", "anthropic"] }
```

Nothing is enabled by default. Every provider lives behind its own Cargo feature
flag, so your binary only pulls in the HTTP client and pays the compile-time cost
for the providers you actually use. Turn on everything with the `full` feature.

## How it's organized

- **`Registry`** is where you register providers (or let `Registry::from_env()` do
  it for you) and it's the one place application code asks for a provider by name.
  In tests you register a `FakeProvider` under the same name instead -- no other
  code has to change.
- Every capability (**Text** and **structured output**, so far) follows the same
  shape: call a method on the registry to get a fluent builder, chain `.with_*()`
  calls to describe the request, then run it -- `.generate()` (all at once) or
  `.stream()` (incrementally, as a `Stream` of events) for Text, `.generate()`
  for structured output.
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
cargo test                    # core crate + tool-calling loop, no network access
cargo test --features full    # also builds the OpenAI/Anthropic providers
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
```

The live smoke tests in `tests/openai_text.rs` and `tests/anthropic_text.rs` only
run when `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are set in your environment; without
them, they print a note and pass trivially, so CI needs no secrets.

## License

MIT -- see [LICENSE](LICENSE).
