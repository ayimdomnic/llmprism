# Contributing to llmprism

Thanks for considering a contribution. This document covers how to get set
up, what's expected of a change, and how to add a new provider -- the most
common kind of substantial contribution this crate gets.

## Getting set up

You need a stable Rust toolchain (see `rust-toolchain.toml` for the exact
channel) and nothing else -- no external services, no Docker, no API keys.

```sh
git clone https://github.com/ayimdomnic/llmprism
cd llmprism
cargo test                    # core crate, no network access, no features needed
cargo test --features full    # every provider
```

## Before opening a pull request

Run the same checks CI runs:

```sh
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --features full
cargo test                                              # default (no) features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps           # see the note below
```

The second `cargo doc` line matters more than it looks: an intra-doc link in
a doc comment that's always compiled (the crate root, `providers/mod.rs`) but
points at a Cargo-feature-gated item will resolve fine under `--all-features`
and then break for anyone building with fewer features enabled. This has
happened more than once in this crate's history -- see the git log for
`fix: broken intra-doc links` if you want the full story.

If you're adding or changing a provider, also build (and ideally doc-check)
that feature on its own:

```sh
cargo build --no-default-features --features <your-provider>
RUSTDOCFLAGS="-D warnings" cargo doc --no-default-features --features <your-provider> --no-deps
```

`full` alone won't catch a feature-gated item that only compiles because some
*other* enabled feature happens to pull in what it needs.

## Commit messages

This repo uses [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, with an optional
scope like `feat(anthropic): ...`), and commit bodies that explain *why*, not
just *what* -- especially for anything where a wire format, a provider quirk,
or a design tradeoff was involved. `git log` in this repo is full of examples;
skim a few recent ones (`git log -20`) to get a feel for the level of detail
expected. A one-line "fix bug" commit is fine for something genuinely trivial;
anything that involved research, a real design decision, or a non-obvious
constraint deserves the reasoning written down, because that reasoning is
what saves the next person (quite possibly you) from re-deriving it or
re-breaking it.

## Adding a new provider

Every provider follows the same shape. The fastest way to add one is to read
an existing provider close to what you're building (`crates/llmprism/src/providers/gemini/`
is a good example of a fully native wire format; `crates/llmprism/src/providers/ollama.rs`
is a good example of reusing another provider's request/response handling)
and mirror its structure:

1. **`crates/llmprism/src/providers/<name>/mod.rs`** -- the `<Name>Provider` struct
   (`api_key`, `base_url`, an HTTP client) and its `impl Provider`, one method
   per capability the provider's real API actually supports. Every method has
   a default `Err(Error::Unsupported)` body on the `Provider` trait, so you
   only implement what genuinely exists -- don't stub out capabilities the
   provider doesn't have.
2. **`crates/llmprism/src/providers/<name>/wire.rs`** -- `serde` structs mirroring the
   provider's actual JSON wire format, one file per direction if request and
   response shapes are unrelated enough to make that clearer.
3. **`crates/llmprism/src/providers/<name>/maps.rs`** -- pure functions translating between
   this crate's provider-agnostic types (`TextRequest`, `Step`, `Message`,
   ...) and the wire types from step 2. Keep these free of I/O so they're
   trivially unit-testable.
4. **`Cargo.toml`** -- a new feature flag. A provider with its own wire
   format only needs `<name> = ["http"]`; a provider that reuses another
   provider's request handling (the way the OpenAI-compatible family and
   Ollama reuse `OpenAiProvider`) depends on that provider's feature instead.
   Add the new feature to `full` too.
5. **`crates/llmprism/src/providers/mod.rs`, `crates/llmprism/src/lib.rs`** -- the `#[cfg(feature = ...)]`
   -gated `pub mod`, and the new feature name added to the crate-root doc
   comment's feature list and the `#[cfg(any(...))]` gate on `pub mod
   providers`.
6. **`crates/llmprism/src/registry.rs`** -- a `Registry::from_env()` block registering the
   provider from its own `<NAME>_API_KEY` env var (unless it needs no API key
   at all, like Ollama -- see that provider's block for the exception
   pattern), plus the env var added to that method's doc comment.
7. **Unit tests** for the mapping functions in step 3 -- these need no
   network access and are what catches most real bugs (wrong field name,
   wrong default, a response shape you didn't handle).
8. **`crates/llmprism/tests/<name>.rs`** -- a live smoke test gated on the provider's real
   API key being set in the environment, printing a note and returning early
   if it isn't. This is what actually confirms the wire format is right
   against the real API; the unit tests only confirm internal consistency.

**Verify wire formats against the provider's own current documentation, not
memory or assumption.** Getting an endpoint path, a field name, or an auth
header wrong ships a provider that's silently non-functional (or worse,
silently wrong) for everyone who enables it. If you're not confident about a
detail, say so in the PR description rather than guessing.

## Scope decisions

Not every capability makes sense for every provider, and this crate would
rather leave something out with a documented reason than guess at a wire
shape or claim support that isn't actually verified. If you're adding a
capability to an existing provider and it's genuinely not supported by that
provider's real API, don't stub it in -- rely on `Provider`'s default
`Err(Error::Unsupported)` and leave a short comment explaining why, the same
way `crates/llmprism/src/providers/openai_compatible.rs`'s module doc explains why that
family is Text-only.

## License

By contributing, you agree your contribution is licensed under this
project's [MIT license](LICENSE).
