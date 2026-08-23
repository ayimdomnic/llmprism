---
name: Bug report
about: Something doesn't work the way this crate's docs say it should
title: ''
labels: bug
assignees: ''
---

**Provider and feature(s) enabled**
e.g. `anthropic`, or `full`.

**What happened**
A clear description of the actual behavior -- an error message, a panic, a
wrong value, a response that doesn't match what the provider's own API
returned.

**What you expected**
What you expected to happen instead, and why (a link to the relevant
provider API doc helps if this looks like a wire-format mismatch).

**Minimal reproduction**
The smallest code that reproduces this -- ideally using
[`FakeProvider`](https://docs.rs/llmprism/latest/llmprism/testing/struct.FakeProvider.html)
if the bug is about request/response mapping rather than something that only
shows up against the real API.

```rust
// your code here
```

**Environment**
- `llmprism` version:
- Rust version (`rustc --version`):
- OS:
