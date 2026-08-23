---
name: Feature request
about: A capability, provider, or request field this crate doesn't support yet
title: ''
labels: enhancement
assignees: ''
---

**What's missing**
A clear description of the capability, provider, or field you need.

**Provider documentation**
If this is about a specific provider's API, a link to the relevant section
of their docs -- this crate tries hard to verify wire formats against a
provider's own documentation rather than guess, so a link saves whoever
picks this up a research step.

**Why this belongs in the crate itself**
Versus, for example, `provider_options` (every request type's escape hatch
for provider-specific fields this crate doesn't model directly yet) or
`provider_tools` (for provider-native, server-side tools) -- if what you need
already fits through one of those, mention that too; it might already be
possible today without a code change here.
