#!/usr/bin/env bash
#
# Cuts a release of one crate in this workspace: verifies the tree is clean
# and green, bumps crates/<crate>/Cargo.toml's version, commits, tags,
# pushes, publishes to crates.io, and creates a GitHub release. Deliberately
# manual, not CI-triggered -- run it yourself when you're ready to ship a
# release, after CHANGELOG.md already has an entry for the version you're
# releasing (this script does not write the changelog for you).
#
# Each crate in this workspace is versioned and released independently --
# llmprism and llmprism-axum are at different versions today, and that's
# expected, not a bug to fix. `llmprism` (the original, single-package
# crate) keeps its historical, unprefixed tag and CHANGELOG heading format
# for continuity (`vX.Y.Z`, `## [X.Y.Z]`); every other crate gets its name
# in both (`<crate>-vX.Y.Z`, `## [<crate> X.Y.Z]`), so two crates released
# at the same version number don't collide in the shared CHANGELOG.md or in
# the repo's tag namespace.
#
# Usage: scripts/release.sh <crate> <version>
#   e.g. scripts/release.sh llmprism 0.3.0
#        scripts/release.sh llmprism-axum 0.1.0

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <crate> <version>  (e.g. $0 llmprism 0.3.0)" >&2
    exit 1
fi

CRATE="$1"
VERSION="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CRATE_MANIFEST="crates/${CRATE}/Cargo.toml"
if [[ ! -f "$CRATE_MANIFEST" ]]; then
    echo "error: no ${CRATE_MANIFEST} -- is '${CRATE}' a workspace member? (see crates/ for the list)" >&2
    exit 1
fi

if [[ "$CRATE" == "llmprism" ]]; then
    TAG="v${VERSION}"
    CHANGELOG_HEADING="## [${VERSION}]"
else
    TAG="${CRATE}-v${VERSION}"
    CHANGELOG_HEADING="## [${CRATE} ${VERSION}]"
fi

confirm() {
    read -r -p "$1 [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
}

echo "==> Checking the working tree is clean"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "error: working tree has uncommitted changes -- commit or stash them first" >&2
    git status --short
    exit 1
fi

echo "==> Checking CHANGELOG.md has an entry for ${CRATE} ${VERSION}"
if ! grep -qF "${CHANGELOG_HEADING}" CHANGELOG.md; then
    echo "error: CHANGELOG.md has no '${CHANGELOG_HEADING}' section yet -- write it first" >&2
    exit 1
fi

echo "==> Checking tag ${TAG} doesn't already exist"
if git rev-parse "${TAG}" >/dev/null 2>&1; then
    echo "error: tag ${TAG} already exists" >&2
    exit 1
fi

echo "==> Running the full verification suite across the workspace (this takes a minute)"
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo build --workspace --features full
cargo test --workspace --features full
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

echo "==> Verifying ${CRATE} itself packages cleanly (cargo publish --dry-run)"
cargo publish --dry-run -p "${CRATE}"

echo "==> Bumping ${CRATE_MANIFEST} to ${VERSION}"
sed -i.bak "0,/^version = \".*\"/s//version = \"${VERSION}\"/" "$CRATE_MANIFEST"
rm -f "${CRATE_MANIFEST}.bak"

git add "$CRATE_MANIFEST" CHANGELOG.md
if git diff --cached --quiet; then
    # Cargo.toml already had this version (e.g. cutting the very first
    # release, where nothing needs bumping) -- nothing to commit, just tag
    # the current HEAD as-is.
    echo "==> ${CRATE_MANIFEST} is already at ${VERSION}; nothing to commit"
else
    echo "==> Committing the version bump"
    git commit -m "chore(release): ${TAG}"
fi

echo "==> Tagging ${TAG}"
git tag -a "${TAG}" -m "${TAG}"

confirm "About to push '${TAG}' and the release commit to origin -- continue?"
git push origin HEAD
git push origin "${TAG}"

confirm "About to run 'cargo publish -p ${CRATE}' for real -- this is IRREVERSIBLE (a version can never be reused, even if yanked). Continue?"
cargo publish -p "${CRATE}"

if command -v gh >/dev/null 2>&1; then
    echo "==> Creating a GitHub release from CHANGELOG.md"
    # Extracts the section between this heading and the next "## " heading,
    # for use as the release notes.
    NOTES="$(awk -v heading="${CHANGELOG_HEADING}" '
        index($0, heading) == 1 { flag=1; next }
        /^## \[/ { flag=0 }
        flag
    ' CHANGELOG.md)"
    gh release create "${TAG}" --title "${TAG}" --notes "${NOTES}"
else
    echo "==> gh CLI not found -- skipping GitHub release creation."
    echo "    Create one manually at: https://github.com/ayimdomnic/llmprism/releases/new?tag=${TAG}"
fi

echo "==> Done. Released ${CRATE} ${TAG}."
