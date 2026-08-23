#!/usr/bin/env bash
#
# Cuts a release of llmprism: verifies the tree is clean and green, bumps
# Cargo.toml's version, commits, tags, pushes, publishes to crates.io, and
# creates a GitHub release. Deliberately manual, not CI-triggered -- run it
# yourself when you're ready to ship a release, after CHANGELOG.md already
# has an entry for the version you're releasing (this script does not write
# the changelog for you).
#
# Usage: scripts/release.sh <version>   (e.g. scripts/release.sh 0.2.0)

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <version>  (e.g. $0 0.2.0)" >&2
    exit 1
fi

VERSION="$1"
TAG="v${VERSION}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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

echo "==> Checking CHANGELOG.md has an entry for ${VERSION}"
if ! grep -q "^## \[${VERSION}\]" CHANGELOG.md; then
    echo "error: CHANGELOG.md has no '## [${VERSION}]' section yet -- write it first" >&2
    exit 1
fi

echo "==> Checking tag ${TAG} doesn't already exist"
if git rev-parse "${TAG}" >/dev/null 2>&1; then
    echo "error: tag ${TAG} already exists" >&2
    exit 1
fi

echo "==> Running the full verification suite (this takes a minute)"
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
cargo build --features full
cargo test --features full
cargo test
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

echo "==> Verifying the package itself builds cleanly (cargo publish --dry-run)"
cargo publish --dry-run

echo "==> Bumping Cargo.toml to ${VERSION}"
sed -i.bak "0,/^version = \".*\"/s//version = \"${VERSION}\"/" Cargo.toml
rm -f Cargo.toml.bak

echo "==> Committing the version bump"
git add Cargo.toml CHANGELOG.md
git commit -m "chore(release): ${TAG}"

echo "==> Tagging ${TAG}"
git tag -a "${TAG}" -m "${TAG}"

confirm "About to push '${TAG}' and the release commit to origin -- continue?"
git push origin HEAD
git push origin "${TAG}"

confirm "About to run 'cargo publish' for real -- this is IRREVERSIBLE (a version can never be reused, even if yanked). Continue?"
cargo publish

if command -v gh >/dev/null 2>&1; then
    echo "==> Creating a GitHub release from CHANGELOG.md"
    # Extracts the section between this version's heading and the next "## "
    # heading, for use as the release notes.
    NOTES="$(awk "/^## \[${VERSION}\]/{flag=1; next} /^## \[/{flag=0} flag" CHANGELOG.md)"
    gh release create "${TAG}" --title "${TAG}" --notes "${NOTES}"
else
    echo "==> gh CLI not found -- skipping GitHub release creation."
    echo "    Create one manually at: https://github.com/ayimdomnic/llmprism/releases/new?tag=${TAG}"
fi

echo "==> Done. Released ${TAG}."
