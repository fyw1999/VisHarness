#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$ENVIRONMENT_DIR/.." && pwd)"
PATCH="$ENVIRONMENT_DIR/patches/verl-vllm024.patch"
APPLIED=false

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not available on PATH." >&2
    exit 1
fi
if [[ ! -f "$PROJECT_ROOT/verl/setup.py" ]]; then
    echo "Error: initialize verl first: git submodule update --init --recursive" >&2
    exit 1
fi

restore_dependencies() {
    if [[ "$APPLIED" == true ]]; then
        git -C "$PROJECT_ROOT/verl" apply --reverse "$PATCH"
    fi
}
trap restore_dependencies EXIT

# Patch package metadata only during installation; leave the upstream checkout
# and all training implementation unchanged. Never reset unrelated user edits.
if ! git -C "$PROJECT_ROOT/verl" apply --reverse --check "$PATCH" 2>/dev/null; then
    git -C "$PROJECT_ROOT/verl" apply --check "$PATCH"
    git -C "$PROJECT_ROOT/verl" apply "$PATCH"
    APPLIED=true
fi

cd -- "$PROJECT_ROOT"
conda env create --file "$ENVIRONMENT_DIR/VisHarness.yml" "$@"
