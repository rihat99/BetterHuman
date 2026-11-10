#!/usr/bin/env bash
# The release tier: every check of the repository, the CUDA tests, the CUDA counters and the licensed tests included,
# stopping at the first failure. Run by hand before a tag, from any directory:
#     bash tools/release_tier.sh --gpu <index>
# <index> is the nvidia-smi (PCI bus) index of a free GPU; the run fails before its first test when torch cannot use
# it. The cuda and licensed tests read the weights under $BETTERHUMAN_MODELS_DIR. docs/guides/run-the-tests.md lists
# the three tiers.
set -euo pipefail

if [[ $# -ne 2 || $1 != --gpu ]]; then
    echo "usage: bash tools/release_tier.sh --gpu <index>" >&2
    exit 2
fi
gpu=$2
cd "$(dirname "$0")/.."

step() {
    echo "== $*"
    "$@"
}
on_gpu() {
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

step uv lock --check
step on_gpu uv run python -c \
    "import sys, torch; sys.exit(None if torch.cuda.is_available() else 'release_tier.sh: torch cannot use GPU $gpu')"
step uv run ruff check .
step uv run ruff format --check .
step uv run python tools/api_snapshot.py --check
step uv run mkdocs build --strict -d "$(mktemp -d)"
step on_gpu uv run pytest tests -q -m cuda
step on_gpu uv run python benchmarks/run.py --device cuda --counters-only
step uv run pytest tests -q -m "not cuda and not licensed"
# tests/conftest.py leaves BetterRobot's internal checks at their default, off, so the full tier above ran as users
# run it; this run turns them on.
step env BETTER_ROBOT_INTERNAL_CHECKS=1 uv run pytest tests -q -m "not cuda and not licensed"
step uv run pytest tests -q -m "licensed and not cuda"
echo "== release tier passed"
