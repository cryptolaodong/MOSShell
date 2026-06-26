#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec uv run python -m xiaobai_app_pack.acceptance.alpha_acceptance run "$@"
