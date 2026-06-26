#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec uv run python -m xiaobai_app_pack.launchers.official_app_launcher start "$@"
