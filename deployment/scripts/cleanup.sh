#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
rm -rf "$ROOT_DIR/backend/data/artifacts"
mkdir -p "$ROOT_DIR/backend/data/artifacts"
echo "Cleaned local artifacts."
