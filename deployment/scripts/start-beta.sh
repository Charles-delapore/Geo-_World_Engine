#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/deployment/docker-compose.beta.yml"
ENV_FILE="$ROOT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  cp "$ROOT_DIR/.env.example" "$ENV_FILE"
  echo "Created .env from .env.example"
fi

docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d postgres redis minio
echo "Infrastructure started. Run full stack with:"
echo "docker compose -f $COMPOSE_FILE --env-file $ENV_FILE up --build"
