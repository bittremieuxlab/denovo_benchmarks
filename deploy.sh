#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Build images
docker compose -f docker-compose.deployment.yaml build

# Start services
docker compose -f docker-compose.deployment.yaml up -d --remove-orphans
