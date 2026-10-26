#!/bin/bash

# Build images
docker compose -f docker-compose.deployment.yaml build

# Start services
docker compose -f docker-compose.deployment.yaml up -d
