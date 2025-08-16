#!/bin/bash
set -e
set -x

docker compose build
docker compose up -d --remove-orphans
docker image prune -f
docker buildx prune --filter until=24h
