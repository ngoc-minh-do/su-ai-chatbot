#!/bin/bash
set -e
set -x

# 1️⃣ Build the GPU-dependent stage manually using buildx
DOCKER_BUILDKIT=1 docker buildx build --target builder --platform linux/amd64 -t su-ai-chatbot-builder .

# 2️⃣ Build the final runtime image (no GPU needed at build time)
docker compose build

# 3️⃣ Run container
docker compose up -d --remove-orphans

# 4️⃣ Cleanup old images
docker image prune -f
docker buildx prune --filter until=24h
