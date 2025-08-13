#!/bin/bash
set -e
set -x

docker compose down --rmi local --remove-orphans
docker compose up -d --remove-orphans
