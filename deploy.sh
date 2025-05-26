#!/bin/bash
set -e
set -x

docker compose down --rmi 'all' --remove-orphans
docker compose up -d --remove-orphans
