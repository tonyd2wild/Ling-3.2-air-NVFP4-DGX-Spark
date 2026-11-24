#!/usr/bin/env bash
# Stop and remove the Ling-3.2-air container. Weights and image stay on disk.
NAME="${NAME:-ling32air}"
docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "stopped $NAME"
free -g
