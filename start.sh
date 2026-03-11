#!/bin/bash
set -e

echo "Starting Python FastAPI Server on port 8001..."
# Run python in the background
uvicorn src.app:app --host 127.0.0.1 --port 8001 &

echo "Starting Go Reverse Proxy on port ${PORT:-8080}..."
# Run Go server in the foreground so the container stays alive
/usr/local/bin/vchat-server