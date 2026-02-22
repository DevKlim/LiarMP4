#!/bin/bash
set -e

echo "Starting vChat Server..."

# Ensure data directories exist
mkdir -p /app/data/videos
mkdir -p /app/data/labels

# Run Uvicorn (FastAPI)
# Host 0.0.0.0 is critical for Docker
# Port 8000 is what docker-compose maps to 8005
echo "Launching uvicorn on port 8000..."
exec uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload