#!/bin/bash

# 1. Start Python FastAPI in the background (Internal Port 8001)
echo "Starting Python Inference Engine..."
export PYTHONPATH=$PYTHONPATH:/app/src
# Use --log-level info to see startup issues
python -m uvicorn src.app:app --host 127.0.0.1 --port 8001 --log-level info &

# Wait longer for Python to initialize, or until port is open
echo "Waiting for Python backend to initialize..."
timeout=30
while ! curl -s http://127.0.0.1:8001/ > /dev/null; do
    sleep 2
    timeout=$((timeout-2))
    if [ $timeout -le 0 ]; then
        echo "Python backend failed to start on time. Logs might show why."
        break
    fi
done

# 2. Start Golang Web Server (Public Port 7860)
echo "Starting Go Web Server..."
/app/vchat-server