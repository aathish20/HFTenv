#!/usr/bin/env bash
export HFT_TASK_ID="easy"

# Run the server
exec uvicorn hftenv.server.app:app --host 0.0.0.0 --port 8001
