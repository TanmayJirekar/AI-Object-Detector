#!/bin/sh
# Starts the Flask backend in the background, then the Streamlit
# frontend in the foreground (keeps the container alive and lets
# `docker logs` show the Streamlit output).
set -e

echo "Starting Flask backend on :5000 ..."
python backend/app.py &
BACKEND_PID=$!

cleanup() {
    echo "Shutting down..."
    kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting Streamlit frontend on :8501 ..."
streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
