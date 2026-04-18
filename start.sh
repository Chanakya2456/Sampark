#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Single-container entry point for the Rail Madad Databricks App.
#
# Architecture inside the container:
#   • FastAPI backend  → uvicorn on 127.0.0.1:8000 (private, loopback only)
#   • Streamlit UI     → streamlit on 0.0.0.0:$DATABRICKS_APP_PORT (public)
#
# The Streamlit process calls the API over loopback via
#   API_BASE_URL=http://127.0.0.1:8000
# which is set below and can be overridden from the App's Environment tab.
# ---------------------------------------------------------------------------
set -uo pipefail

# Databricks Apps injects DATABRICKS_APP_PORT (default 8000). Streamlit MUST own it.
: "${DATABRICKS_APP_PORT:=8080}"

# FastAPI runs on a private internal port — must NOT clash with DATABRICKS_APP_PORT.
: "${API_PORT:=8001}"
export API_PORT

# The Streamlit frontend calls the backend on the loopback interface.
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:${API_PORT}}"

# ── Start FastAPI in the background ─────────────────────────────────────────
uvicorn api:app \
    --host 127.0.0.1 \
    --port "${API_PORT}" \
    --log-level info &
API_PID=$!

# If the child dies, tear the container down so Databricks restarts it cleanly.
trap 'kill "${API_PID}" 2>/dev/null || true' EXIT

# ── Wait for /healthz before serving UI ─────────────────────────────────────
echo "[start] Waiting for FastAPI backend on 127.0.0.1:${API_PORT}…"
for _ in $(seq 1 30); do
    if python - <<PY >/dev/null 2>&1
import os, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{os.environ['API_PORT']}/healthz", timeout=1)
PY
    then
        echo "[start] Backend ready."
        break
    fi
    sleep 1
done

# ── Start Streamlit in the foreground on the public port ────────────────────
# Extra flags needed behind Databricks Apps' reverse proxy so websockets, XSRF,
# and CORS don't blank the page.
exec streamlit run app.py \
    --server.port "${DATABRICKS_APP_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.enableWebsocketCompression false \
    --browser.gatherUsageStats false
