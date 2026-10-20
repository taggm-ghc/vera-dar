#!/bin/sh
# Container entry point for VERA (p3m3 item #73). One image, two Render services:
#   ./start.sh api   FastAPI (uvicorn main:app), health check /health
#   ./start.sh ui    Streamlit UI, health check /_stcore/health
# Port: Render sets PORT; 8000 otherwise (the port the Dockerfile EXPOSEs).
# One uvicorn worker on purpose: public-mode caps are in-memory per process
# (vera/public_mode.py), so extra workers would multiply the limits.
set -eu
PORT="${PORT:-8000}"
case "${1:-}" in
  api|ui) ;;
  *)
    echo "usage: $0 api|ui" >&2
    exit 2
    ;;
esac
# PLAN-ENV C11: this entry point is for the container only. Refuse anywhere else (names only, never values).
if [ "${RENDER:-}" != "true" ] && [ "${VERA_ENV_CONTEXT:-}" != "production" ]; then
  echo "start.sh: refusing to start: neither RENDER=true nor VERA_ENV_CONTEXT=production is set (container entry point only)" >&2
  exit 3
fi
case "${1:-}" in
  api)
    exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1
    ;;
  ui)
    exec streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port "$PORT" \
      --server.headless true --browser.gatherUsageStats false
    ;;
  *)
    echo "usage: $0 api|ui" >&2
    exit 2
    ;;
esac
