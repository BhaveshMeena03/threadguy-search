#!/bin/sh
# Launch the real backend (reads .env for API keys). PORT is set by the
# preview harness; defaults to 8100 when run by hand.
cd "$(dirname "$0")/.." && exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8100}"
