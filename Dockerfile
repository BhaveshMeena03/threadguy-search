FROM python:3.12-slim

WORKDIR /srv

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY widget ./widget
COPY demo ./demo

# Non-root runtime user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/healthz')"

# Shell form so $PORT (set by Render/Heroku-style hosts) is honored.
CMD uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
