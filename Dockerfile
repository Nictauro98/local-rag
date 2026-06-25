# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Bring in uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install deps only; source code excluded so this layer caches across code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY ui/  ./ui/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1

# Default command — overridden per-service in docker-compose.yml
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
