# syntax=docker/dockerfile:1

# ---------- Build stage: resolve and install dependencies with uv ----------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# ---------- Runtime stage: minimal image, non-root user ----------
FROM python:3.12-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app

COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --from=build --chown=app:app /app/src /app/src
COPY --chown=app:app config ./config

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    APP_ENV=paper \
    CONFIG_DIR=/app/config \
    LOG_FORMAT=json

USER app

ENTRYPOINT ["trading-agent"]
CMD ["healthcheck"]
