# syntax=docker/dockerfile:1

# Multi-stage Dockerfile for NVision using uv-only
# Stage 1: builder/test (installs dev deps, runs tests)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy project metadata and install dev dependencies
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY tests /app/tests
COPY requirements-dev.txt /app/requirements-dev.txt

RUN uv pip install -r requirements-dev.txt

# Lint & test (fail the build if these fail)
RUN uv run ruff format --check && \
    uv run ruff check && \
    uv run pytest -q

# Stage 2: runtime image (slim) with only runtime deps
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy only the source and install runtime dependencies
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
COPY requirements.txt /workspace/requirements.txt

RUN uv pip install -r /workspace/requirements.txt

# Default command runs the combined simulations (writes to ./artifacts inside container)
ENTRYPOINT ["uv", "run", "nvision"]
CMD ["--repeats", "3", "--seed", "123", "--loc-max-steps", "100"]
