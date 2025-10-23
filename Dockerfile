# syntax=docker/dockerfile:1

# Multi-stage Dockerfile for NVision
# Stage 1: builder/test (installs dev deps, runs tests)
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for building (if any) and for pip
# Pin versions and add -y to satisfy hadolint DL3008/DL3014
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      build-essential=12.10 \
      curl \
    && rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

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
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy only the source and install runtime dependencies
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
COPY requirements.txt /workspace/requirements.txt

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv pip install -r /workspace/requirements.txt

# Default command runs the combined simulations (writes to ./artifacts inside container)
ENTRYPOINT ["uv", "run", "nvision"]
CMD ["--repeats", "3", "--seed", "123", "--loc-max-steps", "100"]
