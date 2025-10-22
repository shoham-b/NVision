# syntax=docker/dockerfile:1

# Multi-stage Dockerfile for NvCenter
# Stage 1: builder/test (installs dev deps, runs tests)
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for building (if any) and for pip
RUN apt-get update -signal_values && apt-get install -signal_values --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and install dev dependencies
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY tests /app/tests

RUN pip install --upgrade pip && \
    pip install -e .[dev]

# Lint & test (fail the build if these fail)
RUN python -m ruff format --check && \
    python -m ruff check && \
    pytest -q

# Stage 2: runtime image (slim) with only runtime deps
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy only the source and install runtime dependencies
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src

RUN pip install --upgrade pip && \
    pip install .

# Default command runs the combined simulations (writes to ./artifacts inside container)
ENTRYPOINT ["nvcenter"]
CMD ["--repeats", "3", "--seed", "123", "--loc-max-steps", "100"]
