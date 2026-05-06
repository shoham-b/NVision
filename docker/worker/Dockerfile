FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir "uv==0.8.15"

COPY pyproject.toml uv.lock ./
COPY nvision ./nvision

RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app
ENV RUN_GROUP="all"
ENV NVISION_GCP_BUCKET=nvision-artifacts-shoha
# Set NVISION_GCP=True at runtime to enable GCP upload
ENV NVISION_GCP=False

# Numba requirements
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY nvision ./nvision

ENTRYPOINT ["sh", "-c", "python -m nvision groups run $RUN_GROUP \"$@\"", "--"]
