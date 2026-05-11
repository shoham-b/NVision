FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir "uv==0.8.15"

COPY pyproject.toml uv.lock ./
COPY nvision ./nvision

RUN uv sync --frozen --no-dev


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app
ENV RUN_GROUP="all"
ENV NVISION_GCP_BUCKET=nvision-artifacts-shoha
# Set NVISION_GCP=True at runtime to enable GCP upload
ENV NVISION_GCP=False

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY nvision ./nvision

ENTRYPOINT ["sh", "-c", "python -m nvision groups run $RUN_GROUP \"$@\"", "--"]
