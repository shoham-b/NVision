FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir "uv==0.8.15"

COPY pyproject.toml uv.lock ./
COPY nvision ./nvision

RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY nvision ./nvision

ENTRYPOINT ["python", "-m", "nvision", "groups", "run"]
