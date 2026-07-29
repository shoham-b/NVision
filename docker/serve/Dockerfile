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

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY nvision ./nvision
COPY static ./static

# `nv serve` requires <dir>/cache to exist -- with the MySQL cache backend the
# actual data lives in Cloud SQL, not on disk, so an empty placeholder is enough.
RUN mkdir -p artifacts/cache

# Cloud Run injects PORT (defaulted to whatever --port the service was deployed
# with, e.g. 8080) and requires listening on 0.0.0.0. 18080 is the fallback so
# `docker-compose.yml`'s existing "18080:18080" port mapping keeps working.
ENTRYPOINT ["sh", "-c", "python -m nvision serve --host 0.0.0.0 --port \"${PORT:-18080}\" --no-open \"$@\"", "--"]
