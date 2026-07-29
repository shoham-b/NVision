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

# Warm the numba `@njit(cache=True)` on-disk cache into this image layer. Cloud
# Run Job tasks start from a cold container every execution, so without this
# every shard pays the full JIT-compile cost for the lorentzian/voigt EIG
# kernels on its first step. --dry-run still runs the real locator pipeline
# (only cache writes/disk artifacts are skipped), so this exercises the actual
# njit call sites with tiny, cheap inputs (1 repeat, 1 runner).
RUN nv run-single NVCenter-lorentzian "Gauss(0.01)" Bayesian-SBED --repeats 1 --runners 1 --dry-run --no-progress \
    && nv run-single NVCenter-voigt "Gauss(0.01)" Bayesian-SBED --repeats 1 --runners 1 --dry-run --no-progress

ENTRYPOINT ["sh", "-c", "python -m nvision groups run $RUN_GROUP --no-progress \"$@\"", "--"]
