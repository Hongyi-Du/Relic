# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.15 AS uv

FROM python:3.12-slim-bookworm

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RELIC_HOME=/app \
    RELIC_BENCHMARK_ROOT=/app/benchmarks \
    RELIC_OUTPUT_ROOT=/data/outputs \
    RELIC_CACHE_ROOT=/data/cache

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 relic

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY benchmarks ./benchmarks
COPY configs ./configs
COPY environments ./environments
COPY relic ./relic

RUN uv sync --frozen --no-dev --no-editable \
    && mkdir -p /data/outputs /data/cache /data/traces \
    && chown -R relic:relic /data

USER relic

VOLUME ["/data/outputs", "/data/cache", "/data/traces"]
EXPOSE 8765
ENTRYPOINT ["relic"]
CMD ["check-env", "--scope", "core"]
