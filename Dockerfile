FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    NEWSREN_DATA_DIR=/data

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home newsren && mkdir -p /data && chown newsren /data
USER newsren
VOLUME ["/data"]
EXPOSE 8000

# `newsren init` registers approved sources; run `newsren pipeline` from cron
# or set NEWSREN_SCHEDULER=true to run the workflow in-process.
CMD ["sh", "-c", "uv run --no-dev newsren init && uv run --no-dev newsren serve --host 0.0.0.0 --port 8000"]
