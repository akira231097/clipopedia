# syntax=docker/dockerfile:1
# ---- Build a slim image that runs the live bot loop -------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for psycopg / numpy wheels are already covered by slim + wheels.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package with the "live" adapters.
RUN pip install --upgrade pip && pip install ".[live]"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Default backend for a deployed container is the live one.
ENV CLIPOPEDIA_BACKEND=live

ENTRYPOINT ["python", "-m", "clipopedia"]
CMD ["run"]
