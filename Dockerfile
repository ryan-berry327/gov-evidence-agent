# syntax=docker/dockerfile:1

# Multi-stage build: deps go into a venv in the builder stage, and only the
# finished venv + app code are copied into the runtime image. Keeps build tools
# and caches out of the final image.

# ---- Stage 1: builder -----------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first so the install layer is cached unless deps change.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: runtime -----------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser corpus/ ./corpus/
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,os; httpx.get(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health').raise_for_status()" || exit 1

# Shell form so $PORT expands (Azure Container Apps injects the port).
CMD uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT}
