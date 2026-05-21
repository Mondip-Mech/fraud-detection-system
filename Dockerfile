# syntax=docker/dockerfile:1.6

# Multi-stage build: smaller final image, cached dependency layer
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (some packages need C compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a clean virtualenv
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Final stage ----
FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies only (libgomp1 needed by LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 appuser

# Copy the virtualenv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code and model artifacts
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# Run as non-root user (security best practice)
USER appuser

# HuggingFace Spaces expects port 7860; standard local convention is 8000
# We'll honor an env var so the same image works for both
ENV PORT=7860
EXPOSE 7860

# Health check — Docker will mark container unhealthy if /health stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen(f'http://localhost:${PORT}/health').read()" || exit 1

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]