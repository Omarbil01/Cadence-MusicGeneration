FROM python:3.12-slim-bookworm
ARG TORCH_VERSION=2.6.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && pip install --no-cache-dir ".[worker]" \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /outputs /models && chown appuser:appuser /outputs /models
USER appuser
EXPOSE 9101
CMD ["celery", "-A", "musicgen.worker.celery_app:app", "worker", "--loglevel=INFO", "--pool=solo", "--concurrency=1", "--queues=generation", "--without-gossip", "--without-mingle"]
