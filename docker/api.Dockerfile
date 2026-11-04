FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]" \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /outputs && chown appuser:appuser /outputs
USER appuser
EXPOSE 8000
CMD ["uvicorn", "musicgen.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
