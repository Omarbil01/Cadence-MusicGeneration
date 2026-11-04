FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts/manage_admin.py ./scripts/manage_admin.py
RUN pip install --no-cache-dir ".[frontend]" \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /admin-data && chown appuser:appuser /admin-data \
    && chmod 700 /admin-data
USER appuser
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--access-logfile", "-", "musicgen.frontend.app:create_app()"]
