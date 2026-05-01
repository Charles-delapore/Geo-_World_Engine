FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY alembic.ini .
COPY migrations ./migrations
COPY bootstrap_db.py .
COPY celery_worker.py .
COPY tile_worker.py .
COPY docker-entrypoint.sh .
COPY tests ./tests
COPY pyproject.toml .

RUN mkdir -p /app/data && \
    chmod +x /app/docker-entrypoint.sh && \
    chown -R appuser:appgroup /app

USER appuser

ENTRYPOINT ["/app/docker-entrypoint.sh"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
