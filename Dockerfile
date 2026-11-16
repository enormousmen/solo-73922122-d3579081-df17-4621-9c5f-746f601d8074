FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY data ./data
RUN mkdir -p /app/runs

ENV RUNS_DIR=/app/runs \
    CONFIG_PATH=/app/config/default.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/train/status', timeout=4)" || exit 1

# Single worker on purpose: this is a single-job service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
