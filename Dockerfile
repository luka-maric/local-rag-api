FROM python:3.11-slim

WORKDIR /app

# PYTHONPATH ensures the 'app' package is importable when running inside the container.
ENV PYTHONPATH=/app

# libgomp1: OpenMP parallelism required by sentence-transformers
# poppler-utils: PDF-to-image conversion for pdf2image (OCR path)
# tesseract-ocr: OCR engine used by pytesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy manifest first so the pip install layer is cached on code-only rebuilds.
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Non-root user. Pre-create the HuggingFace cache directory so the named
# Docker volume is seeded with correct ownership on first mount.
RUN useradd -m -u 1000 raguser \
    && chown -R raguser:raguser /app \
    && mkdir -p /home/raguser/.cache/huggingface \
    && chown -R raguser:raguser /home/raguser/.cache
USER raguser

EXPOSE 8000

# --start-period=60s: NER + embedding models take 30-45s to load on first start.
# --fail: curl exits non-zero on HTTP >= 400, so a 500 is correctly treated as unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
