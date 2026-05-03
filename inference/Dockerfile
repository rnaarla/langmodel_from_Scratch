FROM python:3.11-slim

LABEL maintainer="LLM-from-Scratch Pipeline"
LABEL description="Inference gateway for the LLM-from-scratch pipeline"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        prometheus-fastapi-instrumentator \
        tokenizers \
        httpx \
        pyyaml \
        tqdm

# Copy project source
COPY model/ ./model/
COPY inference/ ./inference/
COPY tokenizer/ ./tokenizer/

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
