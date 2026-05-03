# Inference

This directory contains the production inference gateway for serving the trained LLM.

---

## Architecture

```
Client
  │
  ▼
FastAPI Gateway (inference/app.py)
  │  ├── POST /generate  — text generation
  │  ├── GET  /healthz   — liveness probe
  │  └── GET  /metrics   — Prometheus metrics
  │
  ├─► vLLM / TGI server (high-throughput GPU serving)  [optional]
  │
  └─► Local TransformerLM (CPU fallback for development)
```

---

## Running Locally

```bash
# CPU-only development (no vLLM required)
MODEL_DIR=checkpoints/final uvicorn inference.app:app --reload

# With vLLM backend (GPU required)
bash inference/serve_vllm.sh ./checkpoints/final
VLLM_BASE_URL=http://localhost:8000 uvicorn inference.app:app --port 9000
```

Test the API:

```bash
curl -X POST http://localhost:9000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 100}'
```

---

## Docker

```bash
docker build -t llm-inference:latest -f inference/Dockerfile .
docker run -p 8000:8000 \
  -e MODEL_DIR=/models/final \
  -e TOKENIZER_DIR=/models/tokenizer \
  -v /path/to/models:/models \
  llm-inference:latest
```

---

## Kubernetes (Helm)

```bash
helm install llm-api helm/llm-inference-chart/ \
  --set image.tag=latest \
  --set env.VLLM_BASE_URL=http://vllm-svc:8000
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `""` | If set, forward requests to vLLM at this URL |
| `MODEL_DIR` | `checkpoints/final` | Path to local model checkpoint directory |
| `TOKENIZER_DIR` | `tokenizer/artifacts` | Path to tokenizer artifacts |
| `MAX_NEW_TOKENS` | `256` | Default generation length cap |

---

## Enterprise Considerations

- **Throughput:** Always use vLLM or TGI in production — they provide continuous batching
  and PagedAttention, which can improve throughput by 10–20× over naive inference.
- **Latency:** For interactive applications, target TTFT (time-to-first-token) < 500 ms and
  ITL (inter-token latency) < 50 ms.
- **Security:** The gateway handles authentication, rate limiting, and input validation.
  Never expose the vLLM OpenAI-compatible endpoint directly to the internet.
- **Observability:** Prometheus metrics are exposed at `/metrics` via
  `prometheus-fastapi-instrumentator`.  Import `monitoring/grafana_dashboard.json` for
  pre-built dashboards.
- **Quantization:** Deploy INT4/AWQ-quantized models for 4× memory reduction with
  < 1 % accuracy loss on most benchmarks.
