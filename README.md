# 🧠 LLM from Scratch — End-to-End Pipeline

A **modular, production-grade pipeline** for building a Large Language Model from scratch —
from raw data collection through pre-training, alignment, inference optimisation, and
continuous monitoring.  This is a *pre-training* pipeline, not just fine-tuning.

---

## Quickstart

```bash
git clone https://github.com/rnaarla/langmodel_from_Scratch
cd langmodel_from_Scratch

# 1. Install dependencies
make install

# 2. Train tokenizer (requires .txt files in data/cleaned/)
make tokenizer

# 3. Smoke-train a tiny model (no GPU required)
make train-tiny

# 4. Evaluate perplexity
make eval

# 5. Start inference API
make serve
```

---

## Pipeline Overview

The full pipeline is documented in [`docs/pipeline.md`](docs/pipeline.md).

| Stage | Name | Description |
|-------|------|-------------|
| **0** | Repository structure | Modular layout, CI/CD, IaC |
| **1** | Scope & scaling laws | Chinchilla ~20 tokens/param, compute budget |
| **2** | Data sourcing & curation | MinHash dedup, PII scrub, mixture, DVC versioning |
| **3** | Tokenizer training | ByteLevel BPE 32k, validation, memmap shards |
| **4** | Architecture | Decoder-only, RMSNorm, RoPE, GQA, SwiGLU, SDPA |
| **5** | Distributed pre-training | FSDP/ZeRO, BF16, AdamW, cosine+warmup, MFU tracking |
| **6** | Evaluation | PPL, MMLU, HellaSwag, ARC, GSM8K, HumanEval |
| **7** | Alignment | SFT + DPO/RLHF, safety tuning |
| **8** | Inference optimisation | INT4/INT8/FP8 quantization, distillation, speculative decoding |
| **9** | Serving | vLLM/TGI, FastAPI gateway, Helm + KEDA |
| **10** | Monitoring & feedback | Prometheus/Grafana, MLflow, active learning loop |
| **11** | CI/CD, IaC, governance | Model card, datasheet, lineage, audit trail |

---

## Directory Map

```
.
├── README.md                     ← You are here
├── Makefile                      ← CLI shortcuts
├── requirements.txt              ← Python dependencies
├── pyproject.toml                ← Ruff / pytest config
├── config/                       ← YAML configs for all stages
├── data/                         ← Data pipeline (prepare + shard)
├── tokenizer/                    ← BPE tokenizer training
├── model/                        ← Architecture, training, checkpointing
│   └── tests/                    ← Unit tests (run with `make test`)
├── eval/                         ← Perplexity + lm-eval-harness wrapper
├── alignment/                    ← SFT + DPO alignment
├── inference/                    ← FastAPI gateway + Dockerfile
├── monitoring/                   ← Prometheus, Grafana, alert rules
├── terraform/                    ← AWS IaC (GPU EC2 + S3)
├── helm/llm-inference-chart/     ← Kubernetes Helm chart
└── docs/                         ← Pipeline doc, model card, datasheet
```

---

## Key Design Choices

- **Architecture:** Decoder-only Transformer with RMSNorm, RoPE, Grouped-Query Attention
  (GQA), SwiGLU MLP, and FlashAttention via `scaled_dot_product_attention`.
- **Training:** `torchrun`-launchable, BF16 autocast, AdamW β=(0.9, 0.95), cosine LR
  with linear warmup, gradient clipping at 1.0, periodic checkpointing with auto-resume.
- **Tokenizer:** 32k ByteLevel BPE (no UNK tokens, robust to any input).
- **Data:** MinHash near-dedup, PII scrub, quality heuristics, memmap binary shards.
- **Alignment:** SFT with response-only loss masking + DPO against a frozen reference.
- **Serving:** FastAPI gateway (optional vLLM proxy), Prometheus metrics, Helm chart.
- **Governance:** Model card template, datasheet template, MLflow lineage.

---

## Scaling Laws Reference

Using **Chinchilla** compute-optimal training (~20 tokens per parameter):

| Model size | Tokens needed | A100 GPU-days (64 GPUs) |
|------------|---------------|------------------------|
| 125 M      | 2.5 B         | ~0.5                   |
| 1 B        | 20 B          | ~4                     |
| 7 B        | 140 B         | ~28                    |
| 70 B       | 1.4 T         | ~280                   |

---

## Contributing

1. Fork the repo and create a branch: `git checkout -b feat/your-feature`.
2. Ensure `make lint` and `make test` pass before opening a PR.
3. Add or update tests for any new functionality.
4. Open a PR — the CI workflow runs automatically on push.

---

## License

See [LICENSE](LICENSE).
