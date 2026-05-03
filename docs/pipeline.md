# Building an LLM from Scratch — Stage-by-Stage Pipeline

This document is the canonical long-form design reference for the LLM-from-scratch pipeline.
Each stage describes purpose, key components, code examples, and enterprise/production
considerations.

---

## Stage 0: Repository Structure & Modular Architecture 🧱

A clean, modular layout makes a from-scratch LLM project tractable across data, modelling,
infra, and eval teams.  Each directory owns exactly one stage of the pipeline and exposes a
clean CLI via `argparse` so that stages can be orchestrated independently (Makefile, Airflow,
Prefect, or CI/CD).

```
langmodel_from_Scratch/
├── config/            ← YAML hyperparameter configs (model, tokenizer, data mixture)
├── data/              ← Cleaning, dedup, PII scrub, memmap sharding
├── tokenizer/         ← BPE tokenizer training & artifacts
├── model/             ← Architecture, training loop, checkpointing
├── eval/              ← Perplexity, lm-evaluation-harness wrapper
├── alignment/         ← SFT, DPO/RLHF
├── inference/         ← FastAPI gateway, vLLM launch, Dockerfile
├── monitoring/        ← Prometheus scrape config, Grafana dashboard, alert rules
├── terraform/         ← GPU EC2 + S3 IaC (AWS)
├── helm/              ← Kubernetes deployment chart
└── docs/              ← This file + model card + datasheet templates
```

**Enterprise note:** Version every stage artifact (dataset hash, tokenizer hash, model checkpoint
SHA) in MLflow or W&B so any production incident can be traced back to the exact training run
and data snapshot.

---

## Stage 1: Define Scope, Scaling Laws & Compute Budget 🎯

Before writing code, decide *what* you are building and *what it will cost*.

### Chinchilla scaling law
Compute-optimal training requires approximately **20 tokens per parameter**:

```python
N = 1.3e9            # parameters
D = 26e9             # tokens (Chinchilla-optimal: ~20×N)
flops = 6 * N * D   # C ≈ 6ND FLOPs

gpu_flops_per_s = 312e12   # A100 BF16 sustained (~50% of 624 TFLOPs peak)
n_gpus = 64
seconds = flops / (gpu_flops_per_s * n_gpus)
print(f"Estimated GPU-days: {seconds / 86400:.1f}")
```

### Hyperparameter heuristics

| Params | d_model | n_layers | n_heads | d_ff   |
|--------|---------|----------|---------|--------|
| 125 M  | 768     | 12       | 12      | 3 072  |
| 1.3 B  | 2 048   | 24       | 16      | 5 504  |
| 7 B    | 4 096   | 32       | 32      | 11 008 |

### Enterprise considerations
- 💰 Reserve GPU capacity early — capacity planning is the hardest constraint.
- 📜 Write the **model card hypothesis** before training: intended use, risks, eval plan.
- 🧾 Get legal sign-off on data sources up front (license review).

---

## Stage 2: Data Sourcing & Curation 📚

LLM quality is **dominated by data quality**, not architecture choices.

### Cleaning pseudocode (see `data/prepare.py`)

```python
for shard in raw_shards:
    docs = extract_text(shard)              # trafilatura / resiliparse
    docs = filter_language(docs, lang="en") # fastText lid.176
    docs = quality_filter(docs)             # length, symbol ratio, avg word len
    docs = dedupe_minhash(docs)             # MinHash-LSH, Jaccard ≥ 0.85
    docs = pii_scrub(docs)                  # emails, phones, SSNs via regex
    docs = toxicity_filter(docs)            # TODO: classifier threshold
    write_parquet(docs, f"cleaned/{shard}")
```

### Data mixture (`config/data_mixture.yaml`)

```yaml
mixture:
  web:           0.60   # Common Crawl / FineWeb
  code:          0.15   # The Stack
  books:         0.10
  wikipedia:     0.05
  arxiv:         0.05
  stackexchange: 0.05
```

### Enterprise considerations
- 🔐 Store `source_url` + `source_license` per document for provenance.
- 🧾 Maintain a **datasheet** (Gebru et al.) for the corpus.
- ⚖️ GDPR/CCPA: support deletion via document-id re-filtering.
- 🔁 Use **DVC** or LakeFS for dataset snapshots; pin every run to a `dataset_sha`.

---

## Stage 3: Tokenizer Training 🔤

The tokenizer is trained **once** and frozen — model weights are forever tied to it.

### Train (see `tokenizer/train_tokenizer.py`)

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["data/cleaned/sample.txt"],
    vocab_size=32_000,
    min_frequency=2,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
)
tokenizer.save_model("tokenizer/artifacts/")
```

### Validation
- Bytes-per-token on held-out English: target **3.5–4.5**.
- Coverage check on code, math symbols, non-Latin scripts.

### Pre-tokenise corpus (`data/shard.py`)
Tokenise once and write `uint16` NumPy memmap `.bin` shards for fast streaming during training:

```bash
python data/shard.py \
    --input-dir data/cleaned \
    --tokenizer-dir tokenizer/artifacts \
    --output-dir data/tokenized \
    --shard-size 500000000
```

### Enterprise considerations
- 🧾 Version tokenizer alongside model: `tokenizer_v1.json`.
- 🔁 Plan for vocabulary expansion via embedding surgery (rare but possible).

---

## Stage 4: Model Architecture Design 🏗️

Implement a modern **decoder-only Transformer** from primitives (see `model/architecture.py`).

### Core components

```python
import torch, torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.qkv   = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj  = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # SwiGLU gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)   # SwiGLU value
        self.w3 = nn.Linear(d_ff, d_model, bias=False)   # SwiGLU down
        self.n_heads = n_heads

    def forward(self, x, cos, sin):
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k = apply_rope(q, cos), apply_rope(k, sin)
        a = scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a)
        h = self.norm2(x)
        x = x + self.w3(nn.functional.silu(self.w1(h)) * self.w2(h))  # SwiGLU
        return x
```

### Design choices
- **RMSNorm** — faster than LayerNorm, equally stable.
- **RoPE** — applied to Q and K; position information via rotation, not addition.
- **GQA** — `n_kv_heads ≤ n_heads`; reduces KV-cache size linearly.
- **SwiGLU** — three-linear gated activation; +1–3 % accuracy vs. ReLU MLP.
- **SDPA** — uses FlashAttention2 kernel when available (torch ≥ 2.0).

### Enterprise considerations
- ✅ Unit-test forward/backward at 10 M-param scale before scaling up.
- 📐 Numerical-stability tests in BF16.
- 🧪 Shape & gradient hooks for debugging.

---

## Stage 5: Distributed Pre-Training 🚀

This is the most **expensive and risky** stage — design for resumability and observability.

### Parallelism strategy

| Strategy | When to use |
|----------|-------------|
| **DDP** | Always (data parallel across GPUs) |
| **FSDP / ZeRO-3** | Essential for > 1 B params (shards params + grads + opt states) |
| **Tensor Parallel** | Shard matmuls across GPUs (Megatron-LM style) |
| **Pipeline Parallel** | Very deep models / cross-node |

### Training loop sketch (see `model/train.py`)

```python
for step, batch in enumerate(loader):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch["input_ids"], labels=batch["input_ids"])
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()

    if step % 100 == 0:
        log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0],
             "grad_norm": grad_norm, "tokens_per_sec": tps, "mfu": mfu})
    if step % 2000 == 0:
        save_checkpoint(model, optimizer, scheduler, step)
```

### Launch

```bash
torchrun --nnodes=8 --nproc_per_node=8 \
  model/train.py --config config/pretrain_1b.yaml
```

### Key metrics to watch
- 📉 **Training loss** — should decrease smoothly; spikes → LR too high or bad shard.
- 📈 **Gradient norm** — divergence to NaN → restart from last checkpoint.
- ⚡ **MFU** — target 40–55 % on A100/H100.

### Enterprise considerations
- 💾 Async checkpointing every 30–60 min; keep last N + best.
- 🛟 Auto-resume on node failure (Slurm requeue / K8s Job restart).
- 🧾 Log every config + git SHA + dataset SHA to MLflow / W&B.
- 🔐 Encrypt checkpoints at rest (these *are* your IP).

---

## Stage 6: Evaluation of the Base Model 📏

### Intrinsic metrics

```bash
python eval/perplexity.py \
    --checkpoint checkpoints/step_200000.pt \
    --data-file data/eval/held_out.bin
```

### Few-shot benchmarks via `lm-evaluation-harness`

```bash
bash eval/run_harness.sh ./checkpoints/step_200000 \
    "mmlu,hellaswag,arc_challenge,gsm8k,humaneval"
```

Recommended benchmarks:
- **MMLU** — knowledge breadth
- **HellaSwag, ARC, PIQA, WinoGrande** — commonsense reasoning
- **GSM8K, MATH** — arithmetic reasoning
- **HumanEval, MBPP** — code generation
- **TriviaQA** — factual recall

### Enterprise considerations
- 🧾 Log every eval to MLflow with checkpoint step.
- 🚨 Flag regressions automatically in CI (see `.github/workflows/eval.yml`).

---

## Stage 7: Alignment — SFT + Preference Optimisation 🧭

A raw base model completes text; alignment teaches it to follow instructions and be helpful.

### SFT (see `alignment/sft.py`)

```python
prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
target = ex["output"]
# loss masked so only target tokens contribute (labels[:prompt_len] = -100)
```

### DPO loss (see `alignment/dpo.py`)

```python
# DPO loss (Rafailov et al., 2023)
reward = beta * (
    (log_pi_chosen - log_pi_ref_chosen) -
    (log_pi_rejected - log_pi_ref_rejected)
)
loss = -F.logsigmoid(reward).mean()
```

### Enterprise considerations
- 🧑‍⚖️ Document refusal taxonomy and override policies.
- 🧾 Track preference data lineage: annotators, guidelines version, inter-rater scores.

---

## Stage 8: Post-Training Optimisation for Inference ⚡

| Technique | Tool | Speedup |
|-----------|------|---------|
| INT8 quantization | bitsandbytes | 2× memory |
| INT4 quantization | GPTQ / AWQ | 4× memory |
| FP8 quantization | TensorRT-LLM | 2× on H100 |
| Speculative decoding | vLLM | 2–3× latency |
| KV-cache paging | PagedAttention | higher throughput |

```bash
# AWQ 4-bit quantization example
python -m awq.quantize \
    --model ./checkpoints/final \
    --w_bit 4 \
    --out ./model_awq
```

---

## Stage 9: Inference Serving 🌐

```bash
# Start vLLM (see inference/serve_vllm.sh)
bash inference/serve_vllm.sh ./model_awq awq 2

# Start FastAPI gateway (proxies to vLLM)
VLLM_BASE_URL=http://localhost:8000 uvicorn inference.app:app --port 9000

# Kubernetes deployment
helm install llm-api helm/llm-inference-chart/ -f helm/llm-inference-chart/values.yaml
```

### Enterprise considerations
- **Continuous batching + PagedAttention** are non-negotiable for production cost.
- **KEDA** for queue-length-based autoscaling (configure in `helm/llm-inference-chart/values.yaml`).
- **Canary deployment**: route 10 % of traffic to new checkpoint; promote after eval gates pass.

---

## Stage 10: Monitoring, Feedback & Continual Improvement 🔁

- **Prometheus/Grafana** dashboards: tokens/sec, TTFT, ITL, GPU memory, KV-cache hit rate.
- **JSONL + MLflow** dual logging of prompts, responses, feedback scores.
- **Active learning loop**: curated SFT/DPO data → next alignment round.
- **Drift detection** on prompt distributions and output embeddings.

Deploy monitoring stack:

```bash
docker compose up prometheus grafana  # using monitoring/*.yml
```

---

## Stage 11: CI/CD, IaC & Governance 🛡️

- **CI gates** (`.github/workflows/ci.yml`): ruff lint, unit tests, tokenizer round-trip,
  smoke train.
- **Nightly eval** (`.github/workflows/eval.yml`): lm-eval-harness on latest checkpoint.
- **Release** (`.github/workflows/release.yml`): tag-driven model card + artifact publish.
- **Terraform** (`terraform/`): GPU EC2 node pool + S3 artifact bucket (AWS).
- **MLflow Model Registry**: stages `pretrained → sft → aligned → quantized → production`.
- **Model card + datasheet + eval report** auto-generated on every release tag.

---

## 🔁 The Build-from-Scratch Flywheel

| Stage | Output | Feeds Into |
|-------|--------|------------|
| 1. Plan | Sizing, budget, scope | All stages |
| 2. Data | Versioned, deduped corpus | Tokenizer, Pre-training |
| 3. Tokenizer | Frozen vocab + merges | Pre-training, Inference |
| 4. Architecture | Tested model code | Pre-training |
| 5. Pre-training | Base checkpoints | Eval, Alignment |
| 6. Eval | Capability report | Go/no-go decisions |
| 7. Alignment | SFT + DPO model | Safety review, Serving |
| 8. Optimise | Quantized/distilled model | Serving |
| 9. Serve | Production endpoint | Monitoring, Feedback |
| 10. Feedback | Curated pref data | Next alignment round |
| 11. Govern | Audit + lineage | Compliance, Trust |
