# 📋 Complete Feature & Capability Catalog — Build-an-LLM-From-Scratch Platform

Below is an exhaustive, categorized list of **every feature/capability** to include in `rnaarla/langmodel_from_Scratch`. It covers what's already in the dispatched scaffold PR **plus** everything that should be layered on top to reach a true production-grade, enterprise-ready foundation-model platform.

Legend: ✅ = in current scaffold PR · 🔜 = recommended next addition · 🏢 = enterprise/regulated-industry add-on

---

## 1. 🧱 Repository & Project Foundation
- ✅ Modular directory layout (`data/`, `tokenizer/`, `model/`, `eval/`, `alignment/`, `inference/`, `monitoring/`, `terraform/`, `helm/`, `docs/`)
- ✅ `Makefile` with reproducible CLI shortcuts
- ✅ `requirements.txt` + `pyproject.toml` (ruff, pytest)
- ✅ `.gitignore` for ML artifacts (checkpoints, mlruns, wandb, *.bin, tfstate)
- 🔜 `pre-commit` hooks (ruff, black, mypy, detect-secrets, nbstripout)
- 🔜 `CODEOWNERS`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
- 🔜 Issue + PR templates, semantic-release / Conventional Commits
- 🔜 Dev container (`.devcontainer/`) + reproducible Docker dev image
- 🔜 `uv` or `poetry` lockfile for deterministic envs
- 🏢 SBOM generation (CycloneDX/Syft) on every release

---

## 2. 🎯 Planning, Scaling & Compute Budgeting
- ✅ Chinchilla scaling-law calculator (~20 tokens/param)
- 🔜 FLOPs / GPU-hours / $-cost estimator CLI (`scripts/estimate_cost.py`)
- 🔜 Capacity-planning notebook (param count vs. memory vs. throughput)
- 🔜 MFU (Model FLOPs Utilization) target table per GPU class (A100, H100, MI300)
- 🏢 Carbon-footprint estimator (CodeCarbon / ML CO2)

---

## 3. 📚 Data Sourcing & Curation
- ✅ `data/prepare.py` skeleton (language filter, quality heuristics, MinHash dedup, PII scrub, Parquet output)
- ✅ `data/shard.py` for memmap binary shards
- ✅ `config/data_mixture.yaml` for domain weighting
- 🔜 Common Crawl / WARC ingestion adapter
- 🔜 The Stack / GitHub code ingestion with license filtering (SPDX)
- 🔜 Wikipedia, ArXiv, StackExchange, Books extractors
- 🔜 Trafilatura / Resiliparse HTML→text extraction
- 🔜 fastText language-ID + per-language quality classifiers
- 🔜 Exact + near-duplicate dedup (MinHash-LSH, suffix array)
- 🔜 Cross-shard global dedup
- 🔜 Toxicity / NSFW / hate-speech classifier filter
- 🔜 PII detection (Presidio) + redaction pipeline
- 🔜 Decontamination against eval benchmarks
- 🔜 Streaming dataset loader (`webdataset` / `mosaicml-streaming`)
- 🔜 DVC / LakeFS dataset versioning
- 🏢 License & provenance ledger per document
- 🏢 GDPR/CCPA right-to-be-forgotten workflow (doc-ID excision + targeted retraining)
- 🏢 Datasheet for Datasets (Gebru et al.) auto-generator
- 🏢 Differential privacy noise injection option (DP-SGD compatible)

---

## 4. 🔤 Tokenization
- ✅ ByteLevel BPE trainer (`tokenizer/train_tokenizer.py`)
- ✅ Tokenizer round-trip unit tests
- 🔜 SentencePiece Unigram alternative
- 🔜 Tiktoken-compatible export
- 🔜 Bytes-per-token / compression-ratio analyzer
- 🔜 Multilingual coverage report
- 🔜 Special-token + chat-template registry (ChatML, Llama, Mistral)
- 🔜 Vocabulary-expansion utility (embedding surgery)
- 🏢 Tokenizer versioning + checksum manifest pinned to model checkpoints

---

## 5. 🏗️ Model Architecture
- ✅ Decoder-only Transformer (RMSNorm, RoPE, GQA, SwiGLU, SDPA flash-attn)
- ✅ `ModelConfig` dataclass + `from_config` constructor
- ✅ Tied embeddings option, dropout, configurable n_kv_heads
- ✅ Architecture unit tests (shape, causal mask, gradient flow)
- 🔜 ALiBi position-embedding alternative
- 🔜 Sliding-window / Mistral-style attention
- 🔜 Mixture-of-Experts (MoE) layer (Switch / Mixtral-style top-k routing)
- 🔜 Mamba / SSM block (hybrid architecture experiment)
- 🔜 Multi-Token Prediction (MTP) head
- 🔜 Rotary embedding YaRN/NTK scaling for context extension
- 🔜 Activation checkpointing toggles
- 🔜 Reference-comparison test vs. HF GPT-NeoX / Llama at small scale
- 🏢 Numerical-stability test suite (BF16/FP16/FP8)

---

## 6. 🚀 Distributed Pre-Training
- ✅ `torchrun`-launchable training loop, BF16 autocast
- ✅ AdamW (β=0.9, 0.95), cosine LR + linear warmup, grad clip 1.0
- ✅ Periodic checkpointing + auto-resume
- ✅ Logs loss / lr / grad_norm / tokens-per-sec / MFU
- 🔜 FSDP (full-shard, hybrid-shard) wrapper
- 🔜 DeepSpeed ZeRO-1/2/3 integration
- 🔜 Megatron-LM tensor parallel
- 🔜 Pipeline parallel (PiPPy / DeepSpeed)
- 🔜 Sequence / context parallel for long context
- 🔜 Flash-Attention 2/3 explicit integration
- 🔜 Fused kernels (apex / liger-kernel)
- 🔜 `torch.compile` integration
- 🔜 Async + sharded checkpointing (`torch.distributed.checkpoint`)
- 🔜 Curriculum / data-mixture scheduler
- 🔜 Loss-spike detector with auto-rollback to last good checkpoint
- 🔜 Determinism mode (seeded, deterministic kernels)
- 🔜 NaN/Inf gradient guard
- 🔜 Slurm + Kubernetes (Volcano / Kubeflow Training Operator) launch templates
- 🏢 Spot/preemptible-aware checkpoint cadence
- 🏢 Multi-cluster federated training option

---

## 7. ⚡ Long-Context & Continued Pre-Training
- 🔜 Context-extension recipe (RoPE θ rescaling, YaRN)
- 🔜 Document packing with attention masks
- 🔜 Continued / domain-adaptive pre-training pipeline
- 🔜 Tokenizer-preserving model upcycling (dense → MoE)

---

## 8. 📏 Evaluation
- ✅ Held-out perplexity + bits-per-byte (`eval/perplexity.py`)
- ✅ `lm-evaluation-harness` wrapper script
- 🔜 Standard benchmarks: MMLU, HellaSwag, ARC, PIQA, WinoGrande, GSM8K, MATH, HumanEval, MBPP, TriviaQA, NQ
- 🔜 Multilingual: MGSM, XNLI, FLORES
- 🔜 Long-context: RULER, Needle-in-a-Haystack, LongBench
- 🔜 Code: BigCodeBench, LiveCodeBench
- 🔜 Reasoning: BBH, AGIEval, ARC-AGI subset
- 🔜 Capability-vs-tokens curve plotter
- 🔜 Eval-set decontamination report
- 🔜 LLM-as-judge harness (MT-Bench, AlpacaEval)
- 🔜 Pairwise human-eval UI
- 🏢 Regression dashboard (CI fails on benchmark drop > threshold)
- 🏢 Domain-specific eval packs (legal, medical, finance) gated by license

---

## 9. 🛡️ Safety, Red-Teaming & Responsible AI
- 🔜 Refusal + safety SFT dataset templates
- 🔜 Constitutional AI critique-and-revise pipeline
- 🔜 Red-team prompt suite (HarmBench, AdvBench, JailbreakBench)
- 🔜 Toxicity / bias evals (RealToxicityPrompts, BBQ, BOLD, WinoBias)
- 🔜 Refusal-rate + over-refusal (XSTest) metrics
- 🔜 Watermarking option (Kirchenbauer et al.)
- 🔜 Output classifier guardrails (Llama Guard / NeMo Guardrails)
- 🔜 Prompt-injection test suite
- 🏢 Model-risk-management report template (NIST AI RMF, EU AI Act mapping)

---

## 10. 🧭 Alignment (SFT + Preference Optimization)
- ✅ SFT skeleton with response-only loss masking
- ✅ DPO skeleton with frozen reference model
- 🔜 IPO, KTO, ORPO, SimPO loss variants
- 🔜 PPO/RLHF loop (TRL-compatible)
- 🔜 Reward-model training pipeline
- 🔜 Rejection sampling + Best-of-N data generator
- 🔜 Self-instruct / Magpie synthetic data generation
- 🔜 Tool-use / function-calling SFT format
- 🔜 Multi-turn chat-template trainer
- 🔜 Annotator-agreement (Krippendorff's α) reporting
- 🏢 Preference-data lineage (annotator ID, guideline version, IRR)

---

## 11. 🔍 Retrieval-Augmented Generation (RAG)
- 🔜 Embedding-model trainer/loader
- 🔜 Vector store adapters (FAISS, pgvector, Qdrant, Weaviate, Milvus)
- 🔜 Hybrid BM25 + dense retrieval
- 🔜 Reranker (Cohere, BGE-reranker) integration
- 🔜 LangChain / LlamaIndex pipelines
- 🔜 Citations + grounded-answer evaluator
- 🏢 Document ACLs propagated to retrieval results

---

## 12. 🛠️ Agentic & Tool-Use Capabilities
- 🔜 Function-calling schema validator
- 🔜 Tool-use eval harness (BFCL, ToolBench)
- 🔜 Plan-and-execute / ReAct agent template
- 🔜 Sandboxed code execution (Pyodide / Firejail)
- 🏢 Audit log of every tool invocation

---

## 13. 🧪 Inference Optimization
- 🔜 INT8 (bitsandbytes), INT4 (GPTQ, AWQ), FP8 (H100/MI300) quantization scripts
- 🔜 Knowledge-distillation pipeline (teacher → student)
- 🔜 Speculative decoding (draft + verifier)
- 🔜 Medusa / EAGLE multi-head speculative
- 🔜 Pruning / sparsification (Wanda, SparseGPT)
- 🔜 LoRA / QLoRA adapter training + merging
- 🔜 ONNX / TensorRT-LLM / OpenVINO export
- 🔜 KV-cache quantization
- 🔜 Continuous-batching benchmark harness

---

## 14. 🌐 Inference Serving
- ✅ FastAPI gateway (`/generate`, `/healthz`, `/metrics`)
- ✅ Dockerfile + vLLM launch script
- 🔜 OpenAI-compatible `/v1/chat/completions` + `/v1/completions` + `/v1/embeddings`
- 🔜 Streaming SSE / WebSocket responses
- 🔜 vLLM, TGI, SGLang, TensorRT-LLM backend selectors
- 🔜 PagedAttention + prefix caching
- 🔜 Multi-LoRA hot-swap serving
- 🔜 Request batching + admission control
- 🔜 Token-level rate limiting + quota per API key
- 🔜 Response caching (Redis)
- 🔜 Structured output / JSON-mode / grammar-constrained decoding (Outlines, XGrammar)
- 🔜 Safe-decoding hooks (logit bias, stop sequences)
- 🏢 mTLS, OAuth2/JWT, API-key rotation, per-tenant isolation

---

## 15. 🖥️ User Interfaces
- 🔜 Streamlit playground (prompt + response + feedback capture)
- 🔜 Gradio chat demo
- 🔜 Next.js / React reference chat UI
- 🔜 Pairwise comparison UI for human eval
- 🏢 SSO (OIDC/SAML) + RBAC on UIs

---

## 16. 🔁 Feedback, Active Learning & Continual Improvement
- 🔜 JSONL + MLflow dual feedback logger
- 🔜 Thumbs-up/down + free-text + rationale capture
- 🔜 Curation pipeline → SFT/DPO dataset
- 🔜 Prodigy + Label Studio integration
- 🔜 Drift detector on prompts and embeddings
- 🔜 Auto-trigger retraining on feedback delta
- 🏢 Human-in-the-loop review queues with SLA tracking

---

## 17. 📈 Monitoring, Observability & SRE
- ✅ Prometheus scrape config, Grafana dashboard placeholder, Alertmanager rules
- 🔜 LLM-specific metrics: TTFT, ITL, tokens/sec, KV-cache hit rate, queue depth, refusal rate
- 🔜 OpenTelemetry traces across gateway → engine → tools
- 🔜 Structured JSON logs (Loki / ELK)
- 🔜 Cost-per-request dashboard
- 🔜 SLO/SLI definitions + error-budget burn alerts
- 🔜 GPU telemetry (DCGM exporter)
- 🔜 Training dashboards (loss, grad-norm, MFU live)
- 🏢 PagerDuty / Opsgenie integration, runbooks

---

## 18. 🗃️ Metadata, Lineage & Feature Store
- 🔜 MLflow model registry stages (`pretrained → sft → aligned → quantized → prod`)
- 🔜 Weights & Biases / Aim alternative trackers
- 🔜 Feast / Tecton feature store integration
- 🔜 Dataset-↔-model lineage graph (OpenLineage / Marquez)
- 🏢 Immutable audit log (signed manifests, transparency log)

---

## 19. ☁️ Infrastructure as Code
- ✅ Terraform: AWS GPU instance + S3 artifacts bucket + SG
- 🔜 GCP (A3/H100) and Azure (ND H100 v5) modules
- 🔜 EKS/GKE/AKS cluster modules with GPU node pools + EFA/IB networking
- 🔜 Lustre/FSx/GCS-Fuse high-throughput training storage
- 🔜 Ray / KubeRay cluster module
- 🔜 Slurm cluster module
- 🔜 Atlantis or Terraform Cloud workflow
- 🏢 KMS encryption everywhere, VPC endpoints, private subnets only

---

## 20. ⛵ Kubernetes & Orchestration
- ✅ Helm chart (deployment, service, ingress) for inference
- 🔜 KEDA ScaledObject (Prometheus / queue triggers)
- 🔜 HPA + PodDisruptionBudget + PriorityClass
- 🔜 GPU node selectors, tolerations, MIG support
- 🔜 Kueue / Volcano gang scheduling for training
- 🔜 ArgoCD / Flux GitOps app-of-apps
- 🔜 Argo Workflows / Kubeflow Pipelines for data + training DAGs
- 🏢 NetworkPolicies, OPA/Gatekeeper, Kyverno policies

---

## 21. 🔁 CI/CD
- ✅ GitHub Actions: lint + tests + tokenizer round-trip + smoke train
- ✅ Nightly eval workflow placeholder
- ✅ Tag-driven release workflow placeholder
- 🔜 Docker buildx + multi-arch + GHCR push
- 🔜 Helm chart lint + `helm test`
- 🔜 Terraform `fmt/validate/plan` checks
- 🔜 Canary + shadow-traffic deploy job
- 🔜 Eval-regression gate (block release on benchmark drop)
- 🔜 SLSA-3 provenance + cosign signing of artifacts
- 🏢 SAST (Semgrep, CodeQL), DAST, secret scanning, dependency review, Trivy/Grype container scans

---

## 22. 🔐 Security & Compliance
- 🔜 Secrets via Vault / AWS Secrets Manager / SOPS
- 🔜 Encryption at rest (KMS) + in transit (mTLS)
- 🔜 Per-tenant data + key isolation
- 🔜 Audit logging of all API + admin actions
- 🔜 Threat model (STRIDE) document
- 🔜 Prompt-injection + data-exfiltration tests in CI
- 🏢 SOC 2 / ISO 27001 control mapping
- 🏢 HIPAA / PCI-DSS / FedRAMP overlays
- 🏢 EU AI Act + NIST AI RMF compliance artifacts
- 🏢 Model + weight-file signing (sigstore)

---

## 23. 🏛️ Governance & Documentation
- 🔜 Model card template (`docs/model_card_template.md`)
- 🔜 Datasheet template (`docs/datasheet_template.md`)
- 🔜 Evaluation report template
- 🔜 Acceptable-use policy + responsible-AI statement
- 🔜 Decision log / ADRs (`docs/adr/`)
- 🔜 Architecture diagrams (Mermaid / Excalidraw)
- 🏢 Model-risk-management committee workflow

---

## 24. 💰 Cost & FinOps
- 🔜 Per-request cost tagging (model, tenant, route)
- 🔜 Spot-vs-on-demand training optimizer
- 🔜 Idle-GPU autoscaler (KEDA scale-to-zero on dev)
- 🔜 Monthly cost report generator
- 🏢 Showback / chargeback by tenant or business unit

---

## 25. 🧰 Developer Experience & Testing
- 🔜 Pytest suites for: data pipeline, tokenizer, model shapes, training step, serving API, Helm render
- 🔜 Property-based tests (Hypothesis) for tokenizer + decoding
- 🔜 Load/perf tests (`locust`, `k6`) against inference
- 🔜 Notebook gallery (`notebooks/`) — tiny train, eval walkthrough, RAG demo, LoRA demo
- 🔜 CLI (`llmfs ...`) unifying all Make targets via Typer
- 🔜 Benchmarks folder with reproducible scripts
- 🔜 Docs site (MkDocs Material or Docusaurus) auto-published to GitHub Pages

---

## 26. 🌍 Community & Release
- 🔜 Versioned releases with changelog
- 🔜 Quickstart Colab / Modal / RunPod templates
- 🔜 HuggingFace Hub publishing script (model + tokenizer + card)
- 🔜 Example end-to-end "Train a 125M model in 1 GPU-hour" tutorial
- 🏢 Enterprise support / commercial-license path

---

### 🎯 Suggested Implementation Order (Phased Roadmap)

| Phase | Focus | Items |
|---|---|---|
| **P0 — Scaffold (in current PR)** | Get the skeleton building & testing | §1, §4✅, §5✅, §6✅, §8✅, §14✅, §19✅, §20✅, §21✅ baseline |
| **P1 — Train a real tiny model** | End-to-end runnable at 125M | Full §3, §6 (FSDP + Flash-Attn), §8 standard benches |
| **P2 — Align & serve** | Make it useful | §10 (SFT+DPO), §13 (quant), §14 (OpenAI-compatible + streaming) |
| **P3 — Productionize** | Reliability + observability | §17, §18, §22, §23 |
| **P4 — Differentiate** | Advanced capabilities | §7, §9, §11, §12, §15, §16 |
| **P5 — Enterprise hardening** | Regulated-industry ready | All 🏢 items, §24, §26 |