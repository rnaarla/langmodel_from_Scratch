# Model Card — {{MODEL_NAME}} ({{VERSION}})

*Generated: {{DATE}}*

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | {{MODEL_NAME}} |
| **Version** | {{VERSION}} |
| **Architecture** | Decoder-only Transformer (RMSNorm, RoPE, GQA, SwiGLU) |
| **Parameters** | {{N_PARAMS}} |
| **Training tokens** | {{TRAINING_TOKENS}} |
| **Context length** | {{MAX_SEQ_LEN}} tokens |
| **Tokenizer** | ByteLevel BPE, vocab size {{VOCAB_SIZE}} |
| **Training hardware** | {{HARDWARE}} |
| **Training duration** | {{TRAINING_DURATION}} |
| **Checkpoint step** | {{CHECKPOINT_STEP}} |
| **Dataset SHA** | {{DATASET_SHA}} |
| **Git SHA** | {{GIT_SHA}} |

---

## Intended Use

**Primary use cases:**
- [Describe intended applications]

**Out-of-scope use cases:**
- This model must not be used for [harmful applications].
- Not suitable for [specific high-stakes domains] without additional safety validation.

---

## Training Data

| Source | Weight | License |
|--------|--------|---------|
| Web (Common Crawl / FineWeb) | 60 % | See source |
| Code (The Stack) | 15 % | Apache 2.0 / MIT |
| Books | 10 % | See source |
| Wikipedia | 5 % | CC-BY-SA 4.0 |
| ArXiv | 5 % | See source |
| StackExchange | 5 % | CC-BY-SA 4.0 |

Refer to the accompanying **datasheet** (`docs/datasheet_template.md`) for full corpus details.

---

## Evaluation Results

### Intrinsic

| Metric | Value | Split |
|--------|-------|-------|
| Perplexity | TBD | Held-out web |
| Bits-per-byte | TBD | Held-out web |

### Benchmark Results (few-shot, lm-evaluation-harness)

| Benchmark | Metric | Score |
|-----------|--------|-------|
| MMLU | 5-shot accuracy | TBD |
| HellaSwag | 10-shot accuracy | TBD |
| ARC-Challenge | 25-shot accuracy | TBD |
| GSM8K | 8-shot | TBD |
| HumanEval | pass@1 | TBD |

---

## Limitations & Risks

- **Hallucination:** The model may generate plausible-sounding but factually incorrect text.
- **Bias:** Training data contains societal biases that may be reflected in outputs.
- **PII leakage:** Despite PII scrubbing in pre-processing, residual personal information
  may appear in completions for rare prompts.
- **Safety:** Not aligned for deployment without SFT + RLHF/DPO alignment.

---

## Ethical Considerations

- Data sourced from public internet; license provenance tracked per document.
- Deduplication and PII scrubbing applied (see `data/prepare.py`).
- Red-teaming: [describe red-team results or link to report].

---

## Caveats & Recommendations

- Always evaluate on your domain before deployment.
- Apply instruction-tuning (SFT) and preference alignment (DPO) before user-facing deployment.
- Monitor outputs in production with Prometheus/Grafana (see `monitoring/`).

---

## Citation

```bibtex
@misc{llm-from-scratch-{{VERSION}},
  title  = {LLM from Scratch — {{MODEL_NAME}}},
  author = {rnaarla},
  year   = {{{DATE[:4]}}},
  url    = {https://github.com/rnaarla/langmodel_from_Scratch},
}
```
