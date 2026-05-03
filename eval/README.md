# Evaluation

This directory contains tools for evaluating the pre-trained base model, both intrinsically
(perplexity, bits-per-byte) and extrinsically (standard NLP benchmarks via lm-evaluation-harness).

---

## Intrinsic Evaluation (`perplexity.py`)

Computes **perplexity** and **bits-per-byte** on a held-out binary shard:

```bash
python eval/perplexity.py \
    --checkpoint checkpoints/step_200000.pt \
    --data-file data/eval/held_out.bin \
    --batch-size 4
```

These metrics are useful for:
- Comparing model quality across checkpoints during training.
- Comparing models with the same tokenizer on the same held-out split.

**Note:** Perplexity is tokenizer-dependent.  Use **bits-per-byte** (BPB) for
cross-tokenizer or cross-model comparisons.

---

## Benchmark Evaluation (`run_harness.sh`)

Uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for
standardised few-shot benchmark evaluation:

```bash
bash eval/run_harness.sh \
    ./checkpoints/step_200000 \
    "mmlu,hellaswag,arc_challenge,gsm8k,humaneval"
```

Recommended benchmark suite:

| Benchmark | Shots | What it measures |
|-----------|-------|------------------|
| MMLU | 5 | Knowledge breadth (57 subjects) |
| HellaSwag | 10 | Commonsense reasoning |
| ARC-Challenge | 25 | Science question answering |
| GSM8K | 8 | Elementary math reasoning |
| HumanEval | 0 | Code generation (pass@1) |
| MBPP | 3 | Python programming |
| TriviaQA | 5 | Factual recall |

---

## Enterprise Considerations

- **Reproducibility:** Always log benchmark results with the exact checkpoint step, git SHA,
  and dataset SHA in MLflow.  This enables tracking of emergent capabilities over training.
- **Regression detection:** The nightly CI workflow (`.github/workflows/eval.yml`) runs
  lm-eval-harness on the latest checkpoint and alerts if scores drop below a threshold.
- **Evaluation contamination:** Ensure held-out eval sets were excluded from training data
  at the deduplication stage (`data/prepare.py` MinHash dedup).
- **Benchmark limitations:** No single benchmark captures all capabilities.  Complement
  automated benchmarks with human evaluation (red-teaming, preference studies) before
  any production deployment.
