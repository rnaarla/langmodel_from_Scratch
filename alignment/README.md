# Alignment

This directory implements the alignment stage: transforming a capable but raw base model into
a helpful, harmless, and honest assistant through supervised fine-tuning (SFT) and direct
preference optimisation (DPO).

---

## Why Alignment?

A pre-trained base model is a next-token predictor over the training distribution.  It does
not understand instructions, maintain consistent persona, or refuse harmful requests.
Alignment bridges this gap through two sequential steps:

1. **SFT (`sft.py`)** — teach the model to follow the instruction format by training on
   `(instruction, input, output)` triples with the loss masked to response tokens only.
2. **DPO (`dpo.py`)** — refine preferences without a separate reward model by training
   on `(prompt, chosen, rejected)` pairs against a frozen reference policy.

---

## Supervised Fine-Tuning

```bash
python alignment/sft.py \
    --model-dir checkpoints/pretrained \
    --data-path data/eval/instructions.jsonl \
    --output-dir checkpoints/sft \
    --epochs 3 --lr 2e-5
```

Dataset format (JSONL):
```json
{"instruction": "Summarise the following text.", "input": "...", "output": "..."}
```

Key implementation detail: the loss is **masked** so only response tokens contribute to the
gradient.  This prevents the model from being penalised for the prompt tokens, which are
deterministic given the instruction format.

---

## Direct Preference Optimisation

```bash
python alignment/dpo.py \
    --model-dir checkpoints/sft \
    --data-path data/eval/preferences.jsonl \
    --output-dir checkpoints/dpo \
    --beta 0.1 --epochs 1
```

Dataset format (JSONL):
```json
{"prompt": "Explain quantum entanglement.", "chosen": "...", "rejected": "..."}
```

The `--beta` parameter controls the KL divergence penalty against the reference policy.
Lower values allow larger deviation from the SFT model; higher values stay closer.

---

## Enterprise Considerations

- **Data quality:** The quality of preference data dominates alignment outcomes.  Invest in
  clear annotation guidelines and measure inter-annotator reliability (Cohen's κ ≥ 0.7).
- **Safety tuning:** Red-team the SFT model before DPO; build a refusal dataset for
  harmful categories and include it in SFT training.
- **Lineage:** Track annotator IDs, guideline versions, and preference data SHA in MLflow
  alongside model checkpoints.
- **Overfitting:** SFT models can overfit to instruction format quickly.  Monitor held-out
  PPL on general text during SFT; stop early if it rises significantly.
- **RLHF vs DPO:** DPO is simpler (no separate reward model or PPO loop) and increasingly
  preferred for smaller teams.  For frontier-scale alignment, consider PPO with a trained
  reward model + constitutional AI critique-and-revise.
