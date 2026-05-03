# Tokenizer

This directory contains the tokenizer training script and, after training, the artefact files
(`vocab.json`, `merges.txt`) that the rest of the pipeline depends on.

---

## Algorithm Choice

This pipeline uses **ByteLevel BPE** (Byte-Pair Encoding on raw bytes), the same algorithm
used by GPT-2, GPT-3, and LLaMA:

- **No UNK tokens** — every possible byte sequence can be encoded.
- **Robust to any input** — handles code, math symbols, emoji, and non-Latin scripts.
- **Good English efficiency** — typically 3.5–4.5 bytes per token.

Alternative choices:
- **SentencePiece Unigram** (T5, mBART) — better for multilingual corpora.
- **tiktoken** (GPT-4) — faster encoding via Rust, but training interface is different.

---

## Training

```bash
python tokenizer/train_tokenizer.py \
    --input-dir data/cleaned \
    --vocab-size 32000 \
    --output-dir tokenizer/artifacts \
    --min-frequency 2
```

Or via the Makefile shortcut:

```bash
make tokenizer
```

Artefacts are saved to `tokenizer/artifacts/`:
- `vocab.json` — token → ID mapping
- `merges.txt` — BPE merge rules

---

## Validation

After training, the script prints the **bytes-per-token** on a sample of the training corpus.
The target range for English text is **3.5–4.5 bytes/token**.  Values outside this range suggest:

- Too low (< 3): very large vocabulary or short training corpus — many rare merge rules.
- Too high (> 5): vocabulary too small — many characters are not merged into meaningful subwords.

Coverage checks to run manually:
```python
ids = tokenizer.encode("def fibonacci(n):").ids
ids = tokenizer.encode("∫ f(x) dx = F(b) − F(a)").ids
ids = tokenizer.encode("Привет мир").ids  # Russian: check non-Latin coverage
```

---

## Enterprise Considerations

- **Versioning:** Commit `tokenizer/artifacts/` under version control or store in your
  artefact registry alongside the model checkpoint.  A model checkpoint is meaningless
  without the exact tokenizer it was trained with.
- **Vocabulary expansion:** Adding tokens post-hoc requires careful embedding surgery and
  re-training of the embedding layer — plan vocabulary size carefully upfront.
- **Multi-lingual:** For multilingual corpora, sample training files proportionally across
  languages; under-represented languages will have poor tokenisation efficiency.
- **Reproducibility:** Pin the `tokenizers` library version in `requirements.txt` — BPE
  training outcomes can differ across library versions.
