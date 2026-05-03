# Data Pipeline

This directory contains scripts to transform raw text corpora into tokenized memory-mapped
shards suitable for streaming during large-scale pre-training.

---

## Stage Overview

| Script | Input | Output |
|--------|-------|--------|
| `prepare.py` | `data/raw/` (txt, WARC, Parquet) | `data/cleaned/` (Parquet) |
| `shard.py` | `data/cleaned/` + tokenizer | `data/tokenized/*.bin` + `index.json` |

---

## Directory Structure

```
data/
├── raw/          ← Unprocessed source corpora (never committed — add to .gitignore)
├── cleaned/      ← Deduplicated, filtered, PII-scrubbed Parquet files
├── tokenized/    ← uint16/uint32 memmap .bin shards + index.json
└── eval/         ← Held-out evaluation sets (.bin + optional JSONL)
```

---

## Data Preparation (`prepare.py`)

Runs a CCNet-style pipeline:

1. **Language detection** — keep only the target language (fastText-based, falls back to
   `langdetect`).
2. **Quality heuristics** — filter on document length, symbol ratio, and average word length.
3. **MinHash near-duplicate removal** — Jaccard similarity threshold 0.85 on 5-gram shingles,
   using `datasketch`.  For production, replace the in-memory LSH with a distributed Redis-backed
   index that spans all shards.
4. **PII scrubbing** — regex-based removal of emails, phone numbers, and SSNs.  TODO: upgrade to
   `presidio-analyzer` for NER-based PII detection in production.
5. **Unicode NFC normalisation** and control-character stripping.

```bash
python data/prepare.py \
    --input-dir data/raw \
    --output-dir data/cleaned \
    --lang en \
    --min-length 200 \
    --dedup-threshold 0.85
```

---

## Sharding (`shard.py`)

Tokenises the cleaned corpus with the trained ByteLevel BPE tokenizer and writes fixed-size
binary shards:

```bash
python data/shard.py \
    --input-dir data/cleaned \
    --tokenizer-dir tokenizer/artifacts \
    --output-dir data/tokenized \
    --shard-size 500000000   # tokens per shard
```

Shards are read during training by `model/train.py` via `numpy.memmap` — no intermediate
copying, enabling training on datasets larger than RAM.

---

## Enterprise Considerations

- **Versioning:** Use DVC (`dvc add data/tokenized`) to snapshot and reproduce datasets.
  Pin every training run to a `dataset_sha` logged in MLflow.
- **Lineage:** Each cleaned document carries `doc_id` and `source_url` columns for audit.
- **Security:** `data/raw/` and `data/cleaned/` should never be committed to git — they are
  listed in `.gitignore`.  Store in encrypted S3 with bucket versioning enabled.
- **GDPR/CCPA:** Support deletion requests by filtering `doc_id` before re-sharding.
- **Reproducibility:** Record the exact `prepare.py` CLI invocation and git SHA in your
  MLflow run so any dataset can be regenerated.
