# Datasheet for Dataset — {{DATASET_NAME}} ({{VERSION}})

*Template adapted from Gebru et al., "Datasheets for Datasets" (2021).*
*Generated: {{DATE}}*

---

## Motivation

**Why was this dataset created?**
To pre-train a decoder-only language model from scratch with full data provenance.

**Who funded the creation?**
[Organisation / team name]

**Who created this dataset?**
[Names / team]

---

## Composition

| Attribute | Details |
|-----------|---------|
| **Total tokens** | {{TOTAL_TOKENS}} |
| **Total documents** | {{TOTAL_DOCS}} |
| **Languages** | {{LANGUAGES}} |
| **Date range** | {{DATE_RANGE}} |
| **Format** | Parquet (cleaned) + uint16 memmap .bin (tokenized) |

### Sources

| Source | Documents | Tokens | License |
|--------|-----------|--------|---------|
| Common Crawl / FineWeb | TBD | TBD | Terms of service |
| The Stack | TBD | TBD | Various open-source |
| Books | TBD | TBD | See per-book |
| Wikipedia | TBD | TBD | CC-BY-SA 4.0 |
| ArXiv | TBD | TBD | See per-paper |
| StackExchange | TBD | TBD | CC-BY-SA 4.0 |

---

## Collection Process

**How was the data collected?**
- Web: Extracted from Common Crawl WARC files using resiliparse / trafilatura.
- Code: Cloned from GitHub repositories with permissive licenses.
- Books: [Source description].

**Over what timeframe?**
{{DATE_RANGE}}

---

## Preprocessing / Cleaning

1. Language detection (fastText lid.176) — English only.
2. Quality heuristics: min/max length, symbol ratio, average word length.
3. MinHash near-duplicate removal (Jaccard threshold 0.85, 5-gram shingles).
4. PII scrubbing: email, phone, SSN via regex (see `data/prepare.py`).
5. Unicode NFC normalisation.
6. Tokenisation: ByteLevel BPE 32k, stored as uint16 memmap shards.

---

## Uses

**Intended uses:**
- Pre-training language models for research and enterprise NLP applications.

**Out-of-scope uses:**
- This dataset must not be used to train models for generating illegal content,
  targeted harassment, or systems that violate applicable laws.

---

## Distribution

**How will the dataset be distributed?**
Via DVC (Data Version Control) with a remote S3 bucket. Access controlled by IAM.

**Are there export controls?**
[Describe any applicable restrictions.]

---

## Maintenance

**Who will maintain this dataset?**
[Team name], reachable at [contact].

**How will updates be communicated?**
Via git tags and CHANGELOG in the repository.

---

## Legal & Ethical

**Does the dataset contain PII?**
PII has been scrubbed via regex patterns. Residual PII risk is low but not zero.
Contact [team] for deletion requests (GDPR/CCPA).

**Were subjects informed?**
Data was sourced from public internet resources; individual consent was not obtained.

**Has the dataset been reviewed for bias?**
[Describe bias audit results or indicate pending.]

---

## Dataset Snapshot

| Attribute | Value |
|-----------|-------|
| Dataset SHA | {{DATASET_SHA}} |
| DVC remote | {{DVC_REMOTE}} |
| Created | {{DATE}} |
| Tokenizer version | {{TOKENIZER_VERSION}} |
