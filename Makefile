# LLM-from-Scratch Pipeline Makefile
# Run `make help` to see all available targets.

PYTHON      ?= python
PIP         ?= pip
TORCH_CPU   ?= https://download.pytorch.org/whl/cpu

# Directories
DATA_DIR        ?= data/tokenized
CLEANED_DIR     ?= data/cleaned
RAW_DIR         ?= data/raw
TOK_DIR         ?= tokenizer/artifacts
CKPT_DIR        ?= checkpoints/run
TINY_CKPT_DIR   ?= checkpoints/tiny
EVAL_CKPT       ?= checkpoints/run

.PHONY: help install lint test tokenizer prepare-data shard \
        train-tiny train eval serve docker-build \
        helm-install tf-plan tf-apply clean

## Show this help message
help:
	@grep -E '^##[[:space:]]' $(MAKEFILE_LIST) | sed 's/^## //'
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## ── Setup ──────────────────────────────────────────────────────────────────
install: ## Install all Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url $(TORCH_CPU)
	$(PIP) install -r requirements.txt

## ── Quality ─────────────────────────────────────────────────────────────────
lint: ## Run ruff linter on all Python source files
	ruff check model/ alignment/ eval/ tokenizer/ data/ inference/ --line-length 100

test: ## Run unit tests (model architecture + tokenizer round-trip)
	pytest model/tests/ -v --tb=short

## ── Data pipeline ────────────────────────────────────────────────────────────
prepare-data: ## Run data cleaning pipeline (data/prepare.py)
	$(PYTHON) data/prepare.py \
		--input-dir $(RAW_DIR) \
		--output-dir $(CLEANED_DIR) \
		--lang en

shard: ## Tokenize cleaned data and write memmap shards (data/shard.py)
	$(PYTHON) data/shard.py \
		--input-dir $(CLEANED_DIR) \
		--tokenizer-dir $(TOK_DIR) \
		--output-dir $(DATA_DIR)

## ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer: ## Train ByteLevel BPE tokenizer on data/cleaned/*.txt
	$(PYTHON) tokenizer/train_tokenizer.py \
		--input-dir $(CLEANED_DIR) \
		--vocab-size 32000 \
		--output-dir $(TOK_DIR)

## ── Training ─────────────────────────────────────────────────────────────────
train-tiny: ## Smoke-train a tiny model for 100 steps (no GPU, no data required)
	$(PYTHON) model/train.py \
		--checkpoint-dir $(TINY_CKPT_DIR) \
		--batch-size 2 \
		--total-steps 100 \
		--warmup-steps 10 \
		--log-interval 10 \
		--checkpoint-interval 50

train: ## Full pre-training run (configure via --config)
	torchrun --nproc_per_node=$(or $(NPROC),1) model/train.py \
		--config config/pretrain_125m.yaml \
		--data-dir $(DATA_DIR) \
		--checkpoint-dir $(CKPT_DIR)

## ── Evaluation ───────────────────────────────────────────────────────────────
eval: ## Compute perplexity on held-out data
	$(PYTHON) eval/perplexity.py \
		--data-file data/eval/held_out.bin \
		--checkpoint $(EVAL_CKPT)

## ── Serving ──────────────────────────────────────────────────────────────────
serve: ## Start FastAPI inference gateway locally
	uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload

docker-build: ## Build the inference Docker image
	docker build -t llm-inference:latest -f inference/Dockerfile .

## ── Kubernetes ───────────────────────────────────────────────────────────────
helm-install: ## Deploy inference chart to current kubectl context
	helm upgrade --install llm-api helm/llm-inference-chart/ \
		--set image.tag=latest

## ── Terraform ────────────────────────────────────────────────────────────────
tf-plan: ## Preview Terraform infrastructure changes
	cd terraform && terraform plan

tf-apply: ## Apply Terraform infrastructure changes
	cd terraform && terraform apply

## ── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Remove cached artifacts and temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	@echo "Clean complete."
