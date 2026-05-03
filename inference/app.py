"""FastAPI inference gateway for the LLM serving layer.

Exposes three endpoints:
* ``POST /generate``  — text generation (proxies to a vLLM backend or runs a local model).
* ``GET  /healthz``   — liveness probe.
* ``GET  /metrics``   — Prometheus metrics (via ``prometheus-fastapi-instrumentator``).

Environment variables
---------------------
``VLLM_BASE_URL``
    If set, requests are forwarded to the vLLM OpenAI-compatible API running at this URL.
    Example: ``http://localhost:8000``.
``MODEL_DIR``
    Path to a local ``TransformerLM`` checkpoint directory (used when vLLM is not available).
``MAX_NEW_TOKENS``
    Default maximum number of tokens to generate (default: 256).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Inference Gateway",
    description="Production inference gateway for the LLM-from-scratch pipeline.",
    version="0.1.0",
)

# Prometheus instrumentation (optional — gracefully skipped if library absent)
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # noqa: PLC0415

    Instrumentator().instrument(app).expose(app)
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed; /metrics unavailable.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt.")
    max_new_tokens: int = Field(256, ge=1, le=4096, description="Maximum tokens to generate.")
    temperature: float = Field(1.0, ge=0.0, le=4.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    stream: bool = Field(False, description="Streaming not yet implemented.")


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    model: str


# ---------------------------------------------------------------------------
# Model / backend loader (lazy, on first request)
# ---------------------------------------------------------------------------

_local_model = None
_local_tokenizer = None


def _load_local_model():
    global _local_model, _local_tokenizer  # noqa: PLW0603
    if _local_model is not None:
        return

    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    from model.architecture import TransformerLM  # noqa: PLC0415
    from model.checkpoint import find_latest_checkpoint, load_checkpoint  # noqa: PLC0415
    from model.config import ModelConfig  # noqa: PLC0415

    model_dir = Path(os.environ.get("MODEL_DIR", "checkpoints/final"))
    tok_dir = Path(os.environ.get("TOKENIZER_DIR", "tokenizer/artifacts"))

    cfg = ModelConfig.tiny()  # TODO: load from model_dir/config.yaml
    model = TransformerLM.from_config(cfg)
    latest = find_latest_checkpoint(model_dir)
    if latest:
        load_checkpoint(latest, model, device="cpu")
    model.eval()
    _local_model = model

    if (tok_dir / "vocab.json").exists():
        _local_tokenizer = ByteLevelBPETokenizer(
            str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt")
        )


def _generate_local(req: GenerateRequest) -> str:
    """Run greedy / top-p sampling on the local model (CPU fallback)."""
    _load_local_model()
    assert _local_tokenizer is not None  # noqa: S101

    ids = _local_tokenizer.encode(req.prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(req.max_new_tokens):
            logits = _local_model(input_ids)  # (1, T, V)
            next_logits = logits[0, -1, :] / max(req.temperature, 1e-6)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]])], dim=1
            )

    generated_ids = input_ids[0, len(ids) :].tolist()
    return _local_tokenizer.decode(generated_ids)


async def _generate_vllm(req: GenerateRequest, base_url: str) -> str:
    """Forward the request to a vLLM OpenAI-compatible endpoint."""
    import httpx  # noqa: PLC0415

    payload = {
        "prompt": req.prompt,
        "max_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{base_url}/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz", summary="Liveness probe")
def healthz():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse, summary="Text generation")
async def generate(req: GenerateRequest):
    vllm_url = os.environ.get("VLLM_BASE_URL", "")
    try:
        if vllm_url:
            text = await _generate_vllm(req, vllm_url)
            model_name = "vllm-proxy"
        else:
            text = _generate_local(req)
            model_name = os.environ.get("MODEL_DIR", "local")
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(
        text=text,
        tokens_generated=len(text.split()),  # approximate
        model=model_name,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")  # noqa: S104
