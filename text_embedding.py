"""
Caption generation (BLIP) + text embedding (Ollama).

Pipeline:
  1. BLIP generates a natural-language caption for every image.
  2. Ollama (nomic-embed-text / all-minilm) converts each caption into a
     dense vector.
  3. Embeddings are L2-normalised for consistent downstream fusion.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from config import CFG, CaptionConfig, EmbeddingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BLIP caption generation
# ---------------------------------------------------------------------------


def load_blip_model(
    cfg: Optional[CaptionConfig] = None,
    device: Optional[torch.device] = None,
) -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    """Download (first run) and load the BLIP captioning model.

    Args:
        cfg: Caption configuration; defaults to ``CFG.captions``.
        device: Torch device for inference; auto-selected if *None*.

    Returns:
        ``(processor, model)`` tuple ready for inference.
    """
    cfg = cfg or CFG.captions

    if device is None:
        device = _select_device()

    logger.info("Loading BLIP model '%s'...", cfg.model_name)
    processor = BlipProcessor.from_pretrained(cfg.model_name)
    model = BlipForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    logger.info("BLIP model loaded on %s.", device)
    return processor, model


def generate_captions(
    image_paths: list[Path],
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    cfg: Optional[CaptionConfig] = None,
    device: Optional[torch.device] = None,
) -> list[str]:
    """Generate one caption per image using BLIP.

    Args:
        image_paths: Paths to images on disk.
        processor: Loaded BLIP processor.
        model: Loaded BLIP model.
        cfg: Caption configuration; defaults to ``CFG.captions``.
        device: Torch device; inferred from *model* if *None*.

    Returns:
        List of caption strings, one per image (same order as *image_paths*).
    """
    cfg = cfg or CFG.captions
    if device is None:
        device = next(model.parameters()).device

    captions: list[str] = []

    # Process images in mini-batches to stay within GPU VRAM limits
    for batch_start in range(0, len(image_paths), cfg.batch_size):
        batch_paths = image_paths[batch_start : batch_start + cfg.batch_size]
        raw_images: list[Image.Image] = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                raw_images.append(img)
            except (OSError, ValueError) as exc:
                logger.warning("Cannot open '%s': %s – using blank image.", path, exc)
                raw_images.append(Image.new("RGB", (224, 224)))

        # Tokenise images into pixel_values tensor
        inputs = processor(
            images=raw_images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                num_beams=cfg.num_beams,
            )

        batch_captions = processor.batch_decode(output_ids, skip_special_tokens=True)
        captions.extend(batch_captions)
        logger.debug(
            "Generated captions for batch %d–%d.",
            batch_start,
            batch_start + len(batch_paths) - 1,
        )

    logger.info("Generated %d captions.", len(captions))
    return captions


# ---------------------------------------------------------------------------
# Ollama text embeddings
# ---------------------------------------------------------------------------


def check_ollama_health(cfg: Optional[EmbeddingConfig] = None) -> bool:
    """Verify that the Ollama server is reachable.

    Args:
        cfg: Embedding configuration; defaults to ``CFG.embeddings``.

    Returns:
        ``True`` if the server responds with HTTP 200, ``False`` otherwise.
    """
    cfg = cfg or CFG.embeddings
    try:
        resp = requests.get(cfg.base_url, timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _embed_single(
    text: str,
    cfg: EmbeddingConfig,
    session: requests.Session,
) -> Optional[list[float]]:
    """Request an embedding vector for a single text from Ollama.

    Args:
        text: Caption or query string.
        cfg: Embedding configuration.
        session: Persistent HTTP session for connection re-use.

    Returns:
        A list of floats representing the embedding, or *None* on failure.
    """
    url = f"{cfg.base_url}/api/embeddings"
    payload = {"model": cfg.model, "prompt": text}

    for attempt in range(cfg.max_retries):
        try:
            resp = session.post(url, json=payload, timeout=cfg.timeout_seconds)
            resp.raise_for_status()
            return resp.json()["embedding"]
        except (requests.RequestException, KeyError) as exc:
            wait = 2**attempt  # exponential back-off
            logger.warning(
                "Embedding attempt %d/%d failed (%s). Retrying in %ds.",
                attempt + 1,
                cfg.max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    logger.error("All retries exhausted for text: '%s'", text[:80])
    return None


def embed_captions(
    captions: list[str],
    cfg: Optional[EmbeddingConfig] = None,
) -> tuple[np.ndarray, list[int]]:
    """Embed a list of captions via Ollama and L2-normalise the result.

    Args:
        captions: Caption strings to embed.
        cfg: Embedding configuration; defaults to ``CFG.embeddings``.

    Returns:
        A tuple ``(embeddings, valid_indices)`` where *embeddings* has shape
        ``(M, embed_dim)`` and *valid_indices* lists the positions in *captions*
        for which embedding succeeded (M ≤ len(captions)).

    Raises:
        RuntimeError: If Ollama is not reachable.
        ValueError: If *captions* is empty.
    """
    if not captions:
        raise ValueError("captions list must not be empty.")

    cfg = cfg or CFG.embeddings

    if not check_ollama_health(cfg):
        raise RuntimeError(
            f"Ollama server not reachable at '{cfg.base_url}'. "
            "Start it with: `ollama serve`"
        )

    logger.info("Embedding %d captions with model '%s'...", len(captions), cfg.model)

    valid_embeddings: list[list[float]] = []
    valid_indices: list[int] = []

    # Use a persistent session to avoid TCP handshake overhead per request
    with requests.Session() as session:
        for idx, caption in enumerate(captions):
            vec = _embed_single(caption, cfg, session)
            if vec is not None:
                valid_embeddings.append(vec)
                valid_indices.append(idx)
            else:
                logger.warning("Skipping caption at index %d (embedding failed).", idx)

    if not valid_embeddings:
        raise RuntimeError("No embeddings could be obtained from Ollama.")

    embeddings = np.array(valid_embeddings, dtype=np.float32)

    # L2 normalise: each row becomes a unit vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for degenerate zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    logger.info(
        "Text embeddings shape: %s  (%d caption(s) failed).",
        embeddings.shape,
        len(captions) - len(valid_indices),
    )
    return embeddings, valid_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    """Return the best available Torch device.

    Returns:
        ``torch.device`` for CUDA, MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
