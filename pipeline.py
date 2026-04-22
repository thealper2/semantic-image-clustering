"""
Pipeline orchestrator and single-image inference.

``run_pipeline`` wires together every stage:
  feature extraction → caption generation → text embedding →
  fusion → PCA → clustering → metrics → visualisations.

``infer_cluster`` runs a trained pipeline on a new, unseen image.
"""

import json
import logging
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize

from clustering import (
    ClusterMetrics,
    ClusterResult,
    PCAResult,
    compute_metrics,
    fuse_features,
    reduce_dimensions,
    run_clustering,
)
from config import CFG, AppConfig
from feature_extraction import discover_images, extract_visual_features
from text_embedding import embed_captions, generate_captions, load_blip_model
from visualization import save_all_plots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline state – everything needed to run inference later
# ---------------------------------------------------------------------------


def save_pipeline_state(
    image_paths: list[str],
    visual_features: np.ndarray,
    captions: list[str],
    text_embeddings: np.ndarray,
    valid_indices: list[int],
    pca_result: PCAResult,
    cluster_result: ClusterResult,
    metrics: ClusterMetrics,
    output_dir: Path,
) -> Path:
    """Persist all pipeline artefacts needed for inference.

    Args:
        image_paths: Ordered list of image path strings.
        visual_features: Raw visual features ``(N, visual_dim)``.
        captions: Generated captions, one per image.
        text_embeddings: Text embedding vectors ``(M, text_dim)``.
        valid_indices: Indices in ``image_paths`` that have text embeddings.
        pca_result: Fitted PCA result.
        cluster_result: Clustering result (labels, algorithm, …).
        metrics: Clustering quality metrics.
        output_dir: Directory to write artefacts.

    Returns:
        Path to the saved state pickle file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "image_paths": image_paths,
        "visual_features": visual_features,
        "captions": captions,
        "text_embeddings": text_embeddings,
        "valid_indices": valid_indices,
        "pca_result": pca_result,
        "cluster_result": cluster_result,
    }
    state_path = output_dir / "pipeline_state.pkl"
    with open(state_path, "wb") as fh:
        pickle.dump(state, fh)

    # Also write human-readable metrics as JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(asdict(metrics), fh, indent=2)

    logger.info("Pipeline state saved → %s", state_path)
    logger.info("Metrics saved → %s", metrics_path)
    return state_path


def load_pipeline_state(state_path: Path) -> dict:
    """Load a previously saved pipeline state.

    Args:
        state_path: Path to the ``.pkl`` file written by
            :func:`save_pipeline_state`.

    Returns:
        Dictionary containing all pipeline artefacts.

    Raises:
        FileNotFoundError: If *state_path* does not exist.
    """
    if not state_path.exists():
        raise FileNotFoundError(f"Pipeline state not found: {state_path}")
    with open(state_path, "rb") as fh:
        state = pickle.load(fh)
    logger.info("Pipeline state loaded from '%s'.", state_path)
    return state


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    image_dir: Optional[Path] = None,
    cfg: Optional[AppConfig] = None,
) -> tuple[ClusterResult, ClusterMetrics, dict[str, Path]]:
    """Execute the complete Semantic Image Clustering pipeline end-to-end.

    Stages:
        1. Discover images in *image_dir*.
        2. Extract visual features (ResNet50 / EfficientNet-B4).
        3. Generate captions (BLIP).
        4. Embed captions (Ollama).
        5. Fuse visual + text vectors.
        6. Reduce dimensions (PCA).
        7. Cluster (HDBSCAN / K-Means++).
        8. Compute metrics.
        9. Save plots and artefacts.

    Args:
        image_dir: Root directory containing images; defaults to
            ``cfg.paths.data_dir``.
        cfg: Full application configuration; defaults to ``CFG``.

    Returns:
        Tuple of ``(ClusterResult, ClusterMetrics, plot_paths)``.

    Raises:
        FileNotFoundError: If *image_dir* does not exist.
        ValueError: If fewer than 2 images are found.
    """
    cfg = cfg or CFG
    image_dir = image_dir or cfg.paths.data_dir

    overall_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Semantic Image Clustering Pipeline  START")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1 – Discover images
    # ------------------------------------------------------------------
    logger.info("[1/8] Discovering images in '%s'…", image_dir)
    image_paths = discover_images(image_dir)
    if len(image_paths) < 2:
        raise ValueError(
            f"Need at least 2 images; found {len(image_paths)} in '{image_dir}'."
        )
    logger.info("Found %d image(s).", len(image_paths))

    # ------------------------------------------------------------------
    # Stage 2 – Visual feature extraction
    # ------------------------------------------------------------------
    logger.info("[2/8] Extracting visual features (%s)…", cfg.features.backbone)
    t0 = time.perf_counter()
    visual_features, ordered_paths = extract_visual_features(image_paths, cfg.features)
    logger.info("Visual extraction done in %.1fs.", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Stage 3 – Caption generation (BLIP)
    # ------------------------------------------------------------------
    logger.info("[3/8] Generating captions (BLIP)…")
    t0 = time.perf_counter()
    processor, blip_model = load_blip_model(cfg.captions)
    path_objects = [Path(p) for p in ordered_paths]
    captions = generate_captions(path_objects, processor, blip_model, cfg.captions)
    # Free GPU memory immediately after captioning
    del blip_model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Caption generation done in %.1fs.", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Stage 4 – Text embedding (Ollama)
    # ------------------------------------------------------------------
    logger.info("[4/8] Embedding captions (%s)…", cfg.embeddings.model)
    t0 = time.perf_counter()
    text_embeddings, valid_indices = embed_captions(captions, cfg.embeddings)
    logger.info("Text embedding done in %.1fs.", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Stage 5 – Fusion
    # ------------------------------------------------------------------
    logger.info("[5/8] Fusing visual + text features…")
    fusion_result = fuse_features(
        visual_features, text_embeddings, valid_indices, cfg.fusion
    )

    # Keep only the paths that have matching text embeddings
    valid_paths = [ordered_paths[i] for i in valid_indices]

    # ------------------------------------------------------------------
    # Stage 6 – PCA
    # ------------------------------------------------------------------
    logger.info("[6/8] Reducing dimensions (PCA)…")
    pca_result = reduce_dimensions(fusion_result.fused, cfg.dim_reduction)

    # ------------------------------------------------------------------
    # Stage 7 – Clustering
    # ------------------------------------------------------------------
    logger.info("[7/8] Clustering (%s)…", cfg.clustering.algorithm)
    cluster_result = run_clustering(pca_result.reduced, cfg.clustering)

    # ------------------------------------------------------------------
    # Stage 8 – Metrics + Plots
    # ------------------------------------------------------------------
    logger.info("[8/8] Computing metrics and generating plots…")
    metrics = compute_metrics(pca_result.reduced, cluster_result.labels)

    plot_paths = save_all_plots(
        features_reduced=pca_result.reduced,
        pca_result=pca_result,
        cluster_result=cluster_result,
        metrics=metrics,
        image_paths=valid_paths,
        output_dir=cfg.paths.plots_dir,
        cfg=cfg.visualization,
    )

    save_pipeline_state(
        image_paths=valid_paths,
        visual_features=visual_features[valid_indices],
        captions=[captions[i] for i in valid_indices],
        text_embeddings=text_embeddings,
        valid_indices=valid_indices,
        pca_result=pca_result,
        cluster_result=cluster_result,
        metrics=metrics,
        output_dir=cfg.paths.output_dir,
    )

    elapsed = time.perf_counter() - overall_start
    logger.info("=" * 60)
    logger.info("Pipeline COMPLETE in %.1fs", elapsed)
    logger.info("  Clusters   : %d", metrics.n_clusters)
    logger.info(
        "  Noise pts  : %d (%.1f %%)", metrics.n_noise, metrics.noise_ratio * 100
    )
    logger.info("  Silhouette : %.4f", metrics.silhouette)
    logger.info("  DB Index   : %.4f", metrics.davies_bouldin)
    logger.info("  CH Score   : %.2f", metrics.calinski_harabasz)
    logger.info("=" * 60)

    return cluster_result, metrics, plot_paths


# ---------------------------------------------------------------------------
# Inference on a new image
# ---------------------------------------------------------------------------


def infer_cluster(
    image_path: Path,
    state_path: Optional[Path] = None,
    cfg: Optional[AppConfig] = None,
) -> dict:
    """Predict the cluster for a single unseen image.

    The image is passed through:
      1. Visual feature extraction (same backbone as training).
      2. BLIP caption generation.
      3. Ollama text embedding.
      4. Fusion.
      5. Trained PCA transform.
      6. Nearest-centroid assignment (Euclidean distance to cluster centroids).

    Args:
        image_path: Path to the query image.
        state_path: Path to the saved pipeline state pickle.
            Defaults to ``cfg.paths.output_dir / "pipeline_state.pkl"``.
        cfg: Application config; defaults to ``CFG``.

    Returns:
        Dictionary with keys:
            ``cluster_id``, ``distance``, ``caption``, ``image_path``.

    Raises:
        FileNotFoundError: If *image_path* or *state_path* does not exist.
    """
    cfg = cfg or CFG
    state_path = state_path or (cfg.paths.output_dir / "pipeline_state.pkl")

    if not image_path.exists():
        raise FileNotFoundError(f"Query image not found: {image_path}")

    state = load_pipeline_state(state_path)
    pca_result: PCAResult = state["pca_result"]
    cluster_result: ClusterResult = state["cluster_result"]
    stored_features: np.ndarray = state["visual_features"]

    # --- Visual feature for the new image ---
    visual_feats, _ = extract_visual_features([image_path], cfg.features)

    # --- Caption + embedding ---
    processor, blip_model = load_blip_model(cfg.captions)
    captions = generate_captions([image_path], processor, blip_model, cfg.captions)
    del blip_model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    text_embs, valid_idx = embed_captions(captions, cfg.embeddings)
    if not valid_idx:
        raise RuntimeError("Text embedding failed for the query image.")

    # --- Fusion (identical to training) ---
    fusion_result = fuse_features(visual_feats, text_embs, [0], cfg.fusion)

    # --- PCA transform (apply fitted PCA – do NOT refit) ---
    reduced_query = pca_result.pca.transform(fusion_result.fused)  # (1, d)

    # --- Nearest-centroid assignment ---
    # Compute centroid of each non-noise cluster from training data
    stored_reduced = pca_result.pca.transform(
        np.hstack(
            [
                normalize(stored_features, norm="l2") * cfg.fusion.visual_weight,
                normalize(state["text_embeddings"], norm="l2") * cfg.fusion.text_weight,
            ]
        )
    )
    unique_labels = np.unique(cluster_result.labels)
    centroids = {}
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = cluster_result.labels == lbl
        centroids[lbl] = stored_reduced[mask].mean(axis=0)

    dists = {
        lbl: float(np.linalg.norm(reduced_query[0] - centroid))
        for lbl, centroid in centroids.items()
    }
    best_cluster = min(dists, key=dists.__getitem__)

    logger.info(
        "Inference → cluster %d (distance: %.4f) | caption: '%s'",
        best_cluster,
        dists[best_cluster],
        captions[0],
    )
    return {
        "cluster_id": int(best_cluster),
        "distance": dists[best_cluster],
        "all_distances": dists,
        "caption": captions[0],
        "image_path": str(image_path),
    }
