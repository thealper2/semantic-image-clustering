"""
Multimodal fusion, dimensionality reduction, and clustering.

Pipeline:
  1. Fusion  – Weighted, normalised concatenation of visual + text vectors.
  2. PCA     – Reduce fused vectors to ≤100 dimensions (≥95 % variance).
  3. Cluster – HDBSCAN (density-based, auto-k) or K-Means++ (fixed-k).
  4. Metrics – Silhouette, Davies-Bouldin, Calinski-Harabasz.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from config import CFG, ClusteringConfig, DimensionReductionConfig, FusionConfig

logger = logging.getLogger(__name__)

# hdbscan is an optional dependency – import lazily so the rest of the module
# is usable even if the package is not installed (K-Means path still works).
try:
    import hdbscan as _hdbscan_module

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning(
        "hdbscan package not found. Install with `pip install hdbscan`. "
        "Falling back to K-Means."
    )


# ---------------------------------------------------------------------------
# Result containers (dataclasses instead of plain dicts for type safety)
# ---------------------------------------------------------------------------


@dataclass
class FusionResult:
    """Output of the multimodal fusion step."""

    fused: np.ndarray  # shape (N, visual_dim + text_dim)
    visual_dim: int
    text_dim: int


@dataclass
class PCAResult:
    """Output of the PCA dimensionality reduction step."""

    reduced: np.ndarray  # shape (N, n_components)
    pca: PCA  # fitted sklearn PCA object
    explained_variance_ratio: np.ndarray
    n_components: int
    cumulative_variance: float


@dataclass
class ClusterResult:
    """Output of the clustering step."""

    labels: np.ndarray  # shape (N,), -1 = noise (HDBSCAN only)
    n_clusters: int
    n_noise: int  # number of points labelled -1
    algorithm: str
    probabilities: Optional[np.ndarray]  # soft membership (HDBSCAN only)


@dataclass
class ClusterMetrics:
    """Intrinsic clustering quality metrics."""

    silhouette: float  # [-1, 1]  higher is better
    davies_bouldin: float  # [0, ∞)   lower is better
    calinski_harabasz: float  # [0, ∞)   higher is better
    n_clusters: int
    n_noise: int
    noise_ratio: float  # fraction of points labelled as noise


# ---------------------------------------------------------------------------
# 1. Fusion
# ---------------------------------------------------------------------------


def fuse_features(
    visual_features: np.ndarray,
    text_embeddings: np.ndarray,
    valid_indices: list[int],
    cfg: Optional[FusionConfig] = None,
) -> FusionResult:
    """Weighted concatenation of visual and text feature vectors.

    Only the images for which text embeddings were successfully obtained
    (tracked via *valid_indices*) are included in the fused output.

    Args:
        visual_features: Array of shape ``(N_total, visual_dim)``.
        text_embeddings: Array of shape ``(M, text_dim)`` where M ≤ N_total.
        valid_indices: Positions in ``visual_features`` that correspond to
            rows in ``text_embeddings`` (output of :func:`embed_captions`).
        cfg: Fusion configuration; defaults to ``CFG.fusion``.

    Returns:
        :class:`FusionResult` containing the fused array of shape
        ``(M, visual_dim + text_dim)``.

    Raises:
        ValueError: On shape mismatches.
    """
    cfg = cfg or CFG.fusion

    if len(valid_indices) != len(text_embeddings):
        raise ValueError(
            f"valid_indices length ({len(valid_indices)}) must match "
            f"text_embeddings rows ({len(text_embeddings)})."
        )

    # Select only the visual rows that have a matching text embedding
    visual_subset = visual_features[valid_indices]  # (M, visual_dim)

    if cfg.normalize_before_fusion:
        # Re-normalise each modality independently so neither dominates by scale
        visual_subset = normalize(visual_subset, norm="l2")
        text_embeddings = normalize(text_embeddings, norm="l2")

    # Apply per-modality weights before concatenation
    weighted_visual = visual_subset * cfg.visual_weight
    weighted_text = text_embeddings * cfg.text_weight

    fused = np.concatenate([weighted_visual, weighted_text], axis=1)
    logger.info(
        "Fused features: visual_weight=%.2f, text_weight=%.2f, output_shape=%s",
        cfg.visual_weight,
        cfg.text_weight,
        fused.shape,
    )
    return FusionResult(
        fused=fused,
        visual_dim=visual_subset.shape[1],
        text_dim=text_embeddings.shape[1],
    )


# ---------------------------------------------------------------------------
# 2. PCA
# ---------------------------------------------------------------------------


def reduce_dimensions(
    features: np.ndarray,
    cfg: Optional[DimensionReductionConfig] = None,
) -> PCAResult:
    """Reduce high-dimensional fused features to a compact representation.

    If ``cfg.n_components`` is *None*, the number of components is determined
    automatically to retain at least ``cfg.variance_threshold`` cumulative
    explained variance, capped at ``cfg.max_components``.

    Args:
        features: Input array of shape ``(N, D)``.
        cfg: PCA configuration; defaults to ``CFG.dim_reduction``.

    Returns:
        :class:`PCAResult` with the reduced array and fitted PCA object.

    Raises:
        ValueError: If *features* is empty.
    """
    cfg = cfg or CFG.dim_reduction

    if features.size == 0:
        raise ValueError("features array must not be empty.")

    n_samples, n_features = features.shape
    logger.info("Running PCA on array of shape %s...", features.shape)

    if cfg.n_components is not None:
        # Fixed number of components requested
        n_components = min(cfg.n_components, n_samples, n_features)
    else:
        # First fit a full PCA to inspect cumulative variance
        pca_full = PCA(random_state=cfg.random_state)
        pca_full.fit(features)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        # Find the smallest k that meets the variance threshold
        n_components = int(np.searchsorted(cum_var, cfg.variance_threshold) + 1)
        # Enforce the hard cap
        n_components = min(n_components, cfg.max_components, n_samples, n_features)
        logger.info(
            "PCA auto-selected %d components to explain %.1f %% variance.",
            n_components,
            cfg.variance_threshold * 100,
        )

    pca = PCA(n_components=n_components, random_state=cfg.random_state)
    reduced = pca.fit_transform(features)

    cumulative = float(pca.explained_variance_ratio_.sum())
    logger.info(
        "PCA complete: %dD → %dD  (%.2f %% variance retained).",
        n_features,
        n_components,
        cumulative * 100,
    )
    return PCAResult(
        reduced=reduced,
        pca=pca,
        explained_variance_ratio=pca.explained_variance_ratio_,
        n_components=n_components,
        cumulative_variance=cumulative,
    )


# ---------------------------------------------------------------------------
# 3. Clustering
# ---------------------------------------------------------------------------


def cluster_hdbscan(
    features: np.ndarray,
    cfg: ClusteringConfig,
) -> ClusterResult:
    """Cluster using HDBSCAN (density-based, automatically determines k).

    Args:
        features: PCA-reduced feature array ``(N, d)``.
        cfg: Clustering configuration.

    Returns:
        :class:`ClusterResult` with labels and soft membership probabilities.

    Raises:
        ImportError: If the *hdbscan* package is not installed.
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError(
            "hdbscan package is required. Install with: pip install hdbscan"
        )

    clusterer = _hdbscan_module.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric=cfg.hdbscan_metric,
        prediction_data=True,  # enables soft cluster membership
    )
    labels = clusterer.fit_predict(features)
    probabilities = clusterer.probabilities_

    n_clusters = int(labels.max()) + 1  # labels are 0-indexed; -1 = noise
    n_noise = int((labels == -1).sum())
    logger.info("HDBSCAN: %d cluster(s), %d noise point(s).", n_clusters, n_noise)
    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        algorithm="hdbscan",
        probabilities=probabilities,
    )


def cluster_kmeans(
    features: np.ndarray,
    cfg: ClusteringConfig,
) -> ClusterResult:
    """Cluster using K-Means++ initialisation.

    Args:
        features: PCA-reduced feature array ``(N, d)``.
        cfg: Clustering configuration.

    Returns:
        :class:`ClusterResult` with integer labels (no noise points).
    """
    km = KMeans(
        n_clusters=cfg.kmeans_n_clusters,
        init="k-means++",
        max_iter=cfg.kmeans_max_iter,
        n_init=cfg.kmeans_n_init,
        random_state=cfg.random_state,
    )
    labels = km.fit_predict(features)
    logger.info("K-Means++: %d cluster(s) fitted.", cfg.kmeans_n_clusters)
    return ClusterResult(
        labels=labels,
        n_clusters=cfg.kmeans_n_clusters,
        n_noise=0,
        algorithm="kmeans",
        probabilities=None,
    )


def run_clustering(
    features: np.ndarray,
    cfg: Optional[ClusteringConfig] = None,
) -> ClusterResult:
    """Dispatch to the appropriate clustering algorithm.

    Args:
        features: PCA-reduced feature array ``(N, d)``.
        cfg: Clustering configuration; defaults to ``CFG.clustering``.

    Returns:
        :class:`ClusterResult`.

    Raises:
        ValueError: If ``cfg.algorithm`` is not recognised.
    """
    cfg = cfg or CFG.clustering

    if cfg.algorithm == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN unavailable – falling back to K-Means++.")
            return cluster_kmeans(features, cfg)
        return cluster_hdbscan(features, cfg)

    if cfg.algorithm == "kmeans":
        return cluster_kmeans(features, cfg)

    raise ValueError(
        f"Unknown clustering algorithm '{cfg.algorithm}'. Choose 'hdbscan' or 'kmeans'."
    )


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    features: np.ndarray,
    labels: np.ndarray,
) -> ClusterMetrics:
    """Compute intrinsic clustering quality metrics.

    Noise points (label == -1) are excluded from all metric calculations
    because they are not assigned to any cluster.

    Args:
        features: PCA-reduced feature array ``(N, d)``.
        labels: Cluster label per sample; -1 denotes noise.

    Returns:
        :class:`ClusterMetrics` dataclass.

    Raises:
        ValueError: If fewer than 2 non-noise clusters exist (metrics
            are undefined in that case).
    """
    n_noise = int((labels == -1).sum())
    # Remove noise points before computing metrics
    mask = labels != -1
    clean_features = features[mask]
    clean_labels = labels[mask]

    unique_labels = np.unique(clean_labels)
    if len(unique_labels) < 2:
        raise ValueError(
            "At least 2 non-noise clusters are required to compute metrics. "
            f"Found: {len(unique_labels)}."
        )

    # Silhouette: mean intra-cluster cohesion vs inter-cluster separation
    sil = silhouette_score(clean_features, clean_labels, metric="euclidean")
    # Davies-Bouldin: average similarity of each cluster to its most similar one
    db = davies_bouldin_score(clean_features, clean_labels)
    # Calinski-Harabász: ratio of between-cluster to within-cluster dispersion
    ch = calinski_harabasz_score(clean_features, clean_labels)

    n_clusters = len(unique_labels)
    noise_ratio = n_noise / len(labels)

    logger.info(
        "Metrics — Silhouette: %.4f | Davies-Bouldin: %.4f | "
        "Calinski-Harabász: %.2f | Clusters: %d | Noise: %d (%.1f %%)",
        sil,
        db,
        ch,
        n_clusters,
        n_noise,
        noise_ratio * 100,
    )
    return ClusterMetrics(
        silhouette=float(sil),
        davies_bouldin=float(db),
        calinski_harabasz=float(ch),
        n_clusters=n_clusters,
        n_noise=n_noise,
        noise_ratio=noise_ratio,
    )
