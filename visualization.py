"""
Visualisation utilities for the Semantic Image Clustering pipeline.

Produces four plot types:
  1. 2-D cluster scatter (UMAP or t-SNE projection).
  2. PCA explained-variance elbow curve.
  3. Cluster size bar chart.
  4. Per-cluster image grid (thumbnail montage).
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

from clustering import ClusterMetrics, ClusterResult, PCAResult
from config import CFG, VisualizationConfig

logger = logging.getLogger(__name__)

# Lazy imports for dimensionality-reduction projectors
try:
    import umap as _umap_module  # umap-learn

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning(
        "umap-learn not installed. Falling back to t-SNE for 2-D projection."
    )

from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# 2-D projection helper
# ---------------------------------------------------------------------------


def project_to_2d(
    features: np.ndarray,
    cfg: Optional[VisualizationConfig] = None,
) -> np.ndarray:
    """Project high-dimensional features to 2-D for scatter plotting.

    Args:
        features: Array of shape ``(N, d)``.
        cfg: Visualisation config; defaults to ``CFG.visualization``.

    Returns:
        Array of shape ``(N, 2)``.
    """
    cfg = cfg or CFG.visualization

    use_umap = cfg.projection == "umap" and UMAP_AVAILABLE

    if use_umap:
        reducer = _umap_module.UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            n_components=2,
            random_state=cfg.random_state,
        )
        logger.info("Running UMAP projection...")
    else:
        if cfg.projection == "umap" and not UMAP_AVAILABLE:
            logger.warning("UMAP requested but unavailable – using t-SNE.")
        reducer = TSNE(
            n_components=2,
            perplexity=cfg.tsne_perplexity,
            n_iter=cfg.tsne_n_iter,
            random_state=cfg.random_state,
        )
        logger.info("Running t-SNE projection...")

    coords = reducer.fit_transform(features)
    logger.info("2-D projection complete: shape=%s.", coords.shape)
    return coords


# ---------------------------------------------------------------------------
# Plot 1: Cluster scatter
# ---------------------------------------------------------------------------


def plot_cluster_scatter(
    features: np.ndarray,
    cluster_result: ClusterResult,
    metrics: ClusterMetrics,
    save_path: Optional[Path] = None,
    cfg: Optional[VisualizationConfig] = None,
) -> plt.Figure:
    """2-D scatter plot coloured by cluster label.

    Args:
        features: PCA-reduced features ``(N, d)``.
        cluster_result: Output from :func:`run_clustering`.
        metrics: Computed quality metrics.
        save_path: If given, save the figure here.
        cfg: Visualisation config; defaults to ``CFG.visualization``.

    Returns:
        Matplotlib figure.
    """
    cfg = cfg or CFG.visualization
    coords = project_to_2d(features, cfg)
    labels = cluster_result.labels

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    unique_labels = np.unique(labels)
    # Assign a colour to every cluster; noise points (-1) get grey
    cmap = plt.get_cmap(cfg.colormap, max(len(unique_labels), 1))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels) if lbl != -1}
    color_map[-1] = (0.5, 0.5, 0.5, 0.4)  # translucent grey for noise

    for lbl in unique_labels:
        mask = labels == lbl
        colour = color_map[lbl]
        label_str = (
            f"Noise ({mask.sum()})" if lbl == -1 else f"Cluster {lbl} ({mask.sum()})"
        )
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colour],
            s=18,
            alpha=0.8 if lbl != -1 else 0.3,
            label=label_str,
            linewidths=0,
        )

    # Annotate with key metrics
    info = (
        f"Algorithm: {cluster_result.algorithm.upper()}  |  "
        f"Clusters: {metrics.n_clusters}  |  "
        f"Noise: {metrics.n_noise} ({metrics.noise_ratio:.1%})\n"
        f"Silhouette: {metrics.silhouette:.3f}  |  "
        f"DB Index: {metrics.davies_bouldin:.3f}  |  "
        f"CH Score: {metrics.calinski_harabasz:.1f}"
    )
    ax.set_title(
        f"Semantic Image Clusters\n{info}",
        color="white",
        fontsize=11,
        pad=14,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    # Legend: limit to 20 entries to avoid a crowded legend
    handles, legend_labels = ax.get_legend_handles_labels()
    if len(handles) > 20:
        handles, legend_labels = handles[:20], legend_labels[:20]
        legend_labels[-1] += " …"
    ax.legend(
        handles,
        legend_labels,
        fontsize=7,
        loc="upper right",
        framealpha=0.2,
        labelcolor="white",
        markerscale=1.5,
    )

    ax.set_xlabel("Dim 1", color="#aaaacc")
    ax.set_ylabel("Dim 2", color="#aaaacc")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=cfg.figure_dpi, bbox_inches="tight")
        logger.info("Cluster scatter saved → %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Plot 2: PCA explained variance elbow
# ---------------------------------------------------------------------------


def plot_pca_variance(
    pca_result: PCAResult,
    save_path: Optional[Path] = None,
    cfg: Optional[VisualizationConfig] = None,
) -> plt.Figure:
    """Elbow curve showing cumulative explained variance per PCA component.

    Args:
        pca_result: Output from :func:`reduce_dimensions`.
        save_path: If given, save the figure here.
        cfg: Visualisation config; defaults to ``CFG.visualization``.

    Returns:
        Matplotlib figure.
    """
    cfg = cfg or CFG.visualization
    evr = pca_result.explained_variance_ratio
    cum_var = np.cumsum(evr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1117")

    components = np.arange(1, len(evr) + 1)

    # Left: individual component variance
    ax1.bar(components, evr * 100, color="#4c72b0", alpha=0.85, width=0.8)
    ax1.set_title("Per-Component Variance", color="white")
    ax1.set_xlabel("Component", color="#aaaacc")
    ax1.set_ylabel("Explained Variance (%)", color="#aaaacc")
    ax1.tick_params(colors="white")

    # Right: cumulative variance with threshold line
    ax2.plot(
        components,
        cum_var * 100,
        color="#55cc88",
        linewidth=2.5,
        marker="o",
        markersize=3,
    )
    ax2.axhline(
        y=95, color="#ff6666", linestyle="--", linewidth=1, label="95 % threshold"
    )
    ax2.axvline(
        x=pca_result.n_components,
        color="#ffaa44",
        linestyle=":",
        linewidth=1.5,
        label=f"Selected: {pca_result.n_components}D ({pca_result.cumulative_variance:.1%})",
    )
    ax2.set_title("Cumulative Explained Variance", color="white")
    ax2.set_xlabel("Number of Components", color="#aaaacc")
    ax2.set_ylabel("Cumulative Variance (%)", color="#aaaacc")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=8, framealpha=0.2, labelcolor="white")

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    fig.suptitle("PCA Dimensionality Reduction Analysis", color="white", fontsize=13)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=cfg.figure_dpi, bbox_inches="tight")
        logger.info("PCA variance plot saved → %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Plot 3: Cluster size bar chart
# ---------------------------------------------------------------------------


def plot_cluster_sizes(
    cluster_result: ClusterResult,
    save_path: Optional[Path] = None,
    cfg: Optional[VisualizationConfig] = None,
) -> plt.Figure:
    """Horizontal bar chart showing the number of images per cluster.

    Args:
        cluster_result: Clustering result.
        save_path: If given, save the figure here.
        cfg: Visualisation config; defaults to ``CFG.visualization``.

    Returns:
        Matplotlib figure.
    """
    cfg = cfg or CFG.visualization
    labels = cluster_result.labels
    unique, counts = np.unique(labels, return_counts=True)

    # Put noise last for readability
    order = np.argsort(unique)
    unique, counts = unique[order], counts[order]
    bar_labels = [f"Noise" if lbl == -1 else f"Cluster {lbl}" for lbl in unique]
    colours = ["#888888" if lbl == -1 else f"C{i}" for i, lbl in enumerate(unique)]

    fig, ax = plt.subplots(figsize=(8, max(4, len(unique) * 0.45)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.barh(bar_labels, counts, color=colours, height=0.65, alpha=0.85)
    ax.bar_label(bars, fmt="%d", padding=4, color="white", fontsize=9)
    ax.set_xlabel("Number of Images", color="#aaaacc")
    ax.set_title("Images per Cluster", color="white", fontsize=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=cfg.figure_dpi, bbox_inches="tight")
        logger.info("Cluster size plot saved → %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Plot 4: Per-cluster thumbnail grid
# ---------------------------------------------------------------------------


def plot_cluster_thumbnails(
    image_paths: list[str],
    labels: np.ndarray,
    cluster_id: int,
    max_images: int = 16,
    thumbnail_size: tuple[int, int] = (96, 96),
    save_path: Optional[Path] = None,
    cfg: Optional[VisualizationConfig] = None,
) -> plt.Figure:
    """Display a grid of thumbnail images belonging to a specific cluster.

    Args:
        image_paths: Ordered list of image file paths (strings).
        labels: Cluster label per image.
        cluster_id: The cluster whose images to display.
        max_images: Maximum thumbnails to show (avoids huge figures).
        thumbnail_size: ``(width, height)`` in pixels for each thumbnail.
        save_path: If given, save the figure here.
        cfg: Visualisation config; defaults to ``CFG.visualization``.

    Returns:
        Matplotlib figure, or an empty figure if the cluster has no members.
    """
    cfg = cfg or CFG.visualization
    indices = np.where(labels == cluster_id)[0][:max_images]

    if len(indices) == 0:
        logger.warning("Cluster %d has no members.", cluster_id)
        return plt.figure()

    ncols = min(4, len(indices))
    nrows = (len(indices) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    fig.patch.set_facecolor("#0f1117")

    # Flatten axes for uniform iteration even when nrows == 1
    axes_flat = np.array(axes).flatten()

    for ax, idx in zip(axes_flat, indices):
        try:
            img = Image.open(image_paths[idx]).convert("RGB").resize(thumbnail_size)
            ax.imshow(np.array(img))
        except (OSError, ValueError):
            ax.set_facecolor("#222233")
        ax.axis("off")
        ax.set_title(
            Path(image_paths[idx]).name[:18],
            fontsize=6,
            color="#bbbbcc",
        )

    # Hide unused subplot slots
    for ax in axes_flat[len(indices) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Cluster {cluster_id}  –  {len(np.where(labels == cluster_id)[0])} image(s)",
        color="white",
        fontsize=11,
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=cfg.figure_dpi, bbox_inches="tight")
        logger.info("Thumbnail grid for cluster %d saved → %s", cluster_id, save_path)

    return fig


# ---------------------------------------------------------------------------
# Convenience: save all standard visualisations at once
# ---------------------------------------------------------------------------


def save_all_plots(
    features_reduced: np.ndarray,
    pca_result: PCAResult,
    cluster_result: ClusterResult,
    metrics: ClusterMetrics,
    image_paths: list[str],
    output_dir: Path,
    cfg: Optional[VisualizationConfig] = None,
) -> dict[str, Path]:
    """Generate and save all four standard plots.

    Args:
        features_reduced: PCA-reduced features ``(N, d)``.
        pca_result: PCA result container.
        cluster_result: Clustering result container.
        metrics: Clustering quality metrics.
        image_paths: Ordered image path strings.
        output_dir: Directory where plots are written.
        cfg: Visualisation config; defaults to ``CFG.visualization``.

    Returns:
        Dict mapping plot name → saved file path.
    """
    cfg = cfg or CFG.visualization
    saved: dict[str, Path] = {}

    scatter_path = output_dir / "cluster_scatter.png"
    plot_cluster_scatter(features_reduced, cluster_result, metrics, scatter_path, cfg)
    saved["scatter"] = scatter_path

    pca_path = output_dir / "pca_variance.png"
    plot_pca_variance(pca_result, pca_path, cfg)
    saved["pca_variance"] = pca_path

    sizes_path = output_dir / "cluster_sizes.png"
    plot_cluster_sizes(cluster_result, sizes_path, cfg)
    saved["cluster_sizes"] = sizes_path

    # Generate a thumbnail grid for each non-noise cluster (up to 5 clusters)
    unique_labels = [lbl for lbl in np.unique(cluster_result.labels) if lbl != -1]
    for lbl in unique_labels[:5]:
        thumb_path = output_dir / f"cluster_{lbl}_thumbnails.png"
        plot_cluster_thumbnails(
            image_paths, cluster_result.labels, lbl, save_path=thumb_path
        )
        saved[f"thumbnails_cluster_{lbl}"] = thumb_path

    logger.info("All plots saved to '%s'.", output_dir)
    return saved
