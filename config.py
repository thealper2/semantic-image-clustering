"""
Configuration module for Semantic Image Clustering.

Centralizes all hyperparameters, paths, and model settings
to avoid magic numbers scattered across the codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
BackboneType = Literal["resnet50", "efficientnet_b4"]
ClusterAlgorithm = Literal["hdbscan", "kmeans"]
EmbedModel = Literal["nomic-embed-text", "all-minilm"]


@dataclass
class PathConfig:
    """File-system paths used throughout the project."""

    data_dir: Path = Path("data/images")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")
    plots_dir: Path = Path("outputs/plots")
    models_dir: Path = Path("outputs/models")

    def __post_init__(self) -> None:
        # Create directories if they don't exist yet
        for p in (
            self.data_dir,
            self.output_dir,
            self.cache_dir,
            self.plots_dir,
            self.models_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    """Visual feature extraction settings."""

    backbone: BackboneType = "resnet50"
    # Input resolution fed to the backbone
    image_size: tuple[int, int] = (224, 224)
    # ImageNet normalisation constants
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Number of images processed in a single forward pass
    batch_size: int = 32
    # Pin memory speeds up CPU→GPU transfers
    pin_memory: bool = True
    num_workers: int = 4


@dataclass
class CaptionConfig:
    """BLIP caption generation settings."""

    model_name: str = "Salesforce/blip-image-captioning-base"
    max_new_tokens: int = 50
    # Number of captions generated per image (beam-search beams)
    num_beams: int = 4
    batch_size: int = 16


@dataclass
class EmbeddingConfig:
    """Ollama text-embedding settings."""

    model: EmbedModel = "nomic-embed-text"
    # Ollama REST API endpoint
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 30
    # Retry on transient network errors
    max_retries: int = 3


@dataclass
class FusionConfig:
    """Visual + text fusion settings."""

    # Weight applied to visual embeddings before concatenation
    visual_weight: float = 0.7
    # Weight applied to text embeddings before concatenation
    text_weight: float = 0.3
    # L2-normalise each modality independently before fusion
    normalize_before_fusion: bool = True


@dataclass
class DimensionReductionConfig:
    """PCA dimensionality reduction settings."""

    # Target number of components (None → use variance_threshold instead)
    n_components: Optional[int] = None
    # Minimum cumulative explained variance to retain
    variance_threshold: float = 0.95
    # Hard cap on the number of PCA dimensions
    max_components: int = 100
    random_state: int = 42


@dataclass
class ClusteringConfig:
    """Clustering algorithm settings."""

    algorithm: ClusterAlgorithm = "hdbscan"

    # --- HDBSCAN knobs ---
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    # "euclidean" works well in PCA-reduced space
    hdbscan_metric: str = "euclidean"

    # --- K-Means++ knobs ---
    kmeans_n_clusters: int = 10
    kmeans_max_iter: int = 300
    kmeans_n_init: int = 10
    random_state: int = 42


@dataclass
class VisualizationConfig:
    """Plot and UMAP/t-SNE visualisation settings."""

    # 2-D projection method used for scatter plots
    projection: Literal["umap", "tsne"] = "umap"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    figure_dpi: int = 150
    colormap: str = "tab20"
    random_state: int = 42


@dataclass
class AppConfig:
    """Top-level config aggregating all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    dim_reduction: DimensionReductionConfig = field(
        default_factory=DimensionReductionConfig
    )
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    # Global random seed for reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Singleton – import `CFG` wherever you need configuration
# ---------------------------------------------------------------------------
CFG = AppConfig()
