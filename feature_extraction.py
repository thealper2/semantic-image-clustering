"""
Visual feature extraction using ResNet50 or EfficientNet-B4.

Each image is passed through the backbone (minus its classification head)
to produce a 2048-D (ResNet50) or 1792-D (EfficientNet-B4) feature vector.
Features are L2-normalised so cosine distance == Euclidean distance on the
unit sphere, which benefits downstream clustering.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from config import CFG, BackboneType, FeatureConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ImageFolderDataset(Dataset):
    """Lightweight dataset that loads images from a flat directory.

    Args:
        image_paths: Ordered list of image file paths.
        transform: torchvision transform applied to each image.
    """

    # Supported image extensions
    VALID_EXTENSIONS: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    )

    def __init__(
        self,
        image_paths: list[Path],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        # Filter out unsupported file types up-front to avoid mid-loop errors
        self.image_paths: list[Path] = [
            p for p in image_paths if p.suffix.lower() in self.VALID_EXTENSIONS
        ]
        if len(self.image_paths) != len(image_paths):
            skipped = len(image_paths) - len(self.image_paths)
            logger.warning("Skipped %d file(s) with unsupported extensions.", skipped)

        self.transform = transform or _default_transform(CFG.features)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        try:
            # Convert to RGB so greyscale / RGBA images are handled uniformly
            image = Image.open(path).convert("RGB")
        except (OSError, ValueError) as exc:
            logger.error("Cannot open image %s: %s", path, exc)
            # Return a black image as a safe fallback so the batch doesn't fail
            image = Image.new("RGB", CFG.features.image_size)

        if self.transform:
            image = self.transform(image)

        return image, str(path)


# ---------------------------------------------------------------------------
# Transform factory
# ---------------------------------------------------------------------------


def _default_transform(cfg: FeatureConfig) -> transforms.Compose:
    """Build the standard ImageNet pre-processing pipeline.

    Args:
        cfg: Feature extraction configuration.

    Returns:
        A composed torchvision transform.
    """
    return transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
        ]
    )


# ---------------------------------------------------------------------------
# Backbone builder
# ---------------------------------------------------------------------------


def build_backbone(backbone: BackboneType, device: torch.device) -> nn.Module:
    """Instantiate the chosen CNN backbone with its classification head removed.

    Args:
        backbone: One of "resnet50" or "efficientnet_b4".
        device: Target device (cpu / cuda / mps).

    Returns:
        Feature extractor in eval mode, moved to *device*.

    Raises:
        ValueError: If an unsupported backbone name is supplied.
    """
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final fully-connected layer → output is (N, 2048)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])

    elif backbone == "efficientnet_b4":
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        # Keep features + adaptive pooling, drop the classifier
        feature_extractor = nn.Sequential(
            model.features,
            model.avgpool,
        )
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. "
            "Choose 'resnet50' or 'efficientnet_b4'."
        )

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    logger.info("Backbone '%s' loaded on %s.", backbone, device)
    return feature_extractor


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_visual_features(
    image_paths: list[Path],
    cfg: Optional[FeatureConfig] = None,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract and L2-normalise visual features for a list of images.

    Args:
        image_paths: Paths to images on disk.
        cfg: Feature extraction configuration; defaults to ``CFG.features``.
        device: Torch device; auto-detected if *None*.

    Returns:
        A tuple ``(features, paths)`` where *features* has shape
        ``(N, feature_dim)`` and *paths* lists the resolved string paths
        in the same order.

    Raises:
        ValueError: If *image_paths* is empty.
    """
    if not image_paths:
        raise ValueError("image_paths must not be empty.")

    cfg = cfg or CFG.features

    # Auto-detect best available device
    if device is None:
        device = _select_device()

    model = build_backbone(cfg.backbone, device)

    dataset = ImageFolderDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # preserve order for alignment with captions
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
    )

    all_features: list[np.ndarray] = []
    all_paths: list[str] = []

    with torch.no_grad():
        for batch_images, batch_paths in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            feats = model(batch_images)
            # Flatten spatial dims: (N, C, 1, 1) → (N, C)
            feats = feats.view(feats.size(0), -1)
            # L2 normalisation – brings all vectors onto the unit hypersphere
            feats = nn.functional.normalize(feats, p=2, dim=1)
            all_features.append(feats.cpu().numpy())
            all_paths.extend(batch_paths)
            logger.debug("Processed batch; total so far: %d", len(all_paths))

    features = np.vstack(all_features)
    logger.info(
        "Extracted visual features: shape=%s, backbone=%s",
        features.shape,
        cfg.backbone,
    )
    return features, all_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    """Pick the fastest available device: CUDA > MPS > CPU.

    Returns:
        A ``torch.device`` instance.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


def discover_images(directory: Path) -> list[Path]:
    """Recursively discover all supported image files under *directory*.

    Args:
        directory: Root directory to search.

    Returns:
        Sorted list of image paths.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    valid_exts = ImageFolderDataset.VALID_EXTENSIONS
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in valid_exts)
    logger.info("Discovered %d image(s) in '%s'.", len(paths), directory)
    return paths
