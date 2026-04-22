"""
Mesop web UI for Semantic Image Clustering.

Pages:
  /           → Home / pipeline launcher
  /results    → Clustering results + plots
  /inference  → Single-image cluster prediction
"""

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mesop as me
import mesop.labs as mel

# Ensure the project root is on sys.path so sibling modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

from config import CFG, BackboneType, ClusterAlgorithm
from pipeline import infer_cluster, load_pipeline_state, run_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App state dataclass
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    """Global Mesop UI state shared across components."""

    # Pipeline configuration inputs (reflected from UI widgets)
    image_dir: str = "data/images"
    backbone: BackboneType = "resnet50"
    algorithm: ClusterAlgorithm = "hdbscan"
    kmeans_k: int = 10
    embed_model: str = "nomic-embed-text"
    visual_weight: float = 0.7

    # Pipeline execution state
    is_running: bool = False
    run_complete: bool = False
    run_error: str = ""
    run_log: list[str] = field(default_factory=list)

    # Results (populated after a successful run)
    n_clusters: int = 0
    n_noise: int = 0
    silhouette: float = 0.0
    davies_bouldin: float = 0.0
    calinski_harabasz: float = 0.0
    cumulative_variance: float = 0.0
    n_components: int = 0

    # Inference
    infer_image_path: str = ""
    infer_result: str = ""
    infer_error: str = ""

    # Active page tab
    active_tab: str = "run"  # "run" | "results" | "inference"


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

BG_DARK = "#0f1117"
BG_CARD = "#1a1d27"
ACCENT = "#6c63ff"
ACCENT_LIGHT = "#8f89ff"
SUCCESS = "#4caf85"
ERROR = "#e05858"
TEXT_PRIMARY = "#e0e0f0"
TEXT_SECONDARY = "#9090b0"
BORDER_COLOR = "#2a2d3e"


def _card_style() -> me.Style:
    return me.Style(
        background=BG_CARD,
        border_radius=12,
        padding=me.Padding.all(20),
        margin=me.Margin(bottom=16),
        border=me.Border.all(me.BorderSide(color=BORDER_COLOR, width=1)),
    )


def _label_style() -> me.Style:
    return me.Style(color=TEXT_SECONDARY, font_size=12, margin=me.Margin(bottom=4))


def _value_style(color: str = TEXT_PRIMARY) -> me.Style:
    return me.Style(color=color, font_size=22, font_weight="700")


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def on_image_dir_change(e: me.InputEvent) -> None:
    """Update image directory path in state."""
    me.state(AppState).image_dir = e.value


def on_backbone_change(e: me.SelectSelectionChangeEvent) -> None:
    """Update backbone model selection."""
    me.state(AppState).backbone = e.value  # type: ignore[assignment]


def on_algorithm_change(e: me.SelectSelectionChangeEvent) -> None:
    """Update clustering algorithm selection."""
    me.state(AppState).algorithm = e.value  # type: ignore[assignment]


def on_kmeans_k_change(e: me.SliderValueChangeEvent) -> None:
    """Update K-Means cluster count."""
    me.state(AppState).kmeans_k = int(e.value)


def on_embed_model_change(e: me.SelectSelectionChangeEvent) -> None:
    """Update Ollama embedding model selection."""
    me.state(AppState).embed_model = e.value


def on_visual_weight_change(e: me.SliderValueChangeEvent) -> None:
    """Update visual-to-text fusion weight."""
    me.state(AppState).visual_weight = round(e.value / 100, 2)


def on_infer_path_change(e: me.InputEvent) -> None:
    """Update inference query image path."""
    me.state(AppState).infer_image_path = e.value


def _run_pipeline_thread(state_snapshot: AppState) -> None:
    """Execute the pipeline in a background thread.

    Mutations to ``me.state`` cannot happen here; results are persisted to
    disk and the UI reads them on the next render cycle.
    """
    try:
        # Override config from UI state
        CFG.features.backbone = state_snapshot.backbone  # type: ignore[assignment]
        CFG.clustering.algorithm = state_snapshot.algorithm  # type: ignore[assignment]
        CFG.clustering.kmeans_n_clusters = state_snapshot.kmeans_k
        CFG.embeddings.model = state_snapshot.embed_model  # type: ignore[assignment]
        CFG.fusion.visual_weight = state_snapshot.visual_weight
        CFG.fusion.text_weight = round(1.0 - state_snapshot.visual_weight, 2)

        cluster_result, metrics, _ = run_pipeline(
            image_dir=Path(state_snapshot.image_dir)
        )

        # Write a lightweight summary JSON for the UI to pick up
        summary = {
            "n_clusters": metrics.n_clusters,
            "n_noise": metrics.n_noise,
            "silhouette": metrics.silhouette,
            "davies_bouldin": metrics.davies_bouldin,
            "calinski_harabasz": metrics.calinski_harabasz,
            "pca_components": 0,  # will be updated below
            "pca_variance": 0.0,
        }
        # Try to read PCA details from state
        try:
            state = load_pipeline_state(CFG.paths.output_dir / "pipeline_state.pkl")
            summary["pca_components"] = state["pca_result"].n_components
            summary["pca_variance"] = state["pca_result"].cumulative_variance
        except Exception:  # noqa: BLE001
            pass

        summary_path = CFG.paths.output_dir / "ui_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

    except Exception as exc:  # noqa: BLE001
        error_path = CFG.paths.output_dir / "ui_error.txt"
        error_path.write_text(str(exc))
        logger.exception("Pipeline failed: %s", exc)


def on_run_pipeline(e: me.ClickEvent) -> None:
    """Start the pipeline in a background thread."""
    state = me.state(AppState)
    # Clear previous run artefacts
    error_path = CFG.paths.output_dir / "ui_error.txt"
    summary_path = CFG.paths.output_dir / "ui_summary.json"
    for p in (error_path, summary_path):
        if p.exists():
            p.unlink()

    state.is_running = True
    state.run_complete = False
    state.run_error = ""

    # Pass a copy of state values to the thread (Mesop state is not thread-safe)
    snapshot = AppState(
        image_dir=state.image_dir,
        backbone=state.backbone,
        algorithm=state.algorithm,
        kmeans_k=state.kmeans_k,
        embed_model=state.embed_model,
        visual_weight=state.visual_weight,
    )
    t = threading.Thread(target=_run_pipeline_thread, args=(snapshot,), daemon=True)
    t.start()


def on_check_status(e: me.ClickEvent) -> None:
    """Poll disk artefacts to update run status in UI state."""
    state = me.state(AppState)
    summary_path = CFG.paths.output_dir / "ui_summary.json"
    error_path = CFG.paths.output_dir / "ui_error.txt"

    if error_path.exists():
        state.run_error = error_path.read_text()
        state.is_running = False
        state.run_complete = False
        return

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        state.n_clusters = summary.get("n_clusters", 0)
        state.n_noise = summary.get("n_noise", 0)
        state.silhouette = summary.get("silhouette", 0.0)
        state.davies_bouldin = summary.get("davies_bouldin", 0.0)
        state.calinski_harabasz = summary.get("calinski_harabasz", 0.0)
        state.n_components = summary.get("pca_components", 0)
        state.cumulative_variance = summary.get("pca_variance", 0.0)
        state.is_running = False
        state.run_complete = True


def on_run_inference(e: me.ClickEvent) -> None:
    """Run single-image inference synchronously."""
    state = me.state(AppState)
    state.infer_error = ""
    state.infer_result = ""
    try:
        result = infer_cluster(Path(state.infer_image_path))
        lines = [
            f"Predicted cluster : {result['cluster_id']}",
            f"Distance to centroid : {result['distance']:.4f}",
            f"Generated caption : {result['caption']}",
            "",
            "Distances to all clusters:",
        ]
        for lbl, dist in sorted(result["all_distances"].items()):
            lines.append(f"  Cluster {lbl}: {dist:.4f}")
        state.infer_result = "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        state.infer_error = str(exc)


def on_tab_select(e: me.ClickEvent) -> None:
    """Switch the active navigation tab."""
    # The key attribute on the button encodes the tab name
    me.state(AppState).active_tab = e.key


# ---------------------------------------------------------------------------
# Reusable UI components (functions, not classes)
# ---------------------------------------------------------------------------


def metric_card(
    label: str, value: str, sub: str = "", color: str = TEXT_PRIMARY
) -> None:
    """Render a single metric card.

    Args:
        label: Short label shown above the value.
        value: Primary numeric or text value.
        sub: Optional smaller sub-label below the value.
        color: CSS colour for the value text.
    """
    with me.box(style=_card_style()):
        me.text(label, style=_label_style())
        me.text(value, style=_value_style(color))
        if sub:
            me.text(sub, style=me.Style(color=TEXT_SECONDARY, font_size=11))


def config_section() -> None:
    """Render the pipeline configuration form."""
    state = me.state(AppState)

    with me.box(style=_card_style()):
        me.text(
            "Pipeline Configuration",
            style=me.Style(
                color=TEXT_PRIMARY,
                font_size=16,
                font_weight="600",
                margin=me.Margin(bottom=16),
            ),
        )

        # Image directory
        me.text("Image Directory", style=_label_style())
        me.input(
            value=state.image_dir,
            on_input=on_image_dir_change,
            style=me.Style(
                width="100%",
                margin=me.Margin(bottom=14),
                color=TEXT_PRIMARY,
            ),
        )

        # Backbone + algorithm side-by-side
        with me.box(
            style=me.Style(
                display="flex",
                flex_direction="row",
                gap=16,
                margin=me.Margin(bottom=14),
            )
        ):
            with me.box(style=me.Style(flex=1)):
                me.text("Backbone", style=_label_style())
                me.select(
                    options=[
                        me.SelectOption(label="ResNet-50", value="resnet50"),
                        me.SelectOption(
                            label="EfficientNet-B4", value="efficientnet_b4"
                        ),
                    ],
                    value=state.backbone,
                    on_selection_change=on_backbone_change,
                    style=me.Style(width="100%", color=TEXT_PRIMARY),
                )

            with me.box(style=me.Style(flex=1)):
                me.text("Clustering Algorithm", style=_label_style())
                me.select(
                    options=[
                        me.SelectOption(label="HDBSCAN (auto-k)", value="hdbscan"),
                        me.SelectOption(label="K-Means++", value="kmeans"),
                    ],
                    value=state.algorithm,
                    on_selection_change=on_algorithm_change,
                    style=me.Style(width="100%", color=TEXT_PRIMARY),
                )

        # K-Means k slider (only shown when K-Means is selected)
        if state.algorithm == "kmeans":
            me.text(
                f"K-Means clusters: {state.kmeans_k}",
                style=_label_style(),
            )
            me.slider(
                min=2,
                max=50,
                value=state.kmeans_k,
                on_value_change=on_kmeans_k_change,
                style=me.Style(width="100%", margin=me.Margin(bottom=14)),
            )

        # Embedding model
        me.text("Text Embedding Model (Ollama)", style=_label_style())
        me.select(
            options=[
                me.SelectOption(label="nomic-embed-text", value="nomic-embed-text"),
                me.SelectOption(label="all-minilm", value="all-minilm"),
            ],
            value=state.embed_model,
            on_selection_change=on_embed_model_change,
            style=me.Style(
                width="100%", margin=me.Margin(bottom=14), color=TEXT_PRIMARY
            ),
        )

        # Visual / text weight slider
        visual_pct = int(state.visual_weight * 100)
        me.text(
            f"Visual weight: {visual_pct} %   |   Text weight: {100 - visual_pct} %",
            style=_label_style(),
        )
        me.slider(
            min=10,
            max=90,
            value=visual_pct,
            on_value_change=on_visual_weight_change,
            style=me.Style(width="100%", margin=me.Margin(bottom=16)),
        )

        # Launch button
        me.button(
            "▶  Run Pipeline",
            on_click=on_run_pipeline,
            disabled=state.is_running,
            style=me.Style(
                background=ACCENT,
                color="white",
                padding=me.Padding.symmetric(vertical=12, horizontal=24),
                border_radius=8,
                font_size=14,
                font_weight="600",
                cursor="pointer",
                width="100%",
            ),
        )


def status_section() -> None:
    """Render pipeline run status / progress feedback."""
    state = me.state(AppState)

    if state.is_running:
        with me.box(style=_card_style()):
            me.text(
                "⏳  Pipeline is running…",
                style=me.Style(color="#ffcc44", font_size=14),
            )
            me.text(
                "Large datasets may take several minutes. "
                "Click 'Refresh Status' to check for completion.",
                style=me.Style(
                    color=TEXT_SECONDARY, font_size=12, margin=me.Margin(top=8)
                ),
            )
            me.button(
                "↻  Refresh Status",
                on_click=on_check_status,
                style=me.Style(
                    background="#2a2d3e",
                    color=TEXT_PRIMARY,
                    padding=me.Padding.symmetric(vertical=8, horizontal=16),
                    border_radius=6,
                    margin=me.Margin(top=12),
                    cursor="pointer",
                ),
            )

    if state.run_error:
        with me.box(style=_card_style()):
            me.text("❌  Pipeline Error", style=me.Style(color=ERROR, font_size=14))
            me.text(
                state.run_error,
                style=me.Style(
                    color=TEXT_SECONDARY, font_size=12, margin=me.Margin(top=8)
                ),
            )

    if state.run_complete:
        with me.box(style=_card_style()):
            me.text(
                "✅  Pipeline complete – see Results tab",
                style=me.Style(color=SUCCESS, font_size=14),
            )


def results_section() -> None:
    """Render clustering results metrics grid."""
    state = me.state(AppState)

    if not state.run_complete:
        with me.box(style=_card_style()):
            me.text(
                "No results yet. Run the pipeline first.",
                style=me.Style(color=TEXT_SECONDARY),
            )
        return

    me.text(
        "Clustering Results",
        style=me.Style(
            color=TEXT_PRIMARY,
            font_size=18,
            font_weight="700",
            margin=me.Margin(bottom=16),
        ),
    )

    # Top-row metrics
    with me.box(
        style=me.Style(
            display="flex",
            flex_direction="row",
            gap=12,
            flex_wrap="wrap",
        )
    ):
        with me.box(style=me.Style(flex=1, min_width=140)):
            metric_card("Clusters", str(state.n_clusters), color=ACCENT_LIGHT)
        with me.box(style=me.Style(flex=1, min_width=140)):
            metric_card("Noise Points", str(state.n_noise), color="#ffaa44")
        with me.box(style=me.Style(flex=1, min_width=140)):
            metric_card(
                "Silhouette",
                f"{state.silhouette:.4f}",
                sub="↑ higher is better",
                color=SUCCESS,
            )
        with me.box(style=me.Style(flex=1, min_width=140)):
            metric_card(
                "Davies-Bouldin",
                f"{state.davies_bouldin:.4f}",
                sub="↓ lower is better",
                color="#ff9966",
            )
        with me.box(style=me.Style(flex=1, min_width=140)):
            metric_card(
                "Calinski-Harabász",
                f"{state.calinski_harabasz:.1f}",
                sub="↑ higher is better",
                color=ACCENT_LIGHT,
            )
        with me.box(style=me.Style(flex=1, min_width=140)):
            metric_card(
                "PCA Dimensions",
                f"{state.n_components}D",
                sub=f"{state.cumulative_variance:.1%} variance",
                color=TEXT_PRIMARY,
            )

    # Plot paths info
    plots_dir = CFG.paths.plots_dir
    with me.box(style=_card_style()):
        me.text(
            "Saved Visualisations",
            style=me.Style(color=TEXT_PRIMARY, font_size=14, font_weight="600"),
        )
        me.text(
            str(plots_dir),
            style=me.Style(color=ACCENT_LIGHT, font_size=12, margin=me.Margin(top=6)),
        )
        plot_files = list(plots_dir.glob("*.png")) if plots_dir.exists() else []
        for pf in plot_files:
            me.text(
                f"• {pf.name}",
                style=me.Style(color=TEXT_SECONDARY, font_size=12),
            )


def inference_section() -> None:
    """Render single-image cluster inference form."""
    state = me.state(AppState)

    with me.box(style=_card_style()):
        me.text(
            "Single-Image Inference",
            style=me.Style(
                color=TEXT_PRIMARY,
                font_size=16,
                font_weight="600",
                margin=me.Margin(bottom=12),
            ),
        )
        me.text(
            "Enter the path to a new image to predict its semantic cluster.",
            style=me.Style(
                color=TEXT_SECONDARY, font_size=12, margin=me.Margin(bottom=12)
            ),
        )
        me.input(
            value=state.infer_image_path,
            placeholder="e.g. /home/user/photo.jpg",
            on_input=on_infer_path_change,
            style=me.Style(
                width="100%", margin=me.Margin(bottom=12), color=TEXT_PRIMARY
            ),
        )
        me.button(
            "🔍  Predict Cluster",
            on_click=on_run_inference,
            disabled=not state.infer_image_path,
            style=me.Style(
                background=ACCENT,
                color="white",
                padding=me.Padding.symmetric(vertical=10, horizontal=20),
                border_radius=8,
                font_size=13,
                font_weight="600",
                cursor="pointer",
            ),
        )

    if state.infer_error:
        with me.box(style=_card_style()):
            me.text("❌  Inference Error", style=me.Style(color=ERROR, font_size=13))
            me.text(
                state.infer_error, style=me.Style(color=TEXT_SECONDARY, font_size=12)
            )

    if state.infer_result:
        with me.box(style=_card_style()):
            me.text("✅  Inference Result", style=me.Style(color=SUCCESS, font_size=13))
            me.text(
                state.infer_result,
                style=me.Style(
                    color=TEXT_PRIMARY,
                    font_size=12,
                    font_family="monospace",
                    white_space="pre",
                    margin=me.Margin(top=8),
                ),
            )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


@me.page(
    path="/",
    title="Semantic Image Clustering",
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap"
    ],
)
def main_page() -> None:
    """Root page: navigation + tab-switched content panels."""
    state = me.state(AppState)

    # ---- App shell ----
    with me.box(
        style=me.Style(
            background=BG_DARK,
            min_height="100vh",
            font_family="'Space Grotesk', sans-serif",
            color=TEXT_PRIMARY,
        )
    ):
        # ---- Header ----
        with me.box(
            style=me.Style(
                background=BG_CARD,
                border=me.Border(bottom=me.BorderSide(color=BORDER_COLOR, width=1)),
                padding=me.Padding.symmetric(vertical=16, horizontal=32),
                display="flex",
                flex_direction="row",
                align_items="center",
                justify_content="space_between",
            )
        ):
            with me.box(style=me.Style(display="flex", align_items="center", gap=12)):
                me.text(
                    "🧠",
                    style=me.Style(font_size=28),
                )
                me.text(
                    "Semantic Image Clustering",
                    style=me.Style(
                        font_size=20,
                        font_weight="700",
                        color=TEXT_PRIMARY,
                        letter_spacing="0.5px",
                    ),
                )
            me.text(
                "ResNet50/EfficientNet · BLIP · Ollama · HDBSCAN/K-Means",
                style=me.Style(color=TEXT_SECONDARY, font_size=12),
            )

        # ---- Navigation tabs ----
        tabs = [
            ("run", "⚙️  Configure & Run"),
            ("results", "📊  Results"),
            ("inference", "🔍  Inference"),
        ]
        with me.box(
            style=me.Style(
                display="flex",
                flex_direction="row",
                background=BG_CARD,
                border=me.Border(bottom=me.BorderSide(color=BORDER_COLOR, width=1)),
                padding=me.Padding.symmetric(horizontal=32),
            )
        ):
            for tab_key, tab_label in tabs:
                is_active = state.active_tab == tab_key
                me.button(
                    tab_label,
                    key=tab_key,
                    on_click=on_tab_select,
                    style=me.Style(
                        background="transparent",
                        color=ACCENT_LIGHT if is_active else TEXT_SECONDARY,
                        border=me.Border(
                            bottom=me.BorderSide(
                                color=ACCENT if is_active else "transparent",
                                width=3,
                            )
                        ),
                        padding=me.Padding.symmetric(vertical=14, horizontal=20),
                        font_size=13,
                        font_weight="600" if is_active else "400",
                        cursor="pointer",
                    ),
                )

        # ---- Content area ----
        with me.box(
            style=me.Style(
                max_width=960,
                margin=me.Margin.symmetric(horizontal="auto"),
                padding=me.Padding.all(24),
            )
        ):
            if state.active_tab == "run":
                config_section()
                status_section()

            elif state.active_tab == "results":
                results_section()

            elif state.active_tab == "inference":
                inference_section()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with: python app.py
    # Or:       mesop app.py
    me.run()
