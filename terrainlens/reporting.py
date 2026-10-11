"""Reporting and publication export module for TerrainLens.

Generates standardized Markdown research reports, tabular CSV metrics,
and publication-ready degradation sensitivity charts. All exports strictly
distinguish ground-truth correctness from baseline stability metrics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_markdown_report(
    config: dict[str, Any],
    metrics_df: pd.DataFrame,
) -> str:
    """Produces a rigorous, concise scientific Markdown experiment report."""
    dataset = config.get("dataset", {})
    feat_cfg = config.get("feature_extraction", {})
    env = config.get("environment", {})
    mode = dataset.get("mode", "terrain_image")
    has_ref = config.get("has_reference_mask", False)

    mode_title = (
        "Elevation Raster (Digital Elevation Model)"
        if mode == "elevation_raster"
        else "Terrain Visual Photographic Image"
    )

    md = []
    md.append(f"# TerrainLens Experiment Report: {config.get('name', 'Terrain Study')}")
    md.append(f"**Experiment ID:** `{config.get('experiment_id')}`  ")
    md.append(f"**Generated:** {config.get('completed_at_utc', 'N/A')}  ")
    md.append(f"**Status:** {config.get('status', 'COMPLETED')}  \n")

    md.append("## 1. Executive Summary & Evaluation Paradigm")
    md.append(
        "> **Methodological Grounding Notice:**  \n"
        "> In accordance with reproducible computer vision standards, this study strictly delineates:\n"
        "> 1. **Correctness Metrics** (Precision, Recall, F1, IoU): Calculated exclusively when an independent ground-truth reference mask is available.\n"
        "> 2. **Stability Metrics** (Dice Retention, Jaccard Overlap, SSIM, MSE): Calculated against the pristine, uncorrupted baseline to measure algorithm resilience under environmental/sensor degradation.\n"
        "> 3. **Physical Elevation vs Visual Contrast**: Edge detection on photographic images captures optical reflectance and shading boundaries, whereas topographic contouring isolates physical elevation isocontours.\n"
    )

    md.append("## 2. Ingestion & Reproducibility Manifest")
    md.append(f"- **Dataset File:** `{dataset.get('file_name')}` ({dataset.get('file_size_bytes', 0):,} bytes)")
    md.append(f"- **Input SHA-256 Checksum:** `{dataset.get('checksum_sha256')}`")
    md.append(f"- **Spatial Dimensions:** {dataset.get('shape')} | **Data Type:** `{dataset.get('dtype')}`")
    md.append(f"- **Analysis Mode:** {mode_title}")
    md.append(f"- **Feature Extraction Method:** `{feat_cfg.get('method')}`")
    md.append(f"- **Ground Truth Reference Mask Present:** {'Yes (Correctness Evaluated)' if has_ref else 'No (Stability Analysis Only)'}")
    md.append("\n**Software Runtime Manifest:**")
    md.append(
        f"Python {env.get('python')} | OpenCV {env.get('opencv')} | NumPy {env.get('numpy')} | "
        f"Rasterio {env.get('rasterio')} | SciPy {env.get('scipy')} | scikit-image {env.get('scikit-image')}"
    )

    # Elevation Geomorphometry if available
    terrain_stats = config.get("terrain_statistics")
    if terrain_stats:
        md.append("\n## 3. Topographic Geomorphometry Summary")
        md.append("| Metric | Value | Units |")
        md.append("|---|---|---|")
        md.append(f"| Minimum Elevation | {terrain_stats.get('elevation_min_m')} | m |")
        md.append(f"| Maximum Elevation | {terrain_stats.get('elevation_max_m')} | m |")
        md.append(f"| Elevation Range (Relief) | {terrain_stats.get('elevation_range_m')} | m |")
        md.append(f"| Mean Elevation | {terrain_stats.get('elevation_mean_m')} | m |")
        md.append(f"| Mean Slope | {terrain_stats.get('slope_mean_deg')} | degrees |")
        md.append(f"| Maximum Slope | {terrain_stats.get('slope_max_deg')} | degrees |")
        md.append(f"| Surface Roughness Index | {terrain_stats.get('surface_roughness')} | index |")

    # Metrics Table
    md.append("\n## 4. Quantitative Results: Baseline vs Degraded Variants")
    md.append(
        "Below is the complete metric audit. Note that **Stability Dice** and **SSIM** "
        "measure degradation robustness relative to the uncorrupted baseline."
    )

    if not metrics_df.empty:
        # Convert df to markdown
        md.append("\n" + metrics_df.to_markdown(index=False) + "\n")

    # Degradation Analysis Breakdown
    md.append("## 5. Robustness & Sensitivity Findings")
    if not metrics_df.empty:
        non_baseline = metrics_df[metrics_df["variant_id"] != "baseline_pristine"]
        if not non_baseline.empty:
            worst_dice_row = non_baseline.loc[non_baseline["stability_dice"].idxmin()]
            best_dice_row = non_baseline.loc[non_baseline["stability_dice"].idxmax()]
            md.append(
                f"- **Most Severe Feature Degradation:** `{worst_dice_row['variant_name']}` "
                f"exhibited lowest stability (Stability Dice: {worst_dice_row['stability_dice']:.4f}, "
                f"Feature Delta: {worst_dice_row['feature_delta_pct']:+.1f}%)."
            )
            md.append(
                f"- **Highest Algorithm Resilience:** `{best_dice_row['variant_name']}` "
                f"maintained optimal retention (Stability Dice: {best_dice_row['stability_dice']:.4f}, "
                f"SSIM: {best_dice_row['ssim']:.4f})."
            )

    md.append("\n## 6. Scientific Limitations & Reproducibility Instructions")
    md.append(
        "1. **Resolution Degradation:** Resampling down to 50% followed by nearest-neighbor restoration simulates pixel quantization noise; edge continuity will be stepped.\n"
        "2. **Single-Band GeoTIFF Assumption:** Topographic analysis evaluates Band 1 elevation values. Multi-band orthomosaics require terrain image edge mode.\n"
        "3. **Re-Execution:** This exact experiment can be reproduced programmatically by feeding `config.json` into `terrainlens.experiment.Experiment.run()`.\n"
    )

    return "\n".join(md)


def create_robustness_plots(metrics_df: pd.DataFrame) -> go.Figure:
    """Generates dual-panel Plotly sensitivity charts comparing stability and feature counts."""
    if metrics_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Metrics Data Available")
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Feature Stability & SSIM vs Pristine Baseline",
            "Feature Pixels & Detection Volume by Variant",
        ),
    )

    # Subplot 1: Stability Dice & SSIM
    fig.add_trace(
        go.Bar(
            x=metrics_df["variant_name"],
            y=metrics_df["stability_dice"],
            name="Stability Dice (Retention)",
            marker_color="#2563EB",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=metrics_df["variant_name"],
            y=metrics_df["ssim"],
            name="SSIM (Structural Similarity)",
            marker_color="#10B981",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Feature Pixels count
    fig.add_trace(
        go.Bar(
            x=metrics_df["variant_name"],
            y=metrics_df["feature_pixels"],
            name="Feature Pixels",
            marker_color="#F59E0B",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        height=450,
        margin={"l": 20, "r": 20, "t": 50, "b": 100},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.35, "xanchor": "center", "x": 0.5},
    )
    fig.update_xaxes(tickangle=-25)
    return fig
