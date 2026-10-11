"""Experiment orchestration and reproducibility manager for TerrainLens.

Coordinates end-to-end experimental pipelines, records library dependency
manifests and cryptographic input checksums, generates persistent experiment
artifacts, and supports 100% reproducible re-execution from JSON configs.
"""

from __future__ import annotations

import datetime
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import terrainlens
from terrainlens.data_loader import LoadedData
from terrainlens.degradation import apply_degradation, get_default_degradation_suite
from terrainlens.evaluation import evaluate_correctness, evaluate_stability
from terrainlens.feature_extraction import extract_features
from terrainlens.preprocessing import (
    normalize_for_visualization,
    preprocess_pipeline,
)
from terrainlens.terrain_analysis import analyze_elevation_raster


def get_library_versions() -> dict[str, str]:
    """Inspects and returns current runtime library versions for reproducibility."""
    import PIL
    import rasterio
    import scipy
    import skimage

    return {
        "python": sys.version.split()[0],
        "terrainlens": terrainlens.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "scipy": scipy.__version__,
        "scikit-image": skimage.__version__,
        "pillow": PIL.__version__,
        "pandas": pd.__version__,
    }


class Experiment:
    """Orchestrates an end-to-end TerrainLens scientific experiment."""

    def __init__(
        self,
        name: str = "Terrain Analysis Experiment",
        description: str = "",
        results_dir: str | Path = "results",
    ):
        self.experiment_id = f"EXP-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.name = name
        self.description = description
        self.results_dir = Path(results_dir) / self.experiment_id
        self.created_at = datetime.datetime.now(datetime.UTC).isoformat()
        self.status = "INITIALIZED"

        # State storage
        self.config: dict[str, Any] = {}
        self.input_data: LoadedData | None = None
        self.reference_mask: np.ndarray | None = None
        self.baseline_image: np.ndarray | None = None
        self.baseline_features: np.ndarray | None = None
        self.baseline_meta: dict[str, Any] = {}
        self.elevation_hillshade: np.ndarray | None = None
        self.elevation_stats: dict[str, Any] = {}
        self.variants: list[dict[str, Any]] = []
        self.metrics_table: pd.DataFrame = pd.DataFrame()

    def run(
        self,
        loaded_data: LoadedData,
        feature_method: str = "canny",
        feature_params: dict[str, Any] | None = None,
        preprocessing_params: dict[str, Any] | None = None,
        degradation_suite: list[dict[str, Any]] | None = None,
        reference_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Executes full experiment pipeline."""
        self.input_data = loaded_data
        self.reference_mask = reference_mask
        self.status = "RUNNING"

        feature_params = feature_params or {}
        preprocessing_params = preprocessing_params or {}
        if degradation_suite is None:
            degradation_suite = get_default_degradation_suite()

        mode = loaded_data.mode

        # Step 1: Preprocessing
        preprocessed, prep_record = preprocess_pipeline(
            data=loaded_data.data,
            mode=mode,
            nodata=loaded_data.nodata,
            smooth=preprocessing_params.get("smooth", True),
            kernel_size=preprocessing_params.get("kernel_size", 5),
            sigma=preprocessing_params.get("sigma", 1.0),
            resize_factor=preprocessing_params.get("resize_factor", 1.0),
        )

        # Baseline visualization array
        if mode == "elevation_raster":
            self.baseline_image = normalize_for_visualization(preprocessed, nodata=loaded_data.nodata)
            # Analytical DEM contours + hillshade + geomorphology
            analysis_res = analyze_elevation_raster(
                elevation=preprocessed,
                contour_interval=feature_params.get("contour_interval", None),
                num_levels=feature_params.get("num_levels", 10),
                hillshade_azimuth=feature_params.get("hillshade_azimuth", 315.0),
                hillshade_altitude=feature_params.get("hillshade_altitude", 45.0),
                nodata=loaded_data.nodata,
            )
            self.baseline_features = analysis_res["contour_map"]
            self.baseline_meta = analysis_res["contour_meta"]
            self.elevation_hillshade = analysis_res["hillshade"]
            self.elevation_stats = analysis_res["stats"]
        else:
            self.baseline_image = preprocessed.copy()
            # Visual edge detection
            feats, f_meta = extract_features(
                image=preprocessed,
                method=feature_method,  # type: ignore
                **feature_params,
            )
            self.baseline_features = feats
            self.baseline_meta = f_meta

        # Evaluate baseline correctness if reference mask provided
        baseline_eval_dict: dict[str, Any] = {}
        if reference_mask is not None and reference_mask.shape == self.baseline_features.shape:
            corr_res = evaluate_correctness(
                prediction=self.baseline_features,
                ground_truth=reference_mask,
                label="Baseline vs Ground Truth Reference",
            )
            baseline_eval_dict = corr_res.to_dict()

        # Build baseline metrics row
        rows: list[dict[str, Any]] = []
        base_row: dict[str, Any] = {
            "variant_id": "baseline_pristine",
            "variant_name": "Pristine Baseline (Uncorrupted)",
            "degradation_type": "none",
            "feature_pixels": int(np.count_nonzero(self.baseline_features)),
            "feature_density_pct": self.baseline_meta.get("feature_density_pct", 0.0),
            "runtime_ms": self.baseline_meta.get("runtime_ms", 0.0),
            # Stability vs baseline is perfect for baseline
            "stability_dice": 1.0,
            "stability_iou": 1.0,
            "ssim": 1.0,
            "mse": 0.0,
            "rmse": 0.0,
            "feature_retention_pct": 100.0,
            "feature_delta_pct": 0.0,
        }
        if baseline_eval_dict:
            base_row.update({
                "correctness_f1": baseline_eval_dict.get("f1_score"),
                "correctness_precision": baseline_eval_dict.get("precision"),
                "correctness_recall": baseline_eval_dict.get("recall"),
                "correctness_iou": baseline_eval_dict.get("iou_jaccard"),
            })
        rows.append(base_row)

        # Step 2: Degradation loop
        self.variants = []
        for deg_spec in degradation_suite:
            deg_id = deg_spec["id"]
            deg_type = deg_spec["type"]
            deg_params = deg_spec.get("params", {})
            deg_name = deg_spec.get("name", deg_id)

            # Degrade original baseline input
            degraded_img, _deg_meta = apply_degradation(
                image=self.baseline_image,
                degradation_type=deg_type,
                **deg_params,
            )

            # Extract features on degraded image
            if mode == "elevation_raster":
                # For elevation rasters, degraded image is analyzed as simulated DEM
                elev_degraded = degraded_img.astype(np.float32)
                var_analysis = analyze_elevation_raster(
                    elevation=elev_degraded,
                    contour_interval=feature_params.get("contour_interval", None),
                    num_levels=feature_params.get("num_levels", 10),
                    hillshade_azimuth=feature_params.get("hillshade_azimuth", 315.0),
                    hillshade_altitude=feature_params.get("hillshade_altitude", 45.0),
                )
                var_feats = var_analysis["contour_map"]
                var_meta = var_analysis["contour_meta"]
                var_hillshade = var_analysis["hillshade"]
            else:
                var_feats, var_meta = extract_features(
                    image=degraded_img,
                    method=feature_method,  # type: ignore
                    **feature_params,
                )
                var_hillshade = None

            # Calculate stability vs pristine baseline
            stab_res = evaluate_stability(
                variant_map=var_feats,
                baseline_map=self.baseline_features,
                label=f"Stability: {deg_name} vs Baseline",
            )

            variant_record: dict[str, Any] = {
                "id": deg_id,
                "name": deg_name,
                "degradation_type": deg_type,
                "degradation_params": deg_params,
                "degraded_image": degraded_img,
                "features": var_feats,
                "hillshade": var_hillshade,
                "feature_meta": var_meta,
                "stability": stab_res.to_dict(),
            }

            # Calculate correctness if reference mask exists
            if reference_mask is not None and reference_mask.shape == var_feats.shape:
                var_corr = evaluate_correctness(
                    prediction=var_feats,
                    ground_truth=reference_mask,
                    label=f"Correctness: {deg_name} vs Ground Truth",
                )
                variant_record["correctness"] = var_corr.to_dict()

            self.variants.append(variant_record)

            # Add to tabular rows
            var_row: dict[str, Any] = {
                "variant_id": deg_id,
                "variant_name": deg_name,
                "degradation_type": deg_type,
                "feature_pixels": int(np.count_nonzero(var_feats)),
                "feature_density_pct": var_meta.get("feature_density_pct", 0.0),
                "runtime_ms": var_meta.get("runtime_ms", 0.0),
                "stability_dice": stab_res.metrics.get("stability_dice", 0.0),
                "stability_iou": stab_res.metrics.get("stability_iou", 0.0),
                "ssim": stab_res.metrics.get("ssim", 0.0),
                "mse": stab_res.metrics.get("mse", 0.0),
                "rmse": stab_res.metrics.get("rmse", 0.0),
                "feature_retention_pct": stab_res.metrics.get("feature_retention_pct", 0.0),
                "feature_delta_pct": stab_res.metrics.get("feature_delta_pct", 0.0),
            }
            if "correctness" in variant_record:
                corr_m = variant_record["correctness"]
                var_row.update({
                    "correctness_f1": corr_m.get("f1_score"),
                    "correctness_precision": corr_m.get("precision"),
                    "correctness_recall": corr_m.get("recall"),
                    "correctness_iou": corr_m.get("iou_jaccard"),
                })
            rows.append(var_row)

        self.metrics_table = pd.DataFrame(rows)

        # Assemble full configuration and metadata dictionary
        self.config = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "created_at_utc": self.created_at,
            "completed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "COMPLETED",
            "environment": get_library_versions(),
            "dataset": {
                "file_name": loaded_data.file_name,
                "file_size_bytes": loaded_data.file_size_bytes,
                "checksum_sha256": loaded_data.checksum_sha256,
                "mode": loaded_data.mode,
                "shape": list(loaded_data.shape),
                "dtype": loaded_data.dtype_str,
                "nodata": loaded_data.nodata,
            },
            "has_reference_mask": reference_mask is not None,
            "preprocessing": prep_record,
            "feature_extraction": {
                "method": feature_method if mode == "terrain_image" else "topographic_contours",
                "parameters": feature_params,
            },
            "degradation_suite": degradation_suite,
            "terrain_statistics": self.elevation_stats if mode == "elevation_raster" else None,
        }

        self.status = "COMPLETED"
        self.save_artifacts()
        return self.config

    def save_artifacts(self) -> Path:
        """Persists config JSON, CSV metrics, and rendered image maps to local disk."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save config JSON
        config_path = self.results_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        # 2. Save metrics CSV
        csv_path = self.results_dir / "metrics.csv"
        self.metrics_table.to_csv(csv_path, index=False)

        # 3. Save Baseline Images
        if self.baseline_image is not None:
            Image.fromarray(self.baseline_image).save(self.results_dir / "baseline_input.png")
        if self.baseline_features is not None:
            Image.fromarray(self.baseline_features).save(self.results_dir / "baseline_features.png")
        if self.elevation_hillshade is not None:
            Image.fromarray(self.elevation_hillshade).save(self.results_dir / "baseline_hillshade.png")

        # 4. Save Variant Images
        for v in self.variants:
            vid = v["id"]
            if "degraded_image" in v and v["degraded_image"] is not None:
                Image.fromarray(v["degraded_image"]).save(self.results_dir / f"variant_{vid}_image.png")
            if "features" in v and v["features"] is not None:
                Image.fromarray(v["features"]).save(self.results_dir / f"variant_{vid}_features.png")

        return self.results_dir


def load_experiment_from_dir(experiment_dir: str | Path) -> dict[str, Any]:
    """Loads an existing experiment's config and metrics from disk."""
    exp_path = Path(experiment_dir)
    config_file = exp_path / "config.json"
    csv_file = exp_path / "metrics.csv"

    if not config_file.exists():
        raise FileNotFoundError(f"Missing config.json in {experiment_dir}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    metrics_df = pd.read_csv(csv_file) if csv_file.exists() else pd.DataFrame()

    return {
        "config": config,
        "metrics_df": metrics_df,
        "dir": exp_path,
    }


def list_saved_experiments(results_dir: str | Path = "results") -> list[dict[str, Any]]:
    """Discovers and catalogs all past experiments saved locally."""
    path = Path(results_dir)
    if not path.exists():
        return []

    experiments = []
    for exp_dir in sorted(path.iterdir(), reverse=True):
        if exp_dir.is_dir() and (exp_dir / "config.json").exists():
            try:
                with open(exp_dir / "config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                experiments.append({
                    "id": cfg.get("experiment_id", exp_dir.name),
                    "name": cfg.get("name", "Unnamed"),
                    "created_at": cfg.get("created_at_utc", ""),
                    "file_name": cfg.get("dataset", {}).get("file_name", "Unknown"),
                    "mode": cfg.get("dataset", {}).get("mode", "Unknown"),
                    "method": cfg.get("feature_extraction", {}).get("method", "Unknown"),
                    "path": str(exp_dir),
                })
            except (json.JSONDecodeError, OSError, KeyError):
                continue
    return experiments
