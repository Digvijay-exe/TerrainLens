"""Evaluation metrics module for TerrainLens.

Computes both Ground-Truth Correctness metrics (Precision, Recall, F1, IoU)
and Baseline Stability metrics (Dice retention, SSIM, MSE, pixel delta %)
with clear scientific categorization and execution timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from skimage.metrics import structural_similarity as compute_ssim


@dataclass
class EvaluationResult:
    """Encapsulates evaluated performance metrics and operational metadata."""

    evaluation_type: Literal["correctness", "stability"]
    target_label: str
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_ms: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_type": self.evaluation_type,
            "target_label": self.target_label,
            "runtime_ms": self.runtime_ms,
            "notes": self.notes,
            **self.metrics,
        }


def binarize_map(feature_map: np.ndarray, threshold: float = 127.0) -> np.ndarray:
    """Converts a feature map into a boolean binary mask."""
    if feature_map.dtype == bool:
        return feature_map
    return feature_map > threshold


def evaluate_correctness(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    label: str = "Prediction vs Reference",
) -> EvaluationResult:
    """Computes Ground-Truth Correctness metrics (Precision, Recall, F1, IoU).

    Used when an explicit reference ground-truth mask is provided.

    Args:
        prediction: Feature map to test.
        ground_truth: Ground truth reference mask.
        label: Descriptive label for this comparison.
    """
    t_start = time.perf_counter()
    pred_bin = binarize_map(prediction)
    gt_bin = binarize_map(ground_truth)

    if pred_bin.shape != gt_bin.shape:
        raise ValueError(
            f"Shape mismatch in evaluation: prediction {pred_bin.shape} vs ground truth {gt_bin.shape}"
        )

    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())
    tn = int(np.logical_and(~pred_bin, ~gt_bin).sum())
    total = int(pred_bin.size)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    dice = float(2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    runtime_ms = (time.perf_counter() - t_start) * 1000.0

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "iou_jaccard": round(iou, 4),
        "dice_similarity": round(dice, 4),
        "accuracy": round(accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }

    notes = (
        "Correctness evaluation relative to validated ground-truth reference. "
        "High F1 and IoU indicate high detection fidelity."
    )

    return EvaluationResult(
        evaluation_type="correctness",
        target_label=label,
        metrics=metrics,
        runtime_ms=round(runtime_ms, 3),
        notes=notes,
    )


def evaluate_stability(
    variant_map: np.ndarray,
    baseline_map: np.ndarray,
    label: str = "Degraded Variant vs Baseline",
) -> EvaluationResult:
    """Computes Baseline Stability metrics between degraded variant and pristine baseline.

    Explicitly labeled as 'stability' to ensure it is never confused with
    accuracy or correctness.

    Args:
        variant_map: Feature map extracted from degraded image.
        baseline_map: Feature map extracted from pristine baseline.
        label: Variant description.
    """
    t_start = time.perf_counter()

    if variant_map.shape != baseline_map.shape:
        raise ValueError(
            f"Shape mismatch: variant {variant_map.shape} vs baseline {baseline_map.shape}"
        )

    var_f = variant_map.astype(np.float32)
    base_f = baseline_map.astype(np.float32)

    # Pixel error metrics
    mse = float(np.mean((var_f - base_f) ** 2))
    rmse = float(np.sqrt(mse))

    # Normalized range for SSIM
    data_range = 255.0 if max(var_f.max(), base_f.max()) > 1.0 else 1.0
    try:
        ssim_val = float(compute_ssim(var_f, base_f, data_range=data_range))
    except (ValueError, TypeError, ZeroDivisionError):
        ssim_val = 0.0

    # Binary feature overlap (retention / overlap stability)
    var_bin = binarize_map(variant_map)
    base_bin = binarize_map(baseline_map)

    intersection = int(np.logical_and(var_bin, base_bin).sum())
    union = int(np.logical_or(var_bin, base_bin).sum())
    var_count = int(var_bin.sum())
    base_count = int(base_bin.sum())

    # Jaccard stability (IoU with baseline)
    stability_iou = float(intersection / union) if union > 0 else 1.0

    # Dice stability / retention rate
    denom = var_count + base_count
    stability_dice = float(2.0 * intersection / denom) if denom > 0 else 1.0

    # Feature pixel density retention percentage
    pixel_retention_pct = (
        float((var_count / base_count) * 100.0) if base_count > 0 else 100.0
    )
    pixel_delta_pct = (
        float(((var_count - base_count) / base_count) * 100.0) if base_count > 0 else 0.0
    )

    runtime_ms = (time.perf_counter() - t_start) * 1000.0

    metrics = {
        "stability_dice": round(stability_dice, 4),
        "stability_iou": round(stability_iou, 4),
        "ssim": round(ssim_val, 4),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "feature_retention_pct": round(pixel_retention_pct, 2),
        "feature_delta_pct": round(pixel_delta_pct, 2),
        "baseline_feature_pixels": base_count,
        "variant_feature_pixels": var_count,
    }

    notes = (
        "Stability evaluation relative to pristine uncorrupted baseline. "
        "Values measure operational robustness against degradation, NOT ground-truth accuracy."
    )

    return EvaluationResult(
        evaluation_type="stability",
        target_label=label,
        metrics=metrics,
        runtime_ms=round(runtime_ms, 3),
        notes=notes,
    )
