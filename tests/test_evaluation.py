"""Unit tests for evaluation metrics module."""

import numpy as np

from terrainlens.evaluation import (
    evaluate_correctness,
    evaluate_stability,
)


def test_evaluate_correctness_perfect():
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:20, 10:20] = True

    res = evaluate_correctness(prediction=mask, ground_truth=mask)
    assert res.evaluation_type == "correctness"
    assert res.metrics["precision"] == 1.0
    assert res.metrics["recall"] == 1.0
    assert res.metrics["f1_score"] == 1.0
    assert res.metrics["iou_jaccard"] == 1.0
    assert res.metrics["true_positives"] == 100


def test_evaluate_correctness_partial():
    gt = np.zeros((32, 32), dtype=bool)
    gt[10:20, 10:20] = True  # 100 pixels

    pred = np.zeros((32, 32), dtype=bool)
    pred[10:15, 10:20] = True  # 50 TP
    pred[25:30, 10:20] = True  # 50 FP

    res = evaluate_correctness(prediction=pred, ground_truth=gt)
    assert res.metrics["true_positives"] == 50
    assert res.metrics["false_positives"] == 50
    assert res.metrics["false_negatives"] == 50
    assert res.metrics["precision"] == 0.5
    assert res.metrics["recall"] == 0.5
    assert res.metrics["f1_score"] == 0.5


def test_evaluate_stability_identical_baseline():
    base = np.zeros((32, 32), dtype=np.uint8)
    base[5:15, 5:15] = 255

    res = evaluate_stability(variant_map=base, baseline_map=base)
    assert res.evaluation_type == "stability"
    assert res.metrics["stability_dice"] == 1.0
    assert res.metrics["stability_iou"] == 1.0
    assert res.metrics["ssim"] == 1.0
    assert res.metrics["mse"] == 0.0
    assert res.metrics["feature_retention_pct"] == 100.0
    assert res.metrics["feature_delta_pct"] == 0.0


def test_evaluate_stability_degraded():
    base = np.zeros((32, 32), dtype=np.uint8)
    base[10:20, 10:20] = 255  # 100 pixels

    variant = np.zeros((32, 32), dtype=np.uint8)
    variant[10:15, 10:20] = 255  # lost 50% pixels

    res = evaluate_stability(variant_map=variant, baseline_map=base)
    assert res.metrics["feature_retention_pct"] == 50.0
    assert res.metrics["feature_delta_pct"] == -50.0
    assert res.metrics["stability_dice"] < 1.0
    assert res.metrics["mse"] > 0.0
