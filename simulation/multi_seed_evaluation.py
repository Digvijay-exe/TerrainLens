"""Repeated-seed evaluation of synthetic GPS and odometry fusion."""

import numpy as np

from simulation.localization import evaluate_localization


def run_multi_seed_evaluation(
    true_positions,
    num_seeds=30,
    base_seed=100,
    gps_noise=2.0,
    odometry_noise=0.25,
    gps_gain=0.25,
):
    if num_seeds < 2:
        raise ValueError("num_seeds must be at least 2")
    if gps_noise < 0 or odometry_noise < 0:
        raise ValueError("Noise values must be non-negative")
    if not 0 <= gps_gain <= 1:
        raise ValueError("gps_gain must be between 0 and 1")

    rows = []

    for seed in range(int(base_seed), int(base_seed) + int(num_seeds)):
        result = evaluate_localization(
            true_positions,
            gps_noise=float(gps_noise),
            odometry_noise=float(odometry_noise),
            gps_gain=float(gps_gain),
            seed=seed,
        )

        rows.append({
            "seed": seed,
            "gps_rmse": result["gps_rmse"],
            "fused_rmse": result["fused_rmse"],
            "improvement_percent": result["improvement_percent"],
            "fusion_better": result["fused_rmse"] < result["gps_rmse"],
        })

    gps = np.array([r["gps_rmse"] for r in rows], dtype=float)
    fused = np.array([r["fused_rmse"] for r in rows], dtype=float)
    improvements = np.array(
        [r["improvement_percent"] for r in rows], dtype=float
    )

    def summary(values):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        margin = 1.96 * std / np.sqrt(len(values))
        return {
            "mean": mean,
            "median": float(np.median(values)),
            "std": std,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
        }

    return {
        "runs": rows,
        "num_seeds": len(rows),
        "base_seed": int(base_seed),
        "gps_noise": float(gps_noise),
        "odometry_noise": float(odometry_noise),
        "gps_gain": float(gps_gain),
        "gps": summary(gps),
        "fused": summary(fused),
        "improvement": summary(improvements),
        "fusion_better_percent": 100.0 * np.mean(
            [r["fusion_better"] for r in rows]
        ),
        "gps_values": gps,
        "fused_values": fused,
        "improvement_values": improvements,
    }
