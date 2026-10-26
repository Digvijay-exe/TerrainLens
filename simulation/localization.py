"""Synthetic GPS/odometry localization experiment."""

import numpy as np


def evaluate_localization(
    true_positions,
    gps_noise=2.0,
    odometry_noise=0.25,
    gps_gain=0.25,
    seed=21,
):
    """Compare noisy GPS and GPS/odometry fusion against ground truth.

    Coordinates and noise are expressed in grid-cell units.
    This is a simplified experimental model, not a physical sensor model.
    """
    truth = np.asarray(true_positions, dtype=float)

    if truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("true_positions must have shape (N, 2)")
    if len(truth) < 2:
        raise ValueError("At least two position samples are required")
    if gps_noise < 0 or odometry_noise < 0:
        raise ValueError("Noise values cannot be negative")
    if not 0.0 <= gps_gain <= 1.0:
        raise ValueError("gps_gain must be between 0 and 1")

    rng = np.random.default_rng(seed)

    gps = truth + rng.normal(0, gps_noise, truth.shape)

    # Simulated odometry measures each movement with random error.
    true_steps = np.diff(truth, axis=0)
    odometry_steps = true_steps + rng.normal(
        0, odometry_noise, true_steps.shape
    )

    fused = np.zeros_like(truth)
    fused[0] = gps[0]

    for i in range(1, len(truth)):
        predicted = fused[i - 1] + odometry_steps[i - 1]
        fused[i] = predicted + gps_gain * (gps[i] - predicted)

    gps_error = np.linalg.norm(gps - truth, axis=1)
    fused_error = np.linalg.norm(fused - truth, axis=1)

    def rmse(errors):
        return float(np.sqrt(np.mean(errors**2)))

    return {
        "truth": truth,
        "gps": gps,
        "fused": fused,
        "gps_error": gps_error,
        "fused_error": fused_error,
        "gps_rmse": rmse(gps_error),
        "fused_rmse": rmse(fused_error),
        "improvement_percent": (
            100 * (rmse(gps_error) - rmse(fused_error)) / rmse(gps_error)
            if rmse(gps_error) > 0
            else 0.0
        ),
    }
