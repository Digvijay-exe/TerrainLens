"""Feature extraction module for TerrainLens.

Implements visual edge and gradient feature extraction algorithms (Canny and Sobel)
for Terrain Images. Note: These algorithms detect photographic visual edge contrast
and must never be conflated with elevation contours.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import cv2
import numpy as np

from terrainlens.preprocessing import to_grayscale


def extract_canny(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extracts photographic visual edges using the Canny edge detector.

    Args:
        image: 2D uint8 grayscale image (or RGB will be converted).
        low_threshold: First hysteresis threshold.
        high_threshold: Second hysteresis threshold.
        aperture_size: Sobel aperture size (3, 5, or 7).
        l2_gradient: L2 norm flag for gradient magnitude computation.

    Returns:
        Tuple of (binary_edge_map [0 or 255], parameter_and_stats_dict).
    """
    gray = to_grayscale(image)
    start_time = time.perf_counter()

    edges = cv2.Canny(
        image=gray,
        threshold1=int(low_threshold),
        threshold2=int(high_threshold),
        apertureSize=int(aperture_size),
        L2gradient=bool(l2_gradient),
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    edge_pixels = int(np.count_nonzero(edges))
    total_pixels = int(edges.size)
    edge_density_pct = (edge_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    meta = {
        "method": "canny",
        "method_label": "Canny Edge Detector (Visual Contrast)",
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "aperture_size": aperture_size,
        "l2_gradient": l2_gradient,
        "edge_pixels": edge_pixels,
        "total_pixels": total_pixels,
        "feature_density_pct": round(edge_density_pct, 4),
        "runtime_ms": round(elapsed_ms, 3),
    }
    return edges, meta


def extract_sobel(
    image: np.ndarray,
    ksize: int = 3,
    scale: float = 1.0,
    delta: float = 0.0,
    threshold: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extracts visual gradient features using Sobel directional operators.

    Args:
        image: 2D uint8 image.
        ksize: Aperture size (1, 3, 5, or 7).
        scale: Optional scale factor for computed derivative values.
        delta: Optional delta value added to the results.
        threshold: If specified (e.g. 50), produces a binary edge mask [0, 255].
                   Otherwise, produces normalized gradient magnitude [0..255].

    Returns:
        Tuple of (gradient_or_binary_map, parameter_and_stats_dict).
    """
    gray = to_grayscale(image)
    start_time = time.perf_counter()

    # Calculate horizontal and vertical derivatives in float32 to avoid overflow
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=int(ksize), scale=scale, delta=delta)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=int(ksize), scale=scale, delta=delta)

    # Gradient magnitude
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize magnitude to 0-255 uint8 range
    max_val = float(np.max(magnitude))
    if max_val > 0:
        norm_mag = np.clip((magnitude / max_val) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm_mag = np.zeros_like(gray, dtype=np.uint8)

    if threshold is not None:
        # Binary thresholding
        _, output_map = cv2.threshold(norm_mag, float(threshold), 255, cv2.THRESH_BINARY)
        feature_pixels = int(np.count_nonzero(output_map))
    else:
        output_map = norm_mag
        # Consider strong gradient pixels (>30) for density estimate
        feature_pixels = int(np.count_nonzero(output_map > 30))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    total_pixels = int(output_map.size)
    feature_density_pct = (feature_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    meta = {
        "method": "sobel",
        "method_label": "Sobel Gradient Magnitude (Visual Intensity Derivative)",
        "ksize": ksize,
        "scale": scale,
        "delta": delta,
        "threshold": threshold,
        "max_gradient_raw": round(max_val, 2),
        "feature_pixels": feature_pixels,
        "total_pixels": total_pixels,
        "feature_density_pct": round(feature_density_pct, 4),
        "runtime_ms": round(elapsed_ms, 3),
    }
    return output_map, meta


def extract_features(
    image: np.ndarray,
    method: Literal["canny", "sobel"] = "canny",
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatches visual feature extraction based on user configuration."""
    if method == "canny":
        return extract_canny(
            image=image,
            low_threshold=kwargs.get("low_threshold", 50),
            high_threshold=kwargs.get("high_threshold", 150),
            aperture_size=kwargs.get("aperture_size", 3),
            l2_gradient=kwargs.get("l2_gradient", False),
        )
    elif method == "sobel":
        return extract_sobel(
            image=image,
            ksize=kwargs.get("ksize", 3),
            scale=kwargs.get("scale", 1.0),
            delta=kwargs.get("delta", 0.0),
            threshold=kwargs.get("threshold", None),
        )
    raise ValueError(f"Unknown feature extraction method '{method}'. Choose 'canny' or 'sobel'.")
