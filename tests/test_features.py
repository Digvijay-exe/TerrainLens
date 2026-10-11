"""Unit tests for feature extraction and terrain analysis modules."""

import numpy as np
import pytest

from terrainlens.feature_extraction import (
    extract_canny,
    extract_features,
    extract_sobel,
)
from terrainlens.terrain_analysis import (
    compute_terrain_statistics,
    extract_topographic_contours,
)


def test_extract_canny_basic():
    # Synthetic image with a bright vertical bar in the center
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 28:36] = 255

    edges, meta = extract_canny(img, low_threshold=50, high_threshold=150)
    assert edges.shape == (64, 64)
    assert edges.dtype == np.uint8
    assert meta["method"] == "canny"
    assert meta["edge_pixels"] > 0
    assert meta["runtime_ms"] >= 0.0

    # Values must be binary 0 or 255
    unique_vals = set(np.unique(edges))
    assert unique_vals.issubset({0, 255})


def test_extract_sobel_continuous_and_thresholded():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 200  # step edge

    # Continuous magnitude
    mag, meta_c = extract_sobel(img, ksize=3, threshold=None)
    assert mag.shape == (64, 64)
    assert meta_c["method"] == "sobel"
    assert meta_c["max_gradient_raw"] > 0

    # Thresholded binary
    bin_map, _meta_b = extract_sobel(img, ksize=3, threshold=50)
    assert bin_map.shape == (64, 64)
    unique_vals = set(np.unique(bin_map))
    assert unique_vals.issubset({0, 255})


def test_extract_features_dispatch():
    img = np.zeros((32, 32), dtype=np.uint8)
    _feats, meta = extract_features(img, method="canny", low_threshold=30, high_threshold=100)
    assert meta["low_threshold"] == 30

    with pytest.raises(ValueError):
        extract_features(img, method="unsupported_method")  # type: ignore


def test_topographic_contours():
    # Synthetic elevation ramp from 100m to 500m
    elev = np.linspace(100, 500, 64 * 64).reshape((64, 64)).astype(np.float32)

    contours, meta = extract_topographic_contours(elev, interval=50.0)
    assert contours.shape == (64, 64)
    assert meta["contour_interval"] == 50.0
    assert meta["num_levels"] > 0
    assert meta["feature_pixels"] > 0


def test_terrain_statistics():
    elev = np.array([
        [100.0, 110.0, 120.0],
        [100.0, 110.0, 120.0],
        [100.0, 110.0, 120.0],
    ], dtype=np.float32)

    stats = compute_terrain_statistics(elev, cell_size=1.0)
    assert stats["elevation_min_m"] == 100.0
    assert stats["elevation_max_m"] == 120.0
    assert stats["elevation_range_m"] == 20.0
    assert stats["slope_mean_deg"] > 0.0
