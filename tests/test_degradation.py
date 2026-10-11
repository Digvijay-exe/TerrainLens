"""Unit tests for degradation module."""

import numpy as np

from terrainlens.degradation import (
    apply_brightness_change,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_resolution_reduction,
    get_default_degradation_suite,
)


def test_gaussian_blur_non_destructive():
    orig = np.random.RandomState(0).randint(0, 255, (64, 64), dtype=np.uint8)
    orig_copy = orig.copy()

    blurred, meta = apply_gaussian_blur(orig, kernel_size=5, sigma=2.0)
    assert np.array_equal(orig, orig_copy), "Original array must never be mutated!"
    assert blurred.shape == orig.shape
    assert meta["kernel_size"] == 5
    assert meta["sigma"] == 2.0


def test_gaussian_noise_reproducibility():
    img = np.ones((64, 64), dtype=np.uint8) * 100

    noisy_1, meta_1 = apply_gaussian_noise(img, sigma=15.0, seed=123)
    noisy_2, _meta_2 = apply_gaussian_noise(img, sigma=15.0, seed=123)
    noisy_diff, _ = apply_gaussian_noise(img, sigma=15.0, seed=999)

    assert np.array_equal(noisy_1, noisy_2), "Deterministic seed must produce identical noise."
    assert not np.array_equal(noisy_1, noisy_diff)
    assert meta_1["sigma"] == 15.0


def test_brightness_change():
    img = np.ones((32, 32), dtype=np.uint8) * 100

    brighter, _meta_b = apply_brightness_change(img, offset=30.0, scale=1.0)
    assert np.all(brighter == 130)

    dimmer, _ = apply_brightness_change(img, offset=-40.0, scale=1.0)
    assert np.all(dimmer == 60)


def test_resolution_reduction_dimensions_aligned():
    img = np.random.RandomState(42).randint(0, 255, (100, 100), dtype=np.uint8)

    downscaled, meta = apply_resolution_reduction(img, scale_factor=0.5)
    # Output must match original dimensions for pixel-aligned evaluation
    assert downscaled.shape == img.shape
    assert meta["scale_factor"] == 0.5
    assert meta["downsampled_dimensions"] == [50, 50]


def test_degradation_suite():
    suite = get_default_degradation_suite()
    assert len(suite) == 4
    types = [s["type"] for s in suite]
    assert "gaussian_blur" in types
    assert "gaussian_noise" in types
    assert "brightness" in types
    assert "resolution_reduction" in types
