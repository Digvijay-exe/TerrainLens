"""Controlled image degradation module for TerrainLens.

Simulates environmental and sensor degradations (optical blur, additive Gaussian
noise, illumination/contrast variations, and spatial resolution loss) with
deterministic parameter logging. The original input is never modified in-place.
"""

from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Applies Gaussian atmospheric or optical blur degradation.

    Args:
        image: Original array (uint8 or float).
        kernel_size: Odd integer kernel size (e.g. 3, 5, 7, 9).
        sigma: Standard deviation of blur kernel.
    """
    arr = image.copy()
    k = int(kernel_size)
    if k <= 1:
        return arr, {"type": "gaussian_blur", "kernel_size": 1, "sigma": 0.0}
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(arr, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    meta = {
        "type": "gaussian_blur",
        "label": f"Gaussian Blur (k={k}, σ={sigma})",
        "kernel_size": k,
        "sigma": float(sigma),
    }
    return blurred, meta


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float = 20.0,
    seed: int | None = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Injects additive zero-mean Gaussian sensor noise.

    Args:
        image: Original image array.
        sigma: Noise standard deviation (in pixel intensity units for uint8, or relative).
        seed: Random seed for exact reproducibility.
    """
    arr = image.copy().astype(np.float32)
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=float(sigma), size=arr.shape).astype(np.float32)

    noisy = arr + noise

    if image.dtype == np.uint8:
        degraded = np.clip(noisy, 0, 255).astype(np.uint8)
    else:
        # Floating-point elevation data
        degraded = noisy.astype(image.dtype)

    meta = {
        "type": "gaussian_noise",
        "label": f"Gaussian Noise (σ={sigma})",
        "sigma": float(sigma),
        "seed": seed,
    }
    return degraded, meta


def apply_brightness_change(
    image: np.ndarray,
    offset: float = 0.0,
    scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Modifies visual brightness and contrast (illumination changes).

    Formula: out = image * scale + offset.

    Args:
        image: Original image array.
        offset: Additive brightness offset (e.g. -40 to +40).
        scale: Multiplicative contrast scale factor (e.g. 0.7 to 1.3).
    """
    arr = image.copy().astype(np.float32)
    modified = (arr * float(scale)) + float(offset)

    if image.dtype == np.uint8:
        degraded = np.clip(modified, 0, 255).astype(np.uint8)
    else:
        degraded = modified.astype(image.dtype)

    meta = {
        "type": "brightness",
        "label": f"Brightness/Contrast (scale={scale}, offset={offset})",
        "offset": float(offset),
        "scale": float(scale),
    }
    return degraded, meta


def apply_resolution_reduction(
    image: np.ndarray,
    scale_factor: float = 0.5,
    target_width: int | None = None,
    interpolation: str = "bilinear",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Simulates lower sensor spatial resolution.

    The image is downsampled by scale_factor and then resampled back to
    original dimensions so that degraded feature maps remain directly
    pixel-aligned for baseline and reference-mask evaluation.

    Args:
        image: Original image array.
        scale_factor: Downsampling factor (e.g. 0.5 for 50% spatial resolution).
        target_width: Alternative explicit downsampled width.
        interpolation: Resampling filter ('bilinear', 'nearest', 'area').
    """
    h_orig, w_orig = image.shape[:2]
    arr = image.copy()

    interp_map = {
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }
    down_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)

    if target_width is not None and target_width > 0:
        w_down = max(4, int(target_width))
        h_down = max(4, round(h_orig * (w_down / w_orig)))
        actual_scale = round(w_down / w_orig, 3)
    else:
        actual_scale = float(scale_factor)
        w_down = max(4, round(w_orig * actual_scale))
        h_down = max(4, round(h_orig * actual_scale))

    # Step 1: Downsample
    low_res = cv2.resize(arr, (w_down, h_down), interpolation=down_flag)

    # Step 2: Upsample back to original dimensions for pixel-aligned evaluation
    aligned_res = cv2.resize(low_res, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    meta = {
        "type": "resolution_reduction",
        "label": f"Resolution Reduction ({actual_scale*100:.0f}%, {w_down}x{h_down})",
        "scale_factor": actual_scale,
        "downsampled_dimensions": [w_down, h_down],
        "original_dimensions": [w_orig, h_orig],
        "interpolation": interpolation,
    }
    return aligned_res, meta


def apply_degradation(
    image: np.ndarray,
    degradation_type: Literal["gaussian_blur", "gaussian_noise", "brightness", "resolution_reduction"],
    **params: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatches degradation application and returns degraded image with parameter dict."""
    if degradation_type == "gaussian_blur":
        return apply_gaussian_blur(
            image=image,
            kernel_size=params.get("kernel_size", 5),
            sigma=params.get("sigma", 2.0),
        )
    elif degradation_type == "gaussian_noise":
        return apply_gaussian_noise(
            image=image,
            sigma=params.get("sigma", 20.0),
            seed=params.get("seed", 42),
        )
    elif degradation_type == "brightness":
        return apply_brightness_change(
            image=image,
            offset=params.get("offset", 0.0),
            scale=params.get("scale", 1.0),
        )
    elif degradation_type == "resolution_reduction":
        return apply_resolution_reduction(
            image=image,
            scale_factor=params.get("scale_factor", 0.5),
            target_width=params.get("target_width", None),
            interpolation=params.get("interpolation", "bilinear"),
        )
    raise ValueError(f"Unknown degradation type '{degradation_type}'.")


def get_default_degradation_suite() -> list[dict[str, Any]]:
    """Returns canonical 4-variant degradation suite specified for research benchmark."""
    return [
        {
            "id": "deg_blur_light",
            "type": "gaussian_blur",
            "name": "Gaussian Blur (k=5, σ=2.0)",
            "params": {"kernel_size": 5, "sigma": 2.0},
        },
        {
            "id": "deg_noise_moderate",
            "type": "gaussian_noise",
            "name": "Gaussian Noise (σ=25.0)",
            "params": {"sigma": 25.0, "seed": 42},
        },
        {
            "id": "deg_brightness_attenuated",
            "type": "brightness",
            "name": "Brightness/Contrast (-30, scale=0.85)",
            "params": {"offset": -30.0, "scale": 0.85},
        },
        {
            "id": "deg_res_half",
            "type": "resolution_reduction",
            "name": "Resolution 50% Resampled",
            "params": {"scale_factor": 0.5, "interpolation": "bilinear"},
        },
    ]
