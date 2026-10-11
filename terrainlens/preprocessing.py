"""Preprocessing module for TerrainLens.

Provides deterministic data transformations including color conversion,
Gaussian smoothing, spatial resampling, dynamic range normalization,
missing-value imputation, and analytical hillshade computation.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy import ndimage


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts 2D or 3D image/raster array into 2D single-channel uint8 array.

    Args:
        image: Input array of shape (H, W) or (H, W, C).

    Returns:
        2D uint8 numpy array.
    """
    if image.ndim == 2:
        if image.dtype == np.uint8:
            return image.copy()
        # Float or int16/uint16 normalization
        return normalize_for_visualization(image)

    if image.ndim == 3:
        if image.shape[2] == 1:
            return to_grayscale(image[:, :, 0])
        elif image.shape[2] == 4:
            # Drop alpha or convert RGBA to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] >= 3:
            return cv2.cvtColor(image[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)

    raise ValueError(f"Unsupported image dimensions: {image.shape}")


def gaussian_smooth(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Applies 2D Gaussian smoothing to an image or elevation array.

    Args:
        image: 2D array.
        kernel_size: Odd integer kernel diameter (e.g. 3, 5, 7).
        sigma: Standard deviation of Gaussian distribution.
    """
    if kernel_size <= 0:
        return image.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def resize_raster(
    image: np.ndarray,
    scale_factor: float | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Resamples raster/image dimensions."""
    h, w = image.shape[:2]
    interp_flags = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }
    flag = interp_flags.get(interpolation.lower(), cv2.INTER_LINEAR)

    if scale_factor is not None:
        if scale_factor <= 0:
            raise ValueError(f"Scale factor must be positive, got {scale_factor}")
        new_w = max(1, round(w * scale_factor))
        new_h = max(1, round(h * scale_factor))
        return cv2.resize(image, (new_w, new_h), interpolation=flag)

    if target_width is not None or target_height is not None:
        new_w = target_width if target_width is not None else w
        new_h = target_height if target_height is not None else h
        return cv2.resize(image, (new_w, new_h), interpolation=flag)

    return image.copy()


def normalize_for_visualization(
    data: np.ndarray,
    nodata: float | None = None,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Normalizes an arbitrary numeric raster into an 8-bit [0, 255] visualization array.

    Replaces NaNs/nodata with local minimum prior to scaling.
    """
    arr = data.astype(np.float32, copy=True)
    mask_valid = np.isfinite(arr)
    if nodata is not None:
        mask_valid &= ~np.isclose(arr, nodata, equal_nan=True)

    if not np.any(mask_valid):
        return np.zeros(arr.shape[:2], dtype=np.uint8)

    valid_vals = arr[mask_valid]
    p_low, p_high = clip_percentiles
    vmin = np.percentile(valid_vals, p_low)
    vmax = np.percentile(valid_vals, p_high)

    if np.isclose(vmin, vmax):
        vmin = np.min(valid_vals)
        vmax = np.max(valid_vals)

    if np.isclose(vmin, vmax):
        out = np.zeros(arr.shape[:2], dtype=np.uint8)
        out[mask_valid] = 128
        return out

    arr_clipped = np.clip(arr, vmin, vmax)
    scaled = (arr_clipped - vmin) / (vmax - vmin) * 255.0
    scaled[~mask_valid] = 0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def handle_missing_elevation(
    elevation: np.ndarray,
    nodata: float | None = None,
    method: str = "nearest",
) -> np.ndarray:
    """Fills missing elevation cells (NaNs or designated nodata values).

    Methods: 'nearest' (Euclidean distance transform), 'median', or 'zero'.
    """
    arr = elevation.astype(np.float32, copy=True)
    invalid_mask = ~np.isfinite(arr)
    if nodata is not None:
        invalid_mask |= np.isclose(arr, nodata, equal_nan=True)

    if not np.any(invalid_mask):
        return arr

    if not np.any(~invalid_mask):
        # Entire raster is invalid
        return np.zeros_like(arr)

    if method == "nearest":
        # Distance transform indices fill
        indices = ndimage.distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
        arr[invalid_mask] = arr[tuple(indices)][invalid_mask]
    elif method == "median":
        valid_median = float(np.nanmedian(arr[~invalid_mask]))
        arr[invalid_mask] = valid_median
    else:
        arr[invalid_mask] = 0.0

    return arr


def calculate_hillshade(
    elevation: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    cell_size: float = 1.0,
) -> np.ndarray:
    """Computes standard cartographic hillshade illumination from an elevation array.

    Args:
        elevation: 2D array of elevations.
        azimuth: Solar azimuth in degrees (clockwise from North, 0-360). Default 315 (NW).
        altitude: Sun elevation angle above horizon in degrees (0-90). Default 45.
        z_factor: Multiplier for elevation exaggeration.
        cell_size: Spatial pixel resolution in ground units.

    Returns:
        2D uint8 hillshade array [0..255].
    """
    clean_elev = handle_missing_elevation(elevation)
    clean_elev = clean_elev * float(z_factor)

    # 3x3 Horn kernel gradient calculations
    # dz/dx and dz/dy
    kernel_dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / (8.0 * cell_size)
    kernel_dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / (8.0 * cell_size)

    dz_dx = cv2.filter2D(clean_elev, -1, kernel_dx)
    dz_dy = cv2.filter2D(clean_elev, -1, kernel_dy)

    # Slope in radians
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    # Aspect in radians
    aspect = np.arctan2(dz_dy, -dz_dx)
    # Map aspect to [0, 2pi]
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)

    # Illumination angles
    zenith_rad = np.radians(90.0 - altitude)
    azimuth_math = 360.0 - azimuth + 90.0
    if azimuth_math >= 360.0:
        azimuth_math -= 360.0
    azimuth_rad = np.radians(azimuth_math)

    # Shading formula: cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)
    shaded = np.cos(zenith_rad) * np.cos(slope) + np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    shaded = np.clip(shaded, 0.0, 1.0) * 255.0
    return shaded.astype(np.uint8)


def preprocess_pipeline(
    data: np.ndarray,
    mode: str = "terrain_image",
    nodata: float | None = None,
    smooth: bool = True,
    kernel_size: int = 5,
    sigma: float = 1.0,
    resize_factor: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Executes deterministic preprocessing pipeline and records audit configuration."""
    params_record: dict[str, Any] = {
        "mode": mode,
        "input_shape": list(data.shape),
        "input_dtype": str(data.dtype),
        "smooth_applied": smooth,
        "kernel_size": kernel_size if smooth else None,
        "sigma": sigma if smooth else None,
        "resize_factor": resize_factor,
        "nodata": nodata,
    }

    if mode == "elevation_raster":
        # Handle nodata/NaN first
        arr = handle_missing_elevation(data, nodata=nodata, method="nearest")
        if smooth and kernel_size > 0:
            arr = gaussian_smooth(arr, kernel_size=kernel_size, sigma=sigma)
        if resize_factor != 1.0:
            arr = resize_raster(arr, scale_factor=resize_factor, interpolation="bilinear")
        processed = arr
    else:
        # Visual terrain image
        gray = to_grayscale(data)
        if smooth and kernel_size > 0:
            gray = gaussian_smooth(gray, kernel_size=kernel_size, sigma=sigma)
        if resize_factor != 1.0:
            gray = resize_raster(gray, scale_factor=resize_factor, interpolation="bilinear")
        processed = gray

    params_record["output_shape"] = list(processed.shape)
    params_record["output_dtype"] = str(processed.dtype)
    return processed, params_record
