"""Validation module for TerrainLens.

Validates input file properties, array dimensions, data types, missing values,
and file sizes according to scientific analysis constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MIN_DIMENSION = 16
MAX_DIMENSION = 10000
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_RASTER_EXTENSIONS = {".tif", ".tiff"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_RASTER_EXTENSIONS


@dataclass
class ValidationResult:
    """Holds the result of a validation check."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_file_path(file_path: str | Path) -> ValidationResult:
    """Validates existence, extension, and file size of an input file."""
    path = Path(file_path)
    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}

    if not path.exists():
        return ValidationResult(
            is_valid=False,
            errors=[f"File not found: {path}"],
            metadata=metadata,
        )

    if not path.is_file():
        return ValidationResult(
            is_valid=False,
            errors=[f"Path is not a regular file: {path}"],
            metadata=metadata,
        )

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ValidationResult(
            is_valid=False,
            errors=[
                f"Unsupported file format '{ext}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            ],
            metadata=metadata,
        )

    file_size = path.stat().st_size
    metadata["file_size_bytes"] = file_size
    metadata["file_size_kb"] = round(file_size / 1024, 2)
    metadata["file_size_mb"] = round(file_size / (1024 * 1024), 2)
    metadata["extension"] = ext

    if file_size == 0:
        errors.append("File is empty (0 bytes).")
    elif file_size > MAX_FILE_SIZE_BYTES:
        errors.append(
            f"File size ({metadata['file_size_mb']} MB) exceeds the maximum limit of {MAX_FILE_SIZE_BYTES / (1024 * 1024)} MB."
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )


def validate_raster_array(
    data: np.ndarray,
    nodata_value: float | None = None,
    mode: str = "terrain_image",
) -> ValidationResult:
    """Validates loaded NumPy raster or image array.

    Args:
        data: Array to inspect.
        nodata_value: Explicit nodata marker if present.
        mode: 'terrain_image' or 'elevation_raster'.
    """
    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}

    if not isinstance(data, np.ndarray):
        return ValidationResult(
            is_valid=False,
            errors=["Input data is not a NumPy ndarray."],
            metadata=metadata,
        )

    metadata["shape"] = list(data.shape)
    metadata["ndim"] = data.ndim
    metadata["dtype"] = str(data.dtype)

    if data.ndim < 2 or data.ndim > 3:
        errors.append(f"Expected 2D or 3D array, got shape {data.shape} ({data.ndim}D).")
        return ValidationResult(is_valid=False, errors=errors, metadata=metadata)

    height, width = data.shape[0], data.shape[1]
    metadata["height"] = height
    metadata["width"] = width

    if height < MIN_DIMENSION or width < MIN_DIMENSION:
        errors.append(
            f"Array dimensions ({width}x{height}) are smaller than minimum allowed ({MIN_DIMENSION}x{MIN_DIMENSION})."
        )

    if height > MAX_DIMENSION or width > MAX_DIMENSION:
        errors.append(
            f"Array dimensions ({width}x{height}) exceed maximum allowed limit ({MAX_DIMENSION}x{MAX_DIMENSION})."
        )

    # Check channels for 3D arrays
    if data.ndim == 3:
        channels = data.shape[2]
        metadata["channels"] = channels
        if mode == "elevation_raster" and channels > 1:
            warnings.append(
                f"Elevation raster mode received a multi-channel raster ({channels} channels). Single-band analysis will use band 1."
            )

    # Missing values check (NaN, Inf)
    nan_count = int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0
    inf_count = int(np.isinf(data).sum()) if np.issubdtype(data.dtype, np.floating) else 0
    nodata_count = 0
    if nodata_value is not None:
        nodata_count = int(np.isclose(data, nodata_value, equal_nan=True).sum())

    total_pixels = data.size
    metadata["total_pixels"] = total_pixels
    metadata["nan_count"] = nan_count
    metadata["inf_count"] = inf_count
    metadata["nodata_count"] = nodata_count

    if nan_count > 0:
        nan_pct = (nan_count / total_pixels) * 100
        metadata["nan_percentage"] = round(nan_pct, 2)
        if nan_pct > 50.0:
            errors.append(f"Raster contains over 50% NaN values ({nan_pct:.1f}%).")
        else:
            warnings.append(f"Raster contains {nan_count} ({nan_pct:.2f}%) NaN values.")

    if inf_count > 0:
        errors.append(f"Raster contains {inf_count} infinite values.")

    # Finite statistics
    finite_mask = np.isfinite(data)
    if nodata_value is not None:
        finite_mask &= ~np.isclose(data, nodata_value, equal_nan=True)

    if np.any(finite_mask):
        valid_vals = data[finite_mask]
        min_val = float(np.min(valid_vals))
        max_val = float(np.max(valid_vals))
        metadata["min_value"] = min_val
        metadata["max_value"] = max_val
        metadata["mean_value"] = round(float(np.mean(valid_vals)), 4)
        metadata["std_value"] = round(float(np.std(valid_vals)), 4)

        if min_val == max_val:
            warnings.append("Raster has zero variance (all valid pixels are identical).")
    else:
        errors.append("Raster has no valid finite pixels.")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )
