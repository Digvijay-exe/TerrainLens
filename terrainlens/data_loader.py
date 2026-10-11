"""Data loader module for TerrainLens.

Supports ingestion of standard photographic images (PNG/JPEG) and single-band
elevation rasters (GeoTIFF) with cryptographic checksum calculation and metadata
extraction.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
from PIL import Image

from terrainlens.validation import (
    SUPPORTED_RASTER_EXTENSIONS,
    ValidationResult,
    validate_file_path,
    validate_raster_array,
)


@dataclass
class LoadedData:
    """Encapsulates loaded dataset with scientific metadata and checksum."""

    data: np.ndarray
    mode: Literal["terrain_image", "elevation_raster"]
    file_name: str
    file_path: str | None = None
    checksum_sha256: str = ""
    file_size_bytes: int = 0
    dtype_str: str = ""
    shape: tuple[int, ...] = ()
    nodata: float | None = None
    crs: str | None = None
    transform: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    validation: ValidationResult | None = None

    @property
    def width(self) -> int:
        return self.shape[1] if len(self.shape) >= 2 else 0

    @property
    def height(self) -> int:
        return self.shape[0] if len(self.shape) >= 2 else 0

    @property
    def channels(self) -> int:
        return self.shape[2] if len(self.shape) == 3 else 1


def compute_sha256(byte_content: bytes) -> str:
    """Computes SHA-256 hex digest for byte stream."""
    return hashlib.sha256(byte_content).hexdigest()


def detect_mode(filename: str, explicit_mode: str | None = None) -> Literal["terrain_image", "elevation_raster"]:
    """Determines analysis mode from user override or file extension."""
    if explicit_mode in ("terrain_image", "elevation_raster"):
        return explicit_mode  # type: ignore

    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_RASTER_EXTENSIONS:
        return "elevation_raster"
    return "terrain_image"


def load_from_path(
    file_path: str | Path,
    mode: Literal["terrain_image", "elevation_raster"] | None = None,
) -> LoadedData:
    """Loads an image or raster from disk path."""
    path = Path(file_path)
    val_res = validate_file_path(path)
    if not val_res.is_valid:
        raise ValueError(f"File validation failed: {'; '.join(val_res.errors)}")

    content = path.read_bytes()
    return load_from_bytes(
        file_bytes=content,
        file_name=path.name,
        mode=mode,
        file_path=str(path.resolve()),
    )


def load_from_bytes(
    file_bytes: bytes,
    file_name: str,
    mode: Literal["terrain_image", "elevation_raster"] | None = None,
    file_path: str | None = None,
) -> LoadedData:
    """Loads an image or raster from raw bytes (e.g. from Streamlit uploader)."""
    if len(file_bytes) == 0:
        raise ValueError("Cannot load empty byte buffer (0 bytes).")

    checksum = compute_sha256(file_bytes)
    resolved_mode = detect_mode(file_name, mode)
    ext = Path(file_name).suffix.lower()

    if ext in SUPPORTED_RASTER_EXTENSIONS:
        # Load via Rasterio in-memory filesystem
        try:
            with rasterio.MemoryFile(file_bytes) as memfile, memfile.open() as dataset:
                count = dataset.count
                crs_str = dataset.crs.to_string() if dataset.crs else None
                transform_tuple = tuple(dataset.transform) if dataset.transform else None
                nodata_val = dataset.nodata

                # Read primary band
                band_1 = dataset.read(1)
                arr = np.array(band_1, copy=True)

                # Meta dict
                meta = {
                    "driver": dataset.driver,
                    "band_count": count,
                    "bounds": tuple(dataset.bounds) if dataset.bounds else None,
                    "dtypes": dataset.dtypes,
                }
        except Exception as e:
            raise ValueError(f"Failed to decode GeoTIFF raster '{file_name}': {e}") from e
    else:
        # Standard PNG/JPEG image loading
        try:
            with Image.open(io.BytesIO(file_bytes)) as pil_img:
                # Convert palette or unusual formats to RGB/L
                if pil_img.mode in ("P", "PA", "1"):
                    pil_img = pil_img.convert("RGB")
                arr = np.array(pil_img)
                meta = {
                    "format": pil_img.format,
                    "color_mode": pil_img.mode,
                }
                crs_str = None
                transform_tuple = None
                nodata_val = None
        except Exception as e:
            raise ValueError(f"Failed to decode image '{file_name}': {e}") from e

    arr_val = validate_raster_array(arr, nodata_value=nodata_val, mode=resolved_mode)
    if not arr_val.is_valid:
        raise ValueError(f"Raster data validation failed: {'; '.join(arr_val.errors)}")

    return LoadedData(
        data=arr,
        mode=resolved_mode,
        file_name=file_name,
        file_path=file_path,
        checksum_sha256=checksum,
        file_size_bytes=len(file_bytes),
        dtype_str=str(arr.dtype),
        shape=arr.shape,
        nodata=nodata_val,
        crs=crs_str,
        transform=transform_tuple,
        metadata=meta,
        validation=arr_val,
    )
