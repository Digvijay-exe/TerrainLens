"""Terrain analysis module for Elevation Rasters (DEM).

Extracts genuine topographic elevation contours at configurable intervals,
computes analytical hillshade, and calculates terrain geomorphometric statistics
(elevation range, slope distributions, terrain roughness).
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from skimage import measure

from terrainlens.preprocessing import calculate_hillshade, handle_missing_elevation


def compute_terrain_statistics(
    elevation: np.ndarray,
    cell_size: float = 1.0,
    nodata: float | None = None,
) -> dict[str, Any]:
    """Calculates comprehensive geomorphic summary statistics from an elevation raster.

    Includes elevation distribution, slope angles (in degrees), and surface roughness.
    """
    clean_elev = handle_missing_elevation(elevation, nodata=nodata)

    min_elev = float(np.min(clean_elev))
    max_elev = float(np.max(clean_elev))
    mean_elev = float(np.mean(clean_elev))
    std_elev = float(np.std(clean_elev))
    elev_range = max_elev - min_elev

    # Slope computation using Horn's 3x3 method
    kernel_dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / (8.0 * cell_size)
    kernel_dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / (8.0 * cell_size)

    dz_dx = cv2.filter2D(clean_elev, -1, kernel_dx)
    dz_dy = cv2.filter2D(clean_elev, -1, kernel_dy)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # Surface roughness (standard deviation of elevation within 3x3 moving window)
    mean_local = cv2.blur(clean_elev, (3, 3))
    sq_diff = (clean_elev - mean_local) ** 2
    roughness = float(np.sqrt(np.mean(sq_diff)))

    return {
        "elevation_min_m": round(min_elev, 2),
        "elevation_max_m": round(max_elev, 2),
        "elevation_mean_m": round(mean_elev, 2),
        "elevation_std_m": round(std_elev, 2),
        "elevation_range_m": round(elev_range, 2),
        "slope_mean_deg": round(float(np.mean(slope_deg)), 2),
        "slope_max_deg": round(float(np.max(slope_deg)), 2),
        "slope_std_deg": round(float(np.std(slope_deg)), 2),
        "surface_roughness": round(roughness, 3),
        "grid_shape": list(clean_elev.shape),
    }


def extract_topographic_contours(
    elevation: np.ndarray,
    interval: float | None = None,
    num_levels: int | None = 10,
    line_thickness: int = 1,
    nodata: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extracts true topographic contour lines from an elevation raster at configurable intervals.

    Args:
        elevation: 2D array of elevations (in meters or elevation units).
        interval: Explicit elevation interval between contour lines (e.g. 50.0m).
                  If None, computed automatically from num_levels.
        num_levels: Number of contour levels to distribute evenly across elevation range.
        line_thickness: Pixel thickness of rendered contour lines.
        nodata: Nodata elevation value.

    Returns:
        Tuple of (binary_contour_map [0 or 255 uint8], metadata_dict).
    """
    start_time = time.perf_counter()
    clean_elev = handle_missing_elevation(elevation, nodata=nodata)

    min_elev = float(np.min(clean_elev))
    max_elev = float(np.max(clean_elev))
    elev_range = max_elev - min_elev

    if elev_range <= 0.001:
        # Flat raster
        blank = np.zeros(clean_elev.shape, dtype=np.uint8)
        return blank, {
            "contour_levels": [],
            "contour_interval": 0.0,
            "contour_count": 0,
            "feature_density_pct": 0.0,
            "runtime_ms": 0.1,
            "note": "Raster has uniform elevation (zero contours extracted).",
        }

    # Determine isoline elevation thresholds
    if interval is not None and interval > 0:
        first_level = np.ceil(min_elev / interval) * interval
        if first_level <= min_elev:
            first_level += interval
        levels = np.arange(first_level, max_elev, interval)
        actual_interval = float(interval)
    else:
        n = max(2, int(num_levels or 10))
        actual_interval = elev_range / (n + 1)
        levels = np.linspace(min_elev + actual_interval, max_elev - actual_interval, n)

    contour_map = np.zeros(clean_elev.shape, dtype=np.uint8)
    all_contours_count = 0
    total_segments_length = 0.0

    # Extract isolines using skimage marching squares
    for level in levels:
        contours = measure.find_contours(clean_elev, level=float(level))
        all_contours_count += len(contours)
        for contour in contours:
            # contour points are (row, col) = (y, x)
            pts = np.fliplr(contour).astype(np.int32)  # converts to (x, y)
            if len(pts) > 1:
                cv2.polylines(
                    contour_map,
                    [pts],
                    isClosed=False,
                    color=255,
                    thickness=int(line_thickness),
                )
                # Approximate segment length
                diffs = np.diff(pts, axis=0)
                total_segments_length += float(np.sum(np.sqrt((diffs**2).sum(axis=1))))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    contour_pixels = int(np.count_nonzero(contour_map))
    total_pixels = int(contour_map.size)
    density_pct = (contour_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    meta = {
        "method": "topographic_contours",
        "method_label": "Topographic Elevation Contours (Physical Isolines)",
        "contour_interval": round(actual_interval, 2),
        "contour_levels": [round(float(lvl), 2) for lvl in levels],
        "num_levels": len(levels),
        "contour_count": all_contours_count,
        "total_line_length_px": round(total_segments_length, 1),
        "feature_pixels": contour_pixels,
        "total_pixels": total_pixels,
        "feature_density_pct": round(density_pct, 4),
        "runtime_ms": round(elapsed_ms, 3),
    }
    return contour_map, meta


def analyze_elevation_raster(
    elevation: np.ndarray,
    contour_interval: float | None = None,
    num_levels: int = 10,
    hillshade_azimuth: float = 315.0,
    hillshade_altitude: float = 45.0,
    nodata: float | None = None,
) -> dict[str, Any]:
    """Runs a complete elevation raster analysis.

    Returns:
        dict with keys:
            - contour_map: 2D uint8 binary line map
            - hillshade: 2D uint8 hillshade image
            - stats: terrain geomorphometry summary
            - contour_meta: metadata of contours
    """
    clean_elev = handle_missing_elevation(elevation, nodata=nodata)
    stats = compute_terrain_statistics(clean_elev, nodata=nodata)
    hillshade = calculate_hillshade(clean_elev, azimuth=hillshade_azimuth, altitude=hillshade_altitude)
    contour_map, contour_meta = extract_topographic_contours(
        clean_elev,
        interval=contour_interval,
        num_levels=num_levels,
        nodata=nodata,
    )

    return {
        "contour_map": contour_map,
        "hillshade": hillshade,
        "stats": stats,
        "contour_meta": contour_meta,
    }
