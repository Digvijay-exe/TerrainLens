"""Unit tests for TerrainLens validation module."""

from pathlib import Path

import numpy as np

from terrainlens.validation import (
    validate_file_path,
    validate_raster_array,
)


def test_validate_file_path_nonexistent():
    res = validate_file_path("non_existent_image.png")
    assert not res.is_valid
    assert any("not found" in err.lower() for err in res.errors)


def test_validate_file_path_unsupported_extension(tmp_path: Path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello terrain")
    res = validate_file_path(txt_file)
    assert not res.is_valid
    assert any("unsupported" in err.lower() for err in res.errors)


def test_validate_file_path_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    res = validate_file_path(empty_file)
    assert not res.is_valid
    assert any("empty" in err.lower() for err in res.errors)


def test_validate_valid_sample_files():
    for f in ["sample_canyon.png", "sample_dem_elevation.tif"]:
        p = Path("data") / f
        if p.exists():
            res = validate_file_path(p)
            assert res.is_valid
            assert res.metadata["file_size_bytes"] > 0


def test_validate_raster_array_valid():
    arr = np.ones((64, 64), dtype=np.uint8) * 128
    res = validate_raster_array(arr)
    assert res.is_valid
    assert res.metadata["width"] == 64
    assert res.metadata["height"] == 64


def test_validate_raster_array_too_small():
    arr = np.zeros((8, 8), dtype=np.uint8)
    res = validate_raster_array(arr)
    assert not res.is_valid
    assert any("smaller than minimum" in err.lower() for err in res.errors)


def test_validate_raster_array_nans():
    arr = np.ones((64, 64), dtype=np.float32)
    # 10% NaNs should trigger warning but be valid
    arr[:6, :6] = np.nan
    res = validate_raster_array(arr)
    assert res.is_valid
    assert len(res.warnings) > 0
    assert res.metadata["nan_count"] == 36

    # >50% NaNs should trigger error
    arr[:50, :] = np.nan
    res_bad = validate_raster_array(arr)
    assert not res_bad.is_valid
    assert any("over 50% nan" in err.lower() for err in res_bad.errors)


def test_validate_raster_array_infinite():
    arr = np.ones((64, 64), dtype=np.float32)
    arr[0, 0] = np.inf
    res = validate_raster_array(arr)
    assert not res.is_valid
    assert any("infinite" in err.lower() for err in res.errors)
