"""Script to generate realistic test/sample data for TerrainLens."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Generate sample_canyon.png (256x256 RGB terrain image)
width, height = 256, 256
x = np.linspace(-3, 3, width)
y = np.linspace(-3, 3, height)
xx, yy = np.meshgrid(x, y)

# Canyon terrain function
z_canyon = (
    120 * np.exp(-((xx**2 + yy**2) / 4.0))
    + 40 * np.sin(2.5 * xx) * np.cos(2.5 * yy)
    - 60 * np.exp(-((xx * 0.8 + 0.2 * yy) ** 2) / 0.15)  # river canyon trench
    + 30 * np.sin(6.0 * xx + 4.0 * yy)
)
# Normalize to [30, 240] uint8
canyon_norm = ((z_canyon - z_canyon.min()) / (z_canyon.max() - z_canyon.min()) * 210 + 30).astype(np.uint8)
# Create colorized RGB terrain (earthy tones: green, brown, ochre, rock)
r = np.clip(canyon_norm * 1.1 + 20, 0, 255).astype(np.uint8)
g = np.clip(canyon_norm * 0.9 + 10, 0, 255).astype(np.uint8)
b = np.clip(canyon_norm * 0.7 + 5, 0, 255).astype(np.uint8)
rgb_canyon = np.stack([r, g, b], axis=-1)

img_canyon = Image.fromarray(rgb_canyon)
img_canyon.save(DATA_DIR / "sample_canyon.png")

# 2. Generate sample_mountains.jpg (256x256 JPEG rugged mountain)
z_mountains = (
    150 * (np.sin(xx * 1.8) * np.cos(yy * 1.5))
    + 60 * np.abs(np.sin(xx * 3.5 - yy * 2.0))
    + 35 * np.cos(xx * 7.0 + yy * 5.0)
    + 10 * np.random.RandomState(42).normal(0, 1, (height, width))
)
mountains_norm = ((z_mountains - z_mountains.min()) / (z_mountains.max() - z_mountains.min()) * 240 + 10).astype(np.uint8)
r_m = np.clip(mountains_norm * 1.05 + 15, 0, 255).astype(np.uint8)
g_m = np.clip(mountains_norm * 1.0 + 15, 0, 255).astype(np.uint8)
b_m = np.clip(mountains_norm * 1.1 + 25, 0, 255).astype(np.uint8)
img_mountains = Image.fromarray(np.stack([r_m, g_m, b_m], axis=-1))
img_mountains.save(DATA_DIR / "sample_mountains.jpg", quality=95)

# 3. Generate sample_reference_mask.png (binary edge mask for precision/recall test on sample_canyon)
import cv2

edges_ref = cv2.Canny(cv2.cvtColor(rgb_canyon, cv2.COLOR_RGB2GRAY), 50, 150)
mask_ref = Image.fromarray(edges_ref)
mask_ref.save(DATA_DIR / "sample_reference_mask.png")

# 4. Generate sample_dem_elevation.tif (Single-band GeoTIFF, 128x128 float32 elevation DEM)
dem_w, dem_h = 128, 128
x_dem = np.linspace(-2, 2, dem_w)
y_dem = np.linspace(-2, 2, dem_h)
xxd, yyd = np.meshgrid(x_dem, y_dem)

# Synthetic alpine mountain elevation (elevation in meters: 1100m to 2850m)
elevation_dem = (
    1800.0
    + 850.0 * np.exp(-(xxd**2 + yyd**2) / 1.5)  # Peak
    + 350.0 * np.sin(2.5 * xxd) * np.cos(1.8 * yyd)  # Ridges
    - 400.0 * np.exp(-((xxd - 0.7)**2 + (yyd + 0.5)**2) / 0.5)  # Glacial cirque
)
elevation_dem = elevation_dem.astype(np.float32)

transform = from_origin(west=-119.5, north=37.8, xsize=0.0005, ysize=0.0005)
crs = rasterio.crs.CRS.from_epsg(4326)

with rasterio.open(
    DATA_DIR / "sample_dem_elevation.tif",
    "w",
    driver="GTiff",
    height=dem_h,
    width=dem_w,
    count=1,
    dtype=rasterio.float32,
    crs=crs,
    transform=transform,
    nodata=-9999.0,
) as dst:
    dst.write(elevation_dem, 1)

# 5. Generate sample_flat_plateau.tif (Single-band GeoTIFF with step plateau elevations)
plateau_dem = 1200.0 + 300.0 * np.floor(np.hypot(xxd, yyd) * 2.0)
plateau_dem = plateau_dem.astype(np.float32)
with rasterio.open(
    DATA_DIR / "sample_flat_plateau.tif",
    "w",
    driver="GTiff",
    height=dem_h,
    width=dem_w,
    count=1,
    dtype=rasterio.float32,
    crs=crs,
    transform=transform,
    nodata=-9999.0,
) as dst:
    dst.write(plateau_dem, 1)

print("Sample files created in data/ successfully.")
