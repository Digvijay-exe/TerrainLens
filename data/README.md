# Sample Datasets for TerrainLens

This directory provides representative lightweight terrain datasets for reproducible research and automated testing.

## Files

| File | Type | Resolution | Data Mode | Description |
|------|------|------------|-----------|-------------|
| `sample_canyon.png` | PNG | 256 x 256 | Terrain Image | Colorized visual synthetic canyon with riverbed trench, sharp cliffs, and undulating terraces. Ideal for visual edge detection. |
| `sample_mountains.jpg` | JPEG | 256 x 256 | Terrain Image | Rugged alpine ridges with rocky textures and compressed photographic shading. |
| `sample_reference_mask.png` | PNG | 256 x 256 | Ground Truth Mask | Binary edge ground-truth reference for `sample_canyon.png` used to evaluate correctness (Precision, Recall, F1, IoU). |
| `sample_dem_elevation.tif` | GeoTIFF | 128 x 128 | Elevation Raster | Single-band float32 Digital Elevation Model (DEM) spanning 1100m–2850m elevation with EPSG:4326 CRS and affine transform. |
| `sample_flat_plateau.tif` | GeoTIFF | 128 x 128 | Elevation Raster | Step-elevation plateau raster designed for testing discrete topographic contour intervals. |

## Important Scientific Distinction
- **Terrain Image mode**: Analyzes visual surface contrast and high-frequency photographic gradients (Canny edges, Sobel gradient). These are **visual edge boundaries**, NOT true elevation contours.
- **Elevation Raster mode**: Analyzes calibrated topographic heights (meters/feet) to extract true isoline contours and physical surface hillshading/slopes.
