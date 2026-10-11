# 🏔️ TerrainLens
### Terrain Understanding & Robustness Analyzer

**TerrainLens** is an open-source, reproducible computer-vision research platform designed to evaluate feature extraction resilience across aerial photographs and digital elevation models (DEM) under controlled environmental and sensor degradations.

Built for geospatial researchers and computer vision engineers, TerrainLens strictly decouples photographic optical boundary detection (Canny edges, Sobel gradients) from true physical geomorphometry (marching squares topographic contours and analytical hillshading). It provides a full experimental suite for stress-testing feature retention against optical blur, sensor noise, illumination attenuation, and spatial downsampling.

---

### 🌟 Key Highlights
- **Multi-Modal Data Ingestion:** Supports PNG/JPEG visual imagery and single-band GeoTIFF rasters with cryptographic SHA-256 integrity audits.
- **Dual-Paradigm Feature Extraction:**
  - *Terrain Images:* Visual edge contrast and gradients via OpenCV (Canny, Sobel).
  - *Elevation Rasters (DEM):* Physical topographic isocontours (skimage marching squares), Horn's 3×3 analytical hillshading, and terrain roughness/slope statistics.
- **Controlled Degradation Engine:** Non-destructive stress-testing against Gaussian blur, zero-mean Gaussian noise, diurnal illumination variance, and pixel-aligned resolution downsampling.
- **Strict Evaluation Metrics:** Distinguishes **Correctness** (Precision, Recall, F1, IoU against ground-truth masks) from **Stability** (Dice retention, SSIM, MSE vs. pristine baselines).
- **Interactive Multi-Page Streamlit Dashboard:** Complete side-by-side visualizers, Plotly sensitivity curves, reproducible JSON manifests, and one-click export bundles (CSV, JSON, PNG, and Markdown reports).
- **100% Offline & CPU-Friendly:** Classical deterministic algorithms powered by OpenCV, Rasterio, SciPy, and scikit-image—no external APIs, cloud services, or GPU dependencies required.
