# TerrainLens Scientific Methodology

## 1. Introduction & Research Scope
TerrainLens is designed for reproducible evaluation of feature extraction algorithms on aerial, satellite, and digital elevation datasets under synthetic environmental and sensor degradations.

### 1.1 Critical Scientific Distinction: Visual Contrast vs Elevation Contours
A fundamental tenet of terrain understanding is distinguishing between photographic imagery and calibrated elevation models:
- **Terrain Image (Photographic PNG/JPEG):** Features are derived from 2D pixel intensity gradients (reflectance, shadows, surface albedo). Algorithms like **Canny Edge Detection** and **Sobel Gradient Magnitude** measure high-frequency visual contrast. These are **visual boundaries**, never physical contours.
- **Elevation Raster (GeoTIFF DEM):** Values correspond to physical metric heights above a vertical datum (e.g., meters above sea level). Features are extracted via marching squares isocontouring to produce **topographic elevation contours (isolines)**, alongside Horn's 3x3 geomorphometric slope and analytical hillshading.

---

## 2. Ingestion & Integrity Verification
Every ingested dataset is subject to cryptographic validation:
- **Checksum Calculation:** SHA-256 hash is computed immediately upon byte ingestion to anchor provenance.
- **Dimensional & Dtype Constraints:** Verifies dimensions ($16 \le \text{dim} \le 10,000$), finite numerical values, and validates against abnormal NaN proportions ($>50\%$).
- **NoData & Missing Value Imputation:** Nodata values in elevation models are filled using Euclidean Distance Transform (nearest-neighbor) interpolation to preserve structural morphology.

---

## 3. Preprocessing Transformations
1. **Grayscale Conversion:** Standard ITU-R BT.601 weighted luma conversion ($Y = 0.299R + 0.587G + 0.114B$) for optical images.
2. **Gaussian Smoothing:** Continuous isotropic 2D Gaussian filter:
   $$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$
3. **Dynamic Range Normalization:** Robust 1st-to-99th percentile contrast stretching for 8-bit visualization while preserving floating-point data for topographic calculations.
4. **Analytical Hillshade:** Cartographic illumination based on solar azimuth ($\theta_{az}$) and altitude ($\theta_{alt}$):
   $$\text{Shade} = 255 \times \max\left(0, \cos(\zeta)\cos(S) + \sin(\zeta)\sin(S)\cos(\theta_{az} - A)\right)$$
   where $\zeta = 90^\circ - \theta_{alt}$, $S$ is surface slope, and $A$ is aspect.

---

## 4. Controlled Degradation Operators
To simulate operational degradation without modifying original inputs:
1. **Gaussian Optical Blur:** Atmospheric scattering and lens defocus ($k \in [3, 11]$, $\sigma \in [0.5, 5.0]$).
2. **Gaussian Sensor Noise:** High-frequency additive electronic/thermal sensor noise with fixed pseudo-random seed:
   $$I_{\text{degraded}}(x, y) = \text{clip}\left(I(x, y) + \mathcal{N}(0, \sigma^2), 0, 255\right)$$
3. **Illumination & Contrast Variance:** Diurnal solar attenuation:
   $$I_{\text{degraded}}(x, y) = \text{clip}\left(I(x, y) \cdot \alpha + \beta, 0, 255\right)$$
4. **Spatial Resolution Downsampling:** Sub-sampling sensor simulation. Downsamples by factor $s \in (0, 1)$ followed by pixel-aligned nearest-neighbor restoration to maintain direct spatial registration with the pristine baseline.

---

## 5. Dual-Pillar Evaluation Framework

TerrainLens strictly separates **Correctness** from **Stability**:

```
                       ┌───────────────────────────────────────────────┐
                       │               Evaluation Mode                 │
                       └──────────────────────┬────────────────────────┘
                                              │
                    ┌─────────────────────────┴────────────────────────┐
                    ▼                                                  ▼
      ┌───────────────────────────┐                      ┌───────────────────────────┐
      │   Correctness Analysis    │                      │    Stability Analysis     │
      │  (vs Ground Truth Mask)   │                      │   (vs Pristine Baseline)  │
      ├───────────────────────────┤                      ├───────────────────────────┤
      │ • Precision, Recall, F1   │                      │ • Stability Dice (overlap)│
      │ • Jaccard Index (IoU)     │                      │ • SSIM & Mean Squared Err │
      │ • Pixel Classification Acc│                      │ • Feature Retention %     │
      └───────────────────────────┘                      └───────────────────────────┘
```

### 5.1 Correctness Metrics (When Reference Mask Exists)
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall:** $\frac{TP}{TP + FN}$
- **F1 Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **IoU (Jaccard):** $\frac{TP}{TP + FP + FN}$

### 5.2 Stability Metrics (Relative to Pristine Baseline)
- **Stability Dice:**
  $$\text{Dice}_{\text{stab}} = \frac{2 \cdot |F_{\text{variant}} \cap F_{\text{baseline}}|}{|F_{\text{variant}}| + |F_{\text{baseline}}|}$$
- **Structural Similarity (SSIM):** Evaluates luminance, contrast, and structural degradation across sliding windows.
- **Feature Retention Rate:**
  $$\text{Retention \%} = \frac{|F_{\text{variant}}|}{|F_{\text{baseline}}|} \times 100\%$$
- **Execution Latency:** Wall-clock CPU runtime in milliseconds measured using high-resolution monotonic clocks (`time.perf_counter()`).
