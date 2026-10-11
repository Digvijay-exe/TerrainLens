# TerrainLens — Terrain Understanding & Robustness Analyzer

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)
[![Linter: ruff](https://img.shields.io/badge/linter-ruff-black.svg)](https://github.com/astral-sh/ruff)

**TerrainLens** is a reproducible computer-vision research platform designed to evaluate feature extraction resilience across aerial photographs and digital elevation models (DEM) under controlled environmental and sensor degradations.

---

## 🎯 Core Research Capabilities

1. **Multi-Modal Terrain Ingestion:** Ingests photographic terrain imagery (PNG, JPEG) and calibrated single-band elevation rasters (GeoTIFF) with automated SHA-256 integrity verification.
2. **Dual-Paradigm Feature Extraction:**
   - **Terrain Image Mode:** Canny Edge Detection and Sobel Gradient Magnitude. (Identifies optical reflectance boundaries; strictly distinguished from elevation contours).
   - **Elevation Raster Mode:** Authentic topographic isocontours, analytical cartographic hillshading (Horn's 3x3 method), and geomorphometric slope/roughness distributions.
3. **Controlled Sensor Degradations:** Gaussian optical blur, additive sensor noise, illumination shifts, and spatial resolution downsampling (preserving pixel alignment).
4. **Separated Metric Evaluation:**
   - **Ground-Truth Correctness:** Precision, Recall, F1-Score, and IoU (when an independent reference mask is supplied).
   - **Baseline Stability:** Dice Retention, SSIM, MSE/RMSE, and Feature Delta % relative to the uncorrupted baseline.
5. **Interactive Multipage Streamlit Dashboard:** Complete side-by-side visualizers, sensitivity plots, JSON audit manifests, and exportable publication bundles (CSV, JSON, PNG, Markdown).

---

## 🏗️ Repository Architecture

```text
TerrainLens/
├── app/
│   ├── dashboard.py           # Page 1: Overview, sample loader & experiment catalog
│   ├── common.py              # Shared session state, styling, and array helpers
│   └── pages/
│       ├── analyze.py         # Page 2: Ingestion, validation & baseline extraction
│       ├── robustness_lab.py  # Page 3: Degradation suites & sensitivity analysis
│       └── reports.py         # Page 4: Markdown reports, metrics tables & exports
├── terrainlens/
│   ├── __init__.py            # Library package root
│   ├── data_loader.py         # Ingestion (PNG/JPEG/GeoTIFF) & SHA-256 checksums
│   ├── validation.py          # Array dimensions, dtypes, NaN bounds & file size
│   ├── preprocessing.py       # Grayscale, Gaussian smoothing, hillshade & nodata fill
│   ├── feature_extraction.py  # Canny edges & Sobel gradient magnitude
│   ├── terrain_analysis.py    # Topographic elevation contours & geomorphometry
│   ├── degradation.py         # Blur, noise, illumination, resolution reduction
│   ├── evaluation.py          # Correctness (Precision/Recall/F1) vs Stability (Dice/SSIM)
│   ├── experiment.py          # Orchestration, library versions, metadata & persistence
│   └── reporting.py           # Markdown research report & Plotly charts
├── tests/
│   ├── test_validation.py     # File, array, dtype, and NaN tests
│   ├── test_features.py       # Canny, Sobel, and contour extraction tests
│   ├── test_degradation.py    # Non-destructive degradation tests
│   └── test_evaluation.py    # Correctness and stability metric validation tests
├── data/
│   ├── sample_canyon.png      # 256x256 visual synthetic canyon
│   ├── sample_mountains.jpg   # 256x256 alpine mountain texture
│   ├── sample_reference_mask.png # Binary ground-truth edge mask
│   ├── sample_dem_elevation.tif  # 128x128 GeoTIFF DEM (1100m–2850m)
│   ├── sample_flat_plateau.tif   # 128x128 GeoTIFF step-plateau DEM
│   └── README.md              # Dataset descriptions
├── results/                   # Local experiment folders (JSON, CSV, PNGs)
├── configs/
│   └── example_experiment.json # Reproducible benchmark configuration
├── docs/
│   ├── methodology.md         # Mathematical definitions and paradigms
│   └── limitations.md         # Scientific scope and constraints
├── requirements.txt           # Deterministic Python dependencies
├── README.md                  # System overview and quickstart guide
└── LICENSE                    # Apache 2.0 Open Source License
```

---

## ⚡ Quickstart

### 1. Installation
Ensure Python 3.11+ is installed. Clone and install dependencies:

```bash
git clone https://github.com/example/TerrainLens.git
cd TerrainLens
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app/dashboard.py
```
Open your browser at `http://localhost:8501` (or port configured).

---

## 🧪 Step-by-Step Review Workflow

1. **Page 1 (Overview):**
   - Click **Load sample_canyon.png** or **Load sample_dem_elevation.tif** from the sample dataset cards.
   - Alternatively, inspect past runs in the **Past Experiments Catalog**.
2. **Page 2 (Analyze):**
   - View validated raster dimensions, dtype, file size, and SHA-256 checksum.
   - Select analysis mode: **Terrain Image** or **Elevation Raster**.
   - Configure feature extraction parameters (Canny thresholds, Sobel kernel, or contour levels/intervals).
   - Click **Run Baseline Feature Extraction** to view side-by-side visualizer and latency metrics.
   - Click **Proceed to Robustness Lab →**.
3. **Page 3 (Robustness Lab):**
   - Choose the **Standard 4-Variant Research Suite** (Gaussian Blur, Gaussian Noise, Illumination Shift, 50% Resolution Reduction) or customize parameters.
   - Click **Execute Robustness Benchmark**.
   - Inspect tabbed side-by-side comparisons, metric tables, and interactive Plotly sensitivity charts.
   - Click **View & Export Reports →**.
4. **Page 4 (Reports):**
   - Review executive scientific Markdown report.
   - Inspect tabular metrics (with clear separation of Correctness vs Stability).
   - Download research bundle: `metrics.csv`, `config.json`, `report.md`, and individual processed PNG image maps.

---

## 🔬 Scientific Methodology & Distinctions

| Evaluation Axis | Criterion | Reference Target | Primary Metrics |
|---|---|---|---|
| **Correctness** | Ground-Truth Fidelity | Independent binary mask | Precision, Recall, F1-Score, IoU |
| **Stability** | Operational Resilience | Pristine uncorrupted baseline | Stability Dice, SSIM, MSE, Feature Retention % |

> **Critical Notice:** Canny and Sobel extract **optical brightness gradients** on photographic images. They must never be conflated with topographic contours, which represent **physical isolines** extracted from calibrated elevation models.

---

## 🛠️ Testing & Code Quality

Run the complete automated unit test suite:
```bash
pytest tests/ -v
```

Run static analysis and linting:
```bash
ruff check .
```

---

