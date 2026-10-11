"""TerrainLens Main Entry Point: Page 1 — Overview.

Displays system architecture, available sample datasets, past experiment logs,
and triggers the 'New Experiment' workflow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path for standalone and Streamlit multipage execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from app.common import init_session_state, render_sidebar_info
from terrainlens.data_loader import load_from_path
from terrainlens.experiment import list_saved_experiments

st.set_page_config(
    page_title="TerrainLens — Overview",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_sidebar_info()

# Header & Overview
st.title("🏔️ TerrainLens")
st.subheader("Terrain Understanding & Robustness Analyzer")
st.markdown(
    "A reproducible computer-vision research platform designed to extract visual features "
    "and physical elevation contours, apply controlled sensor and atmospheric degradations, "
    "and measure algorithm stability."
)

# New Experiment CTA
col_action1, col_action2, col_action3 = st.columns([1.5, 2, 2.5])
with col_action1:
    if st.button("🚀 Start New Experiment", type="primary", use_container_width=True):
        st.session_state.current_experiment = None
        st.session_state.analysis_complete = False
        st.session_state.robustness_complete = False
        st.switch_page("pages/analyze.py")

with col_action2:
    st.caption("👈 Use sidebar or the button above to navigate through the 4-phase research workflow.")

st.markdown("---")

# Quick Sample Data Loader
st.markdown("### 📂 Sample Datasets")
st.markdown(
    "Pre-packaged lightweight datasets ready for immediate evaluation without requiring external uploads."
)

data_dir = Path("data")
sample_files = [
    ("sample_canyon.png", "Terrain Image", "Visual canyon with cliffs, riverbed trench, and ridges"),
    ("sample_mountains.jpg", "Terrain Image", "Rugged alpine mountain texture with high-frequency edges"),
    ("sample_dem_elevation.tif", "Elevation Raster", "Single-band DEM (1100m–2850m) with true topographic relief"),
    ("sample_flat_plateau.tif", "Elevation Raster", "Step-plateau elevation raster for discrete contour intervals"),
]

cols = st.columns(len(sample_files))
for idx, (filename, mode, desc) in enumerate(sample_files):
    fpath = data_dir / filename
    with cols[idx]:
        st.markdown(f"**{filename}**")
        st.caption(f"**Mode:** {mode}")
        st.caption(desc)
        if fpath.exists():
            if st.button(f"Load {filename}", key=f"load_sample_{idx}", use_container_width=True):
                try:
                    loaded = load_from_path(fpath)
                    st.session_state.current_data = loaded
                    st.session_state.active_mode = loaded.mode
                    st.session_state.analysis_complete = False
                    st.session_state.robustness_complete = False

                    # Check if matching reference mask exists
                    ref_mask_path = data_dir / "sample_reference_mask.png"
                    if filename == "sample_canyon.png" and ref_mask_path.exists():
                        ref_loaded = load_from_path(ref_mask_path)
                        st.session_state.reference_mask = ref_loaded.data
                    else:
                        st.session_state.reference_mask = None

                    st.success(f"Loaded '{filename}'! Redirecting to Analyze...")
                    st.switch_page("pages/analyze.py")
                except (ValueError, OSError, RuntimeError) as e:
                    st.error(f"Failed loading: {e}")
        else:
            st.warning("File missing")

st.markdown("---")

# Past Experiments Catalog
st.markdown("### 🧪 Past Experiments Catalog")
saved_exps = list_saved_experiments()

if not saved_exps:
    st.info("No past experiments recorded yet in `results/`. Click **Start New Experiment** to run your first benchmark!")
else:
    st.markdown(f"Found **{len(saved_exps)}** saved experiment runs in local repository.")
    exp_col1, exp_col2 = st.columns([2, 1])

    with exp_col1:
        for exp in saved_exps:
            with st.expander(f"🔬 {exp['name']} — ID: {exp['id']}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Date:** `{exp['created_at']}`")
                    st.write(f"**Input File:** `{exp['file_name']}`")
                with col_b:
                    st.write(f"**Mode:** `{exp['mode']}`")
                    st.write(f"**Method:** `{exp['method']}`")

                p = Path(exp["path"])
                cfg_path = p / "config.json"
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                    st.json(cfg_data.get("environment", {}), expanded=False)

                if st.button("📊 View Reports for this Experiment", key=f"view_{exp['id']}"):
                    st.session_state.selected_experiment_path = exp["path"]
                    st.switch_page("pages/reports.py")

st.markdown("---")
st.markdown("### 📋 Core Scientific Workflow")
st.markdown(
    """
1. **Analyze (Page 2):** Ingest PNG/JPEG or GeoTIFF. Validate dimensions, dtype, and integrity. Extract baseline features:
   - *Terrain Image*: Canny edge detector or Sobel gradient magnitude.
   - *Elevation Raster*: Geomorphometric relief, analytical hillshade, and true topographic contours.
2. **Robustness Lab (Page 3):** Apply controlled degradations (Gaussian blur, Gaussian noise, illumination changes, resolution downsampling). Compute degradation sensitivity and stability metrics vs baseline.
3. **Reports & Exports (Page 4):** Inspect quantitative stability/correctness tables, sensitivity plots, and download CSV, JSON, PNG, and Markdown reports.
"""
)
