"""TerrainLens Page 2 — Analyze.

Handles file upload, format validation, analysis mode selection (Terrain Image vs
Elevation Raster), parameter configuration, and side-by-side baseline feature extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for standalone and Streamlit multipage execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from app.common import init_session_state, render_sidebar_info
from terrainlens.data_loader import LoadedData, load_from_bytes, load_from_path
from terrainlens.feature_extraction import extract_features
from terrainlens.preprocessing import (
    normalize_for_visualization,
    preprocess_pipeline,
)
from terrainlens.terrain_analysis import (
    analyze_elevation_raster,
)

st.set_page_config(
    page_title="TerrainLens — Analyze",
    page_icon="🔍",
    layout="wide",
)

init_session_state()
render_sidebar_info()

st.title("🔍 Phase 1: Ingestion & Baseline Feature Extraction")
st.markdown(
    "Upload raw imagery or single-band elevation rasters, configure preprocessing, and run "
    "baseline feature detection prior to degradation stress testing."
)

# Step 1: Upload / Dataset selection
st.markdown("### 1. Ingest Dataset")
upload_tab1, upload_tab2 = st.tabs(["📤 Upload New File", "📁 Select from Sample Datasets"])

with upload_tab1:
    uploaded_file = st.file_uploader(
        "Upload a Terrain Image (PNG, JPEG) or Elevation DEM Raster (GeoTIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="PNG and JPEG are treated as visual terrain photos. GeoTIFF is processed as true physical elevation data.",
    )
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            loaded = load_from_bytes(
                file_bytes=bytes_data,
                file_name=uploaded_file.name,
                mode=None,
            )
            st.session_state.current_data = loaded
            st.session_state.active_mode = loaded.mode
            st.success(f"Successfully loaded and validated `{uploaded_file.name}`!")
        except (ValueError, OSError, RuntimeError) as e:
            st.error(f"Validation error: {e}")

    # Optional reference ground-truth mask upload
    ref_mask_file = st.file_uploader(
        "Optional: Ground Truth Binary Reference Mask (for Precision/Recall/F1 Correctness)",
        type=["png", "jpg", "jpeg"],
        key="ref_mask_uploader",
        help="If supplied, allows measuring true detection correctness. If omitted, robustness will be measured via stability vs baseline.",
    )
    if ref_mask_file is not None:
        try:
            ref_bytes = ref_mask_file.getvalue()
            ref_data = load_from_bytes(ref_bytes, ref_mask_file.name)
            st.session_state.reference_mask = ref_data.data
            st.success(f"Reference mask loaded: `{ref_mask_file.name}`")
        except (ValueError, OSError, RuntimeError) as e:
            st.error(f"Error loading reference mask: {e}")

with upload_tab2:
    data_dir = Path("data")
    sample_options = ["sample_canyon.png", "sample_mountains.jpg", "sample_dem_elevation.tif", "sample_flat_plateau.tif"]
    selected_sample = st.selectbox("Choose pre-packaged dataset:", sample_options)
    if st.button("Load Selected Sample", type="secondary"):
        p = data_dir / selected_sample
        if p.exists():
            loaded = load_from_path(p)
            st.session_state.current_data = loaded
            st.session_state.active_mode = loaded.mode

            ref_path = data_dir / "sample_reference_mask.png"
            if selected_sample == "sample_canyon.png" and ref_path.exists():
                st.session_state.reference_mask = load_from_path(ref_path).data
            else:
                st.session_state.reference_mask = None

            st.success(f"Loaded sample dataset '{selected_sample}'!")
            st.rerun()

current_data: LoadedData | None = st.session_state.get("current_data")

if current_data is None:
    st.info("Please upload an image/raster or select a sample dataset above to continue.")
    st.stop()

st.markdown("---")

# Step 2: Inspection and Validation Metadata
st.markdown("### 2. Dataset Metadata & Validation Audit")
mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
mcol1.metric("Width × Height", f"{current_data.width} × {current_data.height} px")
mcol2.metric("Data Type", current_data.dtype_str)
mcol3.metric("File Size", f"{current_data.file_size_bytes / 1024:.1f} KB")
mcol4.metric("Channels", str(current_data.channels))
mcol5.metric("Detected Mode", current_data.mode.replace("_", " ").title())

st.caption(f"**Cryptographic Checksum (SHA-256):** `{current_data.checksum_sha256}`")

# Mode selector override
mode_choice = st.radio(
    "Analysis Paradigm Mode:",
    options=["Terrain Image (Visual Optical Contrast)", "Elevation Raster (Physical Elevation Contours)"],
    index=0 if current_data.mode == "terrain_image" else 1,
    help=(
        "Terrain Image uses Canny or Sobel on pixel brightness. "
        "Elevation Raster extracts physical topographic isocontours and analytical hillshading."
    ),
)
effective_mode = "terrain_image" if "Terrain Image" in mode_choice else "elevation_raster"
st.session_state.active_mode = effective_mode

if effective_mode == "terrain_image":
    st.info(
        "ℹ️ **Scientific Distinction Note:** Terrain Image mode detects photographic edge boundaries. "
        "These represent optical reflectance, vegetation, and shadowing boundaries — NOT true elevation contours."
    )
else:
    st.info(
        "ℹ️ **Elevation Mode Note:** Computing authentic topographic elevation contour lines at physical height intervals."
    )

st.markdown("---")

# Step 3: Parameter Configuration
st.markdown("### 3. Pipeline Parameters")
pcol1, pcol2 = st.columns(2)

with pcol1:
    st.markdown("#### Preprocessing Parameters")
    apply_smooth = st.checkbox("Gaussian Smoothing Pre-Filter", value=True)
    kernel_size = st.slider("Smoothing Kernel Size", min_value=3, max_value=15, value=5, step=2) if apply_smooth else 0
    sigma = st.slider("Gaussian Sigma (σ)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) if apply_smooth else 0.0

with pcol2:
    st.markdown("#### Feature Extraction Parameters")
    feature_params = {}
    if effective_mode == "terrain_image":
        method = st.selectbox("Edge Detection Method", ["Canny Edge Detector", "Sobel Gradient Magnitude"])
        if method == "Canny Edge Detector":
            selected_method = "canny"
            low_th = st.slider("Canny Low Threshold", 10, 200, 50, step=5)
            high_th = st.slider("Canny High Threshold", 50, 300, 150, step=5)
            aperture = st.selectbox("Aperture Size", [3, 5, 7], index=0)
            feature_params = {
                "low_threshold": low_th,
                "high_threshold": high_th,
                "aperture_size": aperture,
            }
        else:
            selected_method = "sobel"
            ksize = st.selectbox("Sobel Kernel Size", [1, 3, 5, 7], index=1)
            use_thresh = st.checkbox("Apply Binary Threshold", value=True)
            thresh_val = st.slider("Gradient Magnitude Threshold", 10, 200, 50, step=5) if use_thresh else None
            feature_params = {
                "ksize": ksize,
                "threshold": thresh_val,
            }
    else:
        selected_method = "topographic_contours"
        contour_mode = st.radio("Contour Distribution:", ["Evenly Spaced Levels", "Explicit Elevation Interval (m)"])
        if contour_mode == "Evenly Spaced Levels":
            num_levels = st.slider("Number of Contour Levels", 3, 30, 10)
            feature_params = {"num_levels": num_levels, "contour_interval": None}
        else:
            interval = st.number_input("Elevation Interval (meters)", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
            feature_params = {"contour_interval": interval, "num_levels": None}

        hillshade_az = st.slider("Hillshade Light Azimuth (°)", 0.0, 360.0, 315.0, step=15.0)
        hillshade_alt = st.slider("Hillshade Solar Altitude (°)", 10.0, 90.0, 45.0, step=5.0)
        feature_params["hillshade_azimuth"] = hillshade_az
        feature_params["hillshade_altitude"] = hillshade_alt

st.markdown("---")

# Run Baseline Extraction
if st.button("▶️ Run Baseline Feature Extraction", type="primary", use_container_width=True):
    with st.spinner("Executing preprocessing and baseline feature extraction..."):
        # Preprocessing
        prep_img, prep_rec = preprocess_pipeline(
            data=current_data.data,
            mode=effective_mode,
            nodata=current_data.nodata,
            smooth=apply_smooth,
            kernel_size=kernel_size,
            sigma=sigma,
        )

        if effective_mode == "terrain_image":
            feats, f_meta = extract_features(
                image=prep_img,
                method=selected_method,  # type: ignore
                **feature_params,
            )
            base_vis = prep_img
            hillshade_vis = None
            stats_dict = None
        else:
            analysis = analyze_elevation_raster(
                elevation=prep_img,
                contour_interval=feature_params.get("contour_interval"),
                num_levels=feature_params.get("num_levels", 10),
                hillshade_azimuth=feature_params.get("hillshade_azimuth", 315.0),
                hillshade_altitude=feature_params.get("hillshade_altitude", 45.0),
                nodata=current_data.nodata,
            )
            feats = analysis["contour_map"]
            f_meta = analysis["contour_meta"]
            hillshade_vis = analysis["hillshade"]
            stats_dict = analysis["stats"]
            base_vis = normalize_for_visualization(prep_img, nodata=current_data.nodata)

        # Store baseline in session state
        st.session_state.baseline_input = base_vis
        st.session_state.baseline_features = feats
        st.session_state.baseline_meta = f_meta
        st.session_state.baseline_hillshade = hillshade_vis
        st.session_state.elevation_stats = stats_dict
        st.session_state.feature_params = feature_params
        st.session_state.feature_method = selected_method
        st.session_state.preprocessing_params = {
            "smooth": apply_smooth,
            "kernel_size": kernel_size,
            "sigma": sigma,
        }
        st.session_state.analysis_complete = True
        st.success("Baseline extraction completed successfully!")

# Display Results Side-by-Side
if st.session_state.get("analysis_complete"):
    st.markdown("### 4. Baseline Results & Visual Feature Maps")

    b_input = st.session_state.baseline_input
    b_feats = st.session_state.baseline_features
    b_meta = st.session_state.baseline_meta

    if effective_mode == "elevation_raster" and st.session_state.get("baseline_hillshade") is not None:
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.markdown("**Elevation DEM (Normalized)**")
            st.image(b_input, use_container_width=True, caption=f"DEM Dimensions: {b_input.shape}")
        with rcol2:
            st.markdown("**Analytical Hillshade (Relief)**")
            st.image(st.session_state.baseline_hillshade, use_container_width=True, caption="Illuminated Topography")
        with rcol3:
            st.markdown("**Topographic Contours (Isolines)**")
            st.image(b_feats, use_container_width=True, caption=f"Contours Detected ({b_meta.get('feature_density_pct')}%)")

        if st.session_state.get("elevation_stats"):
            st.markdown("#### Geomorphometric Relief Summary")
            st.json(st.session_state.elevation_stats)
    else:
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.markdown("**Preprocessed Input Image**")
            st.image(b_input, use_container_width=True, caption=f"Input Dimensions: {b_input.shape}")
        with rcol2:
            feat_title = (
                "Canny Visual Edge Map"
                if st.session_state.feature_method == "canny"
                else "Sobel Gradient Map"
            )
            st.markdown(f"**{feat_title}**")
            st.image(b_feats, use_container_width=True, caption=f"Extracted Pixels: {b_meta.get('feature_pixels', 0):,} ({b_meta.get('feature_density_pct', 0.0)}%)")

    # Diagnostic Metrics Bar
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Feature Pixels", f"{b_meta.get('feature_pixels', 0):,}")
    stat_col2.metric("Feature Density", f"{b_meta.get('feature_density_pct', 0.0):.2f}%")
    stat_col3.metric("Processing Latency", f"{b_meta.get('runtime_ms', 0.0):.2f} ms")
    has_ref = st.session_state.get("reference_mask") is not None
    stat_col4.metric("Reference Mask", "Active" if has_ref else "None (Stability Mode)")

    st.markdown("---")
    c_next1, c_next2 = st.columns([1.5, 3])
    with c_next1:
        if st.button("🔬 Proceed to Robustness Lab →", type="primary", use_container_width=True):
            st.switch_page("pages/robustness_lab.py")
    with c_next2:
        st.caption("Apply controlled noise, blur, and resolution degradations to test algorithm stability.")
