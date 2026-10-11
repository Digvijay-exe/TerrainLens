"""Common utilities, session state management, and UI styling for TerrainLens dashboard."""

import io

import numpy as np
import streamlit as st
from PIL import Image


def init_session_state() -> None:
    """Initializes global session state keys."""
    if "current_data" not in st.session_state:
        st.session_state.current_data = None
    if "reference_mask" not in st.session_state:
        st.session_state.reference_mask = None
    if "current_experiment" not in st.session_state:
        st.session_state.current_experiment = None
    if "active_mode" not in st.session_state:
        st.session_state.active_mode = "terrain_image"
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "robustness_complete" not in st.session_state:
        st.session_state.robustness_complete = False


def render_sidebar_info() -> None:
    """Renders persistent sidebar information and environment status."""
    with st.sidebar:
        st.markdown("### 🏔️ TerrainLens")
        st.caption("Terrain Understanding & Robustness Analyzer")
        st.markdown("---")

        if st.session_state.get("current_data"):
            data = st.session_state.current_data
            st.success(f"**Loaded:** `{data.file_name}`")
            st.markdown(f"- **Mode:** `{data.mode}`")
            st.markdown(f"- **Size:** {data.width} × {data.height} px")
            st.markdown(f"- **Type:** `{data.dtype_str}`")
            if data.checksum_sha256:
                st.caption(f"SHA-256: `{data.checksum_sha256[:12]}...`")
        else:
            st.info("No dataset currently active. Upload on the **Analyze** page or load sample data.")

        st.markdown("---")
        st.caption("Offline CPU Computer Vision Architecture")
        st.caption("OpenCV • Rasterio • scikit-image • SciPy")


def array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Converts a numpy array to PNG bytes for download."""
    buf = io.BytesIO()
    if arr.ndim == 2:
        if arr.dtype == bool:
            arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.fromarray(arr.astype(np.uint8))
    img.save(buf, format="PNG")
    return buf.getvalue()
