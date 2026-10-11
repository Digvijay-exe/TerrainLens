"""TerrainLens Page 3 — Robustness Lab.

Executes controlled image degradations, computes stability metrics vs pristine
baseline (and correctness if ground truth is present), and renders interactive
comparison visualizers and sensitivity charts.
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
from terrainlens.degradation import get_default_degradation_suite
from terrainlens.experiment import Experiment
from terrainlens.reporting import create_robustness_plots

st.set_page_config(
    page_title="TerrainLens — Robustness Lab",
    page_icon="🔬",
    layout="wide",
)

init_session_state()
render_sidebar_info()

st.title("🔬 Phase 2: Robustness Lab & Degradation Benchmark")
st.markdown(
    "Stress-test feature extraction stability by applying controlled environmental and sensor "
    "degradations. Measure feature retention, SSIM, and geometric overlap against the pristine baseline."
)

if not st.session_state.get("analysis_complete") or st.session_state.get("current_data") is None:
    st.warning("⚠️ No baseline feature analysis found. Please run baseline extraction in **Analyze** first.")
    if st.button("← Go to Analyze Page", type="primary"):
        st.switch_page("pages/analyze.py")
    st.stop()

current_data = st.session_state.current_data
baseline_img = st.session_state.baseline_input
baseline_feats = st.session_state.baseline_features
ref_mask = st.session_state.get("reference_mask")

st.markdown("### 1. Configure Degradation Suite")

suite_choice = st.radio(
    "Suite Configuration Mode:",
    ["Standard 4-Variant Research Suite", "Customized Degradation Parameters"],
    index=0,
)

if suite_choice == "Standard 4-Variant Research Suite":
    degradation_suite = get_default_degradation_suite()
    st.markdown("Standard benchmark includes:")
    st.markdown("- **1. Gaussian Blur:** Kernel = 5x5, σ = 2.0")
    st.markdown("- **2. Gaussian Noise:** Zero-mean, σ = 25.0 intensity")
    st.markdown("- **3. Illumination Shift:** Brightness offset = -30, Contrast scale = 0.85")
    st.markdown("- **4. Resolution Downsampling:** 50% spatial resolution resampled")
else:
    # Custom configuration
    st.markdown("Customize individual degradation parameters:")
    deg_cols = st.columns(4)

    with deg_cols[0]:
        st.markdown("**Gaussian Blur**")
        blur_enabled = st.checkbox("Include Blur", value=True)
        blur_k = st.slider("Blur Kernel", 3, 11, 5, step=2)
        blur_sigma = st.slider("Blur Sigma (σ)", 0.5, 5.0, 2.0, step=0.5)

    with deg_cols[1]:
        st.markdown("**Gaussian Noise**")
        noise_enabled = st.checkbox("Include Noise", value=True)
        noise_sigma = st.slider("Noise Sigma (σ)", 5.0, 60.0, 25.0, step=5.0)

    with deg_cols[2]:
        st.markdown("**Brightness/Contrast**")
        bright_enabled = st.checkbox("Include Illumination", value=True)
        bright_offset = st.slider("Brightness Offset", -60.0, 60.0, -30.0, step=5.0)
        bright_scale = st.slider("Contrast Scale", 0.5, 1.5, 0.85, step=0.05)

    with deg_cols[3]:
        st.markdown("**Resolution Reduction**")
        res_enabled = st.checkbox("Include Resampling", value=True)
        res_factor = st.slider("Resolution Scale", 0.15, 0.9, 0.5, step=0.05)

    degradation_suite = []
    if blur_enabled:
        degradation_suite.append({
            "id": "custom_blur",
            "type": "gaussian_blur",
            "name": f"Gaussian Blur (k={blur_k}, σ={blur_sigma})",
            "params": {"kernel_size": blur_k, "sigma": blur_sigma},
        })
    if noise_enabled:
        degradation_suite.append({
            "id": "custom_noise",
            "type": "gaussian_noise",
            "name": f"Gaussian Noise (σ={noise_sigma})",
            "params": {"sigma": noise_sigma, "seed": 42},
        })
    if bright_enabled:
        degradation_suite.append({
            "id": "custom_illumination",
            "type": "brightness",
            "name": f"Illumination (scale={bright_scale}, offset={bright_offset})",
            "params": {"offset": bright_offset, "scale": bright_scale},
        })
    if res_enabled:
        degradation_suite.append({
            "id": "custom_resolution",
            "type": "resolution_reduction",
            "name": f"Resolution Downsampled ({res_factor*100:.0f}%)",
            "params": {"scale_factor": res_factor, "interpolation": "bilinear"},
        })

    if not degradation_suite:
        st.error("Please enable at least one degradation type.")
        st.stop()

st.markdown("---")

# Run Experiment Button
exp_name = st.text_input("Experiment Name", value=f"Robustness Benchmark: {current_data.file_name}")

if st.button("🚀 Execute Robustness Benchmark", type="primary", use_container_width=True):
    with st.spinner("Executing degradation suite, extracting features, and computing metrics..."):
        experiment = Experiment(name=exp_name)
        cfg = experiment.run(
            loaded_data=current_data,
            feature_method=st.session_state.feature_method,
            feature_params=st.session_state.feature_params,
            preprocessing_params=st.session_state.preprocessing_params,
            degradation_suite=degradation_suite,
            reference_mask=ref_mask,
        )
        st.session_state.current_experiment = experiment
        st.session_state.robustness_complete = True
        st.success(f"Benchmark completed! Generated Experiment ID: `{experiment.experiment_id}`")

# Display Results
if st.session_state.get("robustness_complete") and st.session_state.get("current_experiment") is not None:
    exp: Experiment = st.session_state.current_experiment

    st.markdown("---")
    st.markdown("### 2. Side-by-Side Visual Inspection")

    # Tabs for each variant
    tab_names = ["Baseline (Pristine)"] + [v["name"] for v in exp.variants]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pristine Baseline Input**")
            st.image(exp.baseline_image, use_container_width=True)
        with c2:
            st.markdown("**Baseline Features**")
            st.image(exp.baseline_features, use_container_width=True)

    for idx, v in enumerate(exp.variants):
        with tabs[idx + 1]:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Degraded Input:** {v['name']}")
                st.image(v["degraded_image"], use_container_width=True)
            with c2:
                st.markdown("**Extracted Features**")
                st.image(v["features"], use_container_width=True)
            with c3:
                st.markdown("**Stability & Metric Audit**")
                stab = v["stability"]
                st.metric("Stability Dice (Retention)", f"{stab.get('stability_dice', 0.0):.4f}")
                st.metric("Structural Similarity (SSIM)", f"{stab.get('ssim', 0.0):.4f}")
                st.metric("Feature Delta %", f"{stab.get('feature_delta_pct', 0.0):+.1f}%")
                if "correctness" in v:
                    corr = v["correctness"]
                    st.metric("F1 Correctness (vs Ground Truth)", f"{corr.get('f1_score', 0.0):.4f}")

    st.markdown("---")
    st.markdown("### 3. Quantitative Robustness Metrics")
    st.caption(
        "Notice: **Stability Metrics** evaluate resistance against degradation relative to baseline. "
        "**Correctness Metrics** evaluate accuracy relative to ground truth reference."
    )

    # Style and format dataframe
    df_display = exp.metrics_table.copy()
    st.dataframe(df_display, use_container_width=True)

    st.markdown("### 4. Degradation Sensitivity Plots")
    fig = create_robustness_plots(exp.metrics_table)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c_rep1, c_rep2 = st.columns([1.5, 3])
    with c_rep1:
        if st.button("📑 View & Export Reports →", type="primary", use_container_width=True):
            st.switch_page("pages/reports.py")
    with c_rep2:
        st.caption("Inspect reproducible JSON configuration, Markdown summary report, and download research bundles.")
