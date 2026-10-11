"""TerrainLens Page 4 — Reports & Publication Exports.

Provides inspection of reproducible JSON configurations, tabular CSV metrics,
interactive sensitivity plots, and download mechanisms for CSV, JSON, PNG, and
Markdown scientific reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path for standalone and Streamlit multipage execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from app.common import init_session_state, render_sidebar_info
from terrainlens.experiment import (
    Experiment,
    list_saved_experiments,
    load_experiment_from_dir,
)
from terrainlens.reporting import create_robustness_plots, generate_markdown_report

st.set_page_config(
    page_title="TerrainLens — Reports",
    page_icon="📊",
    layout="wide",
)

init_session_state()
render_sidebar_info()

st.title("📊 Phase 3: Research Reports & Artifact Exports")
st.markdown(
    "Review verifiable experimental audits, reproducible JSON configuration manifests, "
    "and export publication-ready datasets and reports."
)

# Determine active experiment
active_exp: Experiment | None = st.session_state.get("current_experiment")
config_data: dict | None = None
metrics_df: pd.DataFrame = pd.DataFrame()
exp_dir: Path | None = None

if active_exp is not None:
    config_data = active_exp.config
    metrics_df = active_exp.metrics_table
    exp_dir = active_exp.results_dir
elif st.session_state.get("selected_experiment_path"):
    loaded_dict = load_experiment_from_dir(st.session_state.selected_experiment_path)
    config_data = loaded_dict["config"]
    metrics_df = loaded_dict["metrics_df"]
    exp_dir = loaded_dict["dir"]
else:
    # Try loading most recent saved experiment
    saved_list = list_saved_experiments()
    if saved_list:
        st.info("Loading most recent completed experiment from repository...")
        most_recent = saved_list[0]
        loaded_dict = load_experiment_from_dir(most_recent["path"])
        config_data = loaded_dict["config"]
        metrics_df = loaded_dict["metrics_df"]
        exp_dir = loaded_dict["dir"]
    else:
        st.warning("No experiment records found. Please run a benchmark in **Analyze** and **Robustness Lab** first.")
        if st.button("← Go to Analyze"):
            st.switch_page("pages/analyze.py")
        st.stop()

# Experiment Selector
saved_list = list_saved_experiments()
if len(saved_list) > 1:
    exp_options = {f"{e['name']} ({e['id']})": e["path"] for e in saved_list}
    chosen_label = st.selectbox(
        "Select Experiment to Inspect:",
        list(exp_options.keys()),
        index=0,
    )
    chosen_path = exp_options[chosen_label]
    if exp_dir is None or str(exp_dir) != chosen_path:
        loaded_dict = load_experiment_from_dir(chosen_path)
        config_data = loaded_dict["config"]
        metrics_df = loaded_dict["metrics_df"]
        exp_dir = loaded_dict["dir"]

st.markdown("---")

# Header & Audit Notice
hcol1, hcol2, hcol3 = st.columns([2, 1, 1])
hcol1.markdown(f"### {config_data.get('name', 'TerrainLens Experiment')}")
hcol2.metric("Experiment ID", config_data.get("experiment_id", "N/A")[-12:])
hcol3.metric("Analysis Mode", config_data.get("dataset", {}).get("mode", "N/A"))

st.warning(
    "⚖️ **Scientific Evaluation Labeling Notice:** "
    "All exports and metrics explicitly distinguish **Correctness** (tested against independent Ground Truth) "
    "from **Stability** (operational invariance against uncorrupted baseline under degradation)."
)

# Tabs: Report / Metrics / Sensitivity Charts / JSON Manifest / Downloads
tab_report, tab_table, tab_charts, tab_config, tab_export = st.tabs([
    "📑 Markdown Report",
    "🔢 Metrics Table",
    "📈 Sensitivity Charts",
    "⚙️ Configuration JSON",
    "💾 Export Center",
])

# Generate Markdown Report
md_report_content = generate_markdown_report(config_data, metrics_df)

with tab_report:
    st.markdown("#### Scientific Research Summary Report")
    st.markdown(md_report_content)

with tab_table:
    st.markdown("#### Tabular Metric Audit")
    st.caption("Includes runtime latency, stability metrics vs baseline, and correctness (if reference mask used).")
    st.dataframe(metrics_df, use_container_width=True)

    csv_data = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Metrics CSV",
        data=csv_data,
        file_name=f"{config_data.get('experiment_id')}_metrics.csv",
        mime="text/csv",
        type="primary",
    )

with tab_charts:
    st.markdown("#### Degradation Sensitivity Curves")
    chart_fig = create_robustness_plots(metrics_df)
    st.plotly_chart(chart_fig, use_container_width=True)

with tab_config:
    st.markdown("#### Complete Reproducibility JSON Manifest")
    st.caption("Contains library versions, input file SHA-256 checksum, preprocessing flags, and exact degradation parameters.")
    st.json(config_data, expanded=True)

    json_str = json.dumps(config_data, indent=2).encode("utf-8")
    st.download_button(
        "📥 Download Configuration JSON",
        data=json_str,
        file_name=f"{config_data.get('experiment_id')}_config.json",
        mime="application/json",
    )

with tab_export:
    st.markdown("#### Complete Publication Download Bundle")
    st.markdown("Download individual scientific assets or reports:")

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("**1. Structured Data & Documentation**")
        st.download_button(
            "📄 Download Markdown Report (.md)",
            data=md_report_content.encode("utf-8"),
            file_name=f"{config_data.get('experiment_id')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "📊 Download Metrics Table (.csv)",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{config_data.get('experiment_id')}_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "⚙️ Download Reproducibility Manifest (.json)",
            data=json.dumps(config_data, indent=2).encode("utf-8"),
            file_name=f"{config_data.get('experiment_id')}_config.json",
            mime="application/json",
            use_container_width=True,
        )

    with dcol2:
        st.markdown("**2. Processed Image Maps (PNG)**")
        if exp_dir and exp_dir.exists():
            png_files = sorted(exp_dir.glob("*.png"))
            if png_files:
                st.caption(f"Found {len(png_files)} visual artifacts in `{exp_dir.name}`:")
                for pf in png_files:
                    pbytes = pf.read_bytes()
                    st.download_button(
                        f"🖼️ Download {pf.name}",
                        data=pbytes,
                        file_name=pf.name,
                        mime="image/png",
                        key=f"dl_{pf.name}",
                        use_container_width=True,
                    )
            else:
                st.info("No PNG images cached in experiment directory.")
        else:
            st.info("Run experiment to save image artifacts.")

st.markdown("---")
if st.button("🔄 Start Another Experiment", type="secondary"):
    st.session_state.current_data = None
    st.session_state.current_experiment = None
    st.session_state.analysis_complete = False
    st.session_state.robustness_complete = False
    st.switch_page("pages/analyze.py")
