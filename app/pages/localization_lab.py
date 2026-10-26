"""TerrainLens synthetic localization evaluation."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.cooperative_sim import run_simulation
from simulation.localization import evaluate_localization

st.set_page_config(
    page_title="TerrainLens | Localization Lab",
    page_icon="📍",
    layout="wide",
)

st.title("TerrainLens | Localization Lab")
st.caption("Synthetic GPS and odometry fusion evaluation")

st.warning(
    "This is a simplified software experiment using synthetic sensor noise. "
    "Results do not represent real GPS hardware or field-tested performance."
)

with st.sidebar:
    st.header("Experiment settings")
    gps_noise = st.slider(
        "GPS noise (grid cells)", 0.0, 8.0, 2.0, 0.25
    )
    odometry_noise = st.slider(
        "Odometry noise (grid cells)", 0.0, 2.0, 0.25, 0.05
    )
    gps_gain = st.slider(
        "GPS correction gain", 0.0, 1.0, 0.25, 0.05
    )
    seed = st.number_input(
        "Sensor random seed", min_value=0, max_value=99999, value=21
    )
    run_button = st.button("Run experiment", type="primary")

if run_button or "terrainlens_localization" not in st.session_state:
    with st.spinner("Generating trajectory and evaluating estimates..."):
        scenario = run_simulation(size=45, ticks=65, seed=7)

        # The rover trace contains its actual simulated grid coordinates.
        result = evaluate_localization(
            scenario["rover_trace"],
            gps_noise=gps_noise,
            odometry_noise=odometry_noise,
            gps_gain=gps_gain,
            seed=int(seed),
        )

        st.session_state["terrainlens_localization"] = result
        st.session_state["terrainlens_localization_settings"] = (
            gps_noise, odometry_noise, gps_gain, int(seed)
        )

result = st.session_state["terrainlens_localization"]
settings = st.session_state["terrainlens_localization_settings"]

# Recalculate when a slider changes and the user presses Run experiment.
if settings != (gps_noise, odometry_noise, gps_gain, int(seed)):
    st.info("Settings changed. Select Run experiment to apply them.")

m1, m2, m3 = st.columns(3)
m1.metric("GPS RMSE", f"{result['gps_rmse']:.3f} cells")
m2.metric("Fused RMSE", f"{result['fused_rmse']:.3f} cells")
m3.metric(
    "RMSE change",
    f"{result['improvement_percent']:+.1f}%",
    help="Positive means fusion reduced error; negative means it increased error.",
)

fig = go.Figure()
for key, label, color, dash in [
    ("truth", "Ground truth", "#00cc96", "solid"),
    ("gps", "Noisy GPS", "#ef553b", "dot"),
    ("fused", "Fused estimate", "#636efa", "solid"),
]:
    positions = result[key]
    fig.add_trace(
        go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode="lines+markers",
            name=label,
            line={"color": color, "dash": dash, "width": 2},
            marker={"size": 4},
        )
    )

fig.update_layout(
    title="Rover position estimates",
    xaxis_title="Grid X",
    yaxis_title="Grid Y",
    yaxis={"scaleanchor": "x", "scaleratio": 1},
    height=550,
)
st.plotly_chart(fig, width="stretch")

errors = go.Figure()
errors.add_trace(
    go.Scatter(
        y=result["gps_error"],
        mode="lines",
        name="GPS position error",
    )
)
errors.add_trace(
    go.Scatter(
        y=result["fused_error"],
        mode="lines",
        name="Fused position error",
    )
)
errors.update_layout(
    title="Position error at each rover sample",
    xaxis_title="Trajectory sample",
    yaxis_title="Euclidean error (grid cells)",
    height=400,
)
st.plotly_chart(errors, width="stretch")

st.subheader("Experiment interpretation")
if result["fused_rmse"] < result["gps_rmse"]:
    st.success(
        "Fusion reduced RMSE in this run. Repeat with multiple seeds "
        "before drawing a general conclusion."
    )
elif result["fused_rmse"] > result["gps_rmse"]:
    st.warning(
        "Fusion increased RMSE in this run. The filter gain and sensor "
        "noise assumptions may need investigation."
    )
else:
    st.info("GPS and fused RMSE are equal in this run.")

st.write(
    "The filter predicts motion from noisy odometry and corrects that "
    "prediction toward GPS measurements. It uses a fixed correction gain, "
    "not a full Kalman filter. The first position is initialized from GPS."
)

st.download_button(
    "Export localization results (CSV)",
    data=(
        "sample,true_x,true_y,gps_x,gps_y,fused_x,fused_y,"
        "gps_error,fused_error\n"
        + "\n".join(
            f"{i},{result['truth'][i,0]:.6f},{result['truth'][i,1]:.6f},"
            f"{result['gps'][i,0]:.6f},{result['gps'][i,1]:.6f},"
            f"{result['fused'][i,0]:.6f},{result['fused'][i,1]:.6f},"
            f"{result['gps_error'][i]:.6f},{result['fused_error'][i]:.6f}"
            for i in range(len(result["truth"]))
        )
    ),
    file_name="terrainlens_localization_experiment.csv",
    mime="text/csv",
)
