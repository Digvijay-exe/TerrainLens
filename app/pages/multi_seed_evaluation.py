"""TerrainLens multi-seed localization dashboard."""

import csv
import io
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.cooperative_sim import run_simulation
from simulation.multi_seed_evaluation import run_multi_seed_evaluation

st.set_page_config(
    page_title="TerrainLens | Multi-Seed Evaluation",
    layout="wide",
)

st.title("TerrainLens | Multi-Seed Localization Evaluation")
st.caption("Repeated synthetic trials: GPS versus GPS/odometry fusion")

st.warning(
    "Synthetic simulation only. These results do not establish real-world "
    "vehicle performance. Confidence intervals use a normal approximation."
)

with st.sidebar:
    st.header("Experiment settings")
    n = st.slider("Number of sensor seeds", 10, 200, 30, 10)
    base_seed = st.number_input(
        "First sensor seed", min_value=0, max_value=1000000, value=100
    )
    gps_noise = st.slider("GPS noise (grid cells)", 0.0, 8.0, 2.0, 0.25)
    odom_noise = st.slider(
        "Odometry noise (grid cells)", 0.0, 2.0, 0.25, 0.05
    )
    gain = st.slider("GPS correction gain", 0.0, 1.0, 0.25, 0.05)
    trajectory_seed = st.number_input(
        "Trajectory seed", min_value=0, max_value=99999, value=7
    )
    run = st.button("Run multi-seed experiment", type="primary")

if run:
    with st.spinner(f"Running {n} trials..."):
        scenario = run_simulation(
            size=45, ticks=65, seed=int(trajectory_seed)
        )
        result = run_multi_seed_evaluation(
            scenario["rover_trace"],
            num_seeds=int(n),
            base_seed=int(base_seed),
            gps_noise=float(gps_noise),
            odometry_noise=float(odom_noise),
            gps_gain=float(gain),
        )
        result["trajectory_seed"] = int(trajectory_seed)
        st.session_state["multi_seed_result"] = result

result = st.session_state.get("multi_seed_result")
if result is None:
    st.info("Choose settings in the sidebar and run the experiment.")
    st.stop()

gps = result["gps"]
fused = result["fused"]
improvement = result["improvement"]

a, b, c, d = st.columns(4)
a.metric("Trials", result["num_seeds"])
b.metric("Mean GPS RMSE", f"{gps['mean']:.3f} cells")
c.metric("Mean fused RMSE", f"{fused['mean']:.3f} cells")
d.metric("Fusion wins", f"{result['fusion_better_percent']:.1f}%")

st.write(
    f"Fixed trajectory seed: {result['trajectory_seed']} · "
    f"Sensor seeds: {result['base_seed']}–"
    f"{result['base_seed'] + result['num_seeds'] - 1}"
)

summary_rows = []
for name, values in [
    ("GPS RMSE (cells)", gps),
    ("Fused RMSE (cells)", fused),
    ("RMSE improvement (%)", improvement),
]:
    summary_rows.append({
        "Metric": name,
        "Mean": values["mean"],
        "Median": values["median"],
        "Standard deviation": values["std"],
        "95% CI lower": values["ci95_low"],
        "95% CI upper": values["ci95_high"],
        "Minimum": values["min"],
        "Maximum": values["max"],
    })

st.subheader("Statistical summary")
st.dataframe(summary_rows, hide_index=True, width="stretch")

fig = go.Figure()
fig.add_trace(go.Box(
    y=result["gps_values"], name="Noisy GPS", boxpoints="all"
))
fig.add_trace(go.Box(
    y=result["fused_values"], name="GPS + odometry", boxpoints="all"
))
fig.update_layout(
    title="RMSE distribution across trials",
    yaxis_title="Position RMSE (grid cells)",
    height=450,
)
st.plotly_chart(fig, width="stretch")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=[r["seed"] for r in result["runs"]],
    y=[r["gps_rmse"] for r in result["runs"]],
    mode="lines+markers", name="GPS RMSE",
))
fig2.add_trace(go.Scatter(
    x=[r["seed"] for r in result["runs"]],
    y=[r["fused_rmse"] for r in result["runs"]],
    mode="lines+markers", name="Fused RMSE",
))
fig2.update_layout(
    title="Error by sensor seed",
    xaxis_title="Sensor seed",
    yaxis_title="RMSE (grid cells)",
    height=400,
)
st.plotly_chart(fig2, width="stretch")

if result["fusion_better_percent"] > 50:
    st.success(
        "Fusion had lower RMSE in more than half of these simulated trials. "
        "This is limited to the tested configuration."
    )
else:
    st.warning(
        "Fusion did not win in more than half of these trials. "
        "Review the noise assumptions and filter gain."
    )

per_seed = io.StringIO()
writer = csv.DictWriter(
    per_seed,
    fieldnames=[
        "seed", "gps_rmse", "fused_rmse",
        "improvement_percent", "fusion_better",
    ],
)
writer.writeheader()
writer.writerows(result["runs"])

st.download_button(
    "Download per-seed results (CSV)",
    data=per_seed.getvalue(),
    file_name="terrainlens_multi_seed_per_seed.csv",
    mime="text/csv",
)

summary_csv = io.StringIO()
writer = csv.writer(summary_csv)
writer.writerow([
    "metric", "mean", "median", "std_dev",
    "ci95_low", "ci95_high", "minimum", "maximum",
])
for name, values in [
    ("gps_rmse", gps),
    ("fused_rmse", fused),
    ("improvement_percent", improvement),
]:
    writer.writerow([
        name, values["mean"], values["median"], values["std"],
        values["ci95_low"], values["ci95_high"],
        values["min"], values["max"],
    ])

st.download_button(
    "Download statistical summary (CSV)",
    data=summary_csv.getvalue(),
    file_name="terrainlens_multi_seed_summary.csv",
    mime="text/csv",
)

with st.expander("Method and limitations"):
    st.markdown(
        """
        - All trials use the same simulated rover trajectory.
        - Sensor-noise seeds vary between trials.
        - GPS and fusion are compared using the same GPS observations
          within each trial.
        - Fusion is the existing fixed-gain estimator, not a full Kalman filter.
        - The confidence interval is a normal approximation for the mean.
        - These are repeated synthetic trials, not independent field tests.
        """
    )
