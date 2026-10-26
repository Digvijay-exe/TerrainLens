"""TerrainLens cooperative drone and ground-rover simulation."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Ensure imports work when Streamlit launches this page from the app folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.cooperative_sim import run_simulation


st.set_page_config(
    page_title="TerrainLens | Cooperative Simulation",
    page_icon="🛰️",
    layout="wide",
)

st.title("TerrainLens | Cooperative Navigation Lab")
st.caption(
    "Software-only research simulation · Aerial terrain scouting · "
    "Ground-rover navigation · Localization evaluation"
)

st.info(
    "All vehicles and sensor readings are simulated. "
    "This page does not connect to or control real vehicles."
)

with st.sidebar:
    st.header("Simulation settings")
    ticks = st.slider("Simulation steps", 20, 150, 65, 5)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=7)
    run_button = st.button("Run simulation", type="primary", use_container_width=True)
    show_truth = st.checkbox("Show ground-truth obstacles for evaluation", value=False)

if run_button:
    with st.spinner("Simulating drone observations and rover navigation..."):
        st.session_state["terrainlens_coop_result"] = run_simulation(
            size=45,
            ticks=int(ticks),
            seed=int(seed),
        )

result = st.session_state.get("terrainlens_coop_result")

if result is None:
    st.subheader("Ready to simulate")
    st.write(
        "Choose the number of steps and random seed in the sidebar, "
        "then select **Run simulation**."
    )
    st.markdown(
        """
        **The first experiment includes**
        - Synthetic elevation and terrain roughness
        - A drone that reveals terrain as it flies
        - A rover that plans routes using its observed map
        - A* route planning with additional costs for unknown terrain
        - A reproducible environment for later localization experiments
        """
    )
    st.stop()

elevation = result["elevation"]
known = result["known"]
obstacles = result["truth_obstacles"]
rover_trace = result["rover_trace"]
drone_trace = result["drone_trace"]
rover_position = result["rover_position"]
start = result["start"]
goal = result["goal"]

total_cells = known.size
coverage = 100 * np.count_nonzero(known) / total_cells
rover_distance = sum(
    np.hypot(b[0] - a[0], b[1] - a[1])
    for a, b in zip(rover_trace, rover_trace[1:])
)
localization_error = 0.0  # Exact simulated position; noisy estimates come later.

m1, m2, m3, m4 = st.columns(4)
m1.metric("Map coverage", f"{coverage:.1f}%")
m2.metric("Rover distance", f"{rover_distance:.1f} cells")
m3.metric("Route replans", str(result["replans"]))
m4.metric(
    "Destination",
    "Reached" if result["reached"] else "Not reached",
)

st.caption(
    "Current rover coordinates: "
    f"({rover_position[0]}, {rover_position[1]}) · "
    f"Ground-truth position error: {localization_error:.1f} cells"
)

# Plot only terrain cells revealed to the rover/drone.
observed_elevation = np.where(known, elevation, np.nan)
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=observed_elevation,
        colorscale="Earth",
        colorbar={"title": "Elevation"},
        hovertemplate="X: %{x}<br>Y: %{y}<br>Elevation: %{z:.2f}<extra></extra>",
        name="Observed terrain",
    )
)

if np.any(obstacles & known):
    oy, ox = np.where(obstacles & known)
    fig.add_trace(
        go.Scatter(
            x=ox,
            y=oy,
            mode="markers",
            marker={"symbol": "x", "size": 7, "color": "black"},
            name="Observed obstacle",
        )
    )

if rover_trace:
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in rover_trace],
            y=[p[1] for p in rover_trace],
            mode="lines+markers",
            line={"color": "#00e5ff", "width": 3},
            marker={"size": 4},
            name="Rover path",
        )
    )

if drone_trace:
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in drone_trace],
            y=[p[1] for p in drone_trace],
            mode="lines",
            line={"color": "#ff4fd8", "width": 2, "dash": "dash"},
            name="Drone flight path",
        )
    )
    dx, dy = drone_trace[-1]
    fig.add_trace(
        go.Scatter(
            x=[dx],
            y=[dy],
            mode="markers",
            marker={"symbol": "triangle-up", "size": 15, "color": "#ff4fd8"},
            name="Drone position",
        )
    )

fig.add_trace(
    go.Scatter(
        x=[start[0]],
        y=[start[1]],
        mode="markers",
        marker={"symbol": "circle", "size": 13, "color": "#00ff88"},
        name="Rover start",
    )
)
fig.add_trace(
    go.Scatter(
        x=[goal[0]],
        y=[goal[1]],
        mode="markers",
        marker={"symbol": "star", "size": 17, "color": "#ffcc00"},
        name="Destination",
    )
)

fig.update_layout(
    title="Shared terrain map — unknown cells remain hidden",
    xaxis_title="Grid X",
    yaxis_title="Grid Y",
    yaxis={"autorange": "reversed", "scaleanchor": "x", "scaleratio": 1},
    height=650,
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    margin={"l": 10, "r": 10, "t": 90, "b": 10},
)

st.plotly_chart(fig, width="stretch")

if show_truth:
    truth_fig = go.Figure(
        go.Heatmap(
            z=elevation,
            colorscale="Earth",
            colorbar={"title": "Elevation"},
        )
    )
    oy, ox = np.where(obstacles)
    truth_fig.add_trace(
        go.Scatter(
            x=ox,
            y=oy,
            mode="markers",
            marker={"symbol": "x", "size": 6, "color": "red"},
            name="All simulated obstacles",
        )
    )
    truth_fig.update_layout(
        title="Ground-truth terrain (evaluation view)",
        xaxis_title="Grid X",
        yaxis_title="Grid Y",
        yaxis={"autorange": "reversed", "scaleanchor": "x", "scaleratio": 1},
        height=600,
    )
    st.plotly_chart(truth_fig, width="stretch")

st.subheader("Experiment notes")
st.write(
    "This initial version measures map coverage and rover movement. "
    "It does not yet estimate noisy localization or perform visual terrain matching. "
    "Those will be separate experimental features so their accuracy can be evaluated honestly."
)

with st.expander("Simulation configuration"):
    st.json(
        {
            "grid_size": list(elevation.shape),
            "random_seed": int(seed),
            "requested_steps": int(ticks),
            "revealed_cells": int(np.count_nonzero(known)),
            "total_cells": int(total_cells),
            "known_obstacles": int(np.count_nonzero(obstacles & known)),
            "ground_truth_obstacles": int(np.count_nonzero(obstacles)),
            "rover_trace_points": len(rover_trace),
            "drone_trace_points": len(drone_trace),
            "destination_reached": bool(result["reached"]),
        }
    )
