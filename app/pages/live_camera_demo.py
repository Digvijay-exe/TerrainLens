"""TerrainLens live phone-camera and A* classroom demonstration."""

import heapq
import time
import threading
from urllib.parse import urlparse

import cv2
import numpy as np
import streamlit as st
import math
import time
import plotly.graph_objects as go


st.set_page_config(
    page_title="TerrainLens | Live Camera Demo",
    page_icon="📷",
    layout="wide",
)

GRID_SIZE = 45


class LatestFrameReader:
    """Continuously read the stream and retain only the latest frame."""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(
            target=self._read_loop, daemon=True
        )
        self.thread.start()

    def _read_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        if not self.thread.is_alive():
            self.cap.release()
        else:
            # Do not release a capture while its reader is still using it.
            # A daemon thread prevents shutdown from hanging indefinitely.
            pass



def find_path(occupied, start, goal):
    """Find an 8-connected A* route through unoccupied grid cells."""
    rows, cols = occupied.shape

    if occupied[start] or occupied[goal]:
        return None

    def heuristic(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    queue = [(heuristic(start, goal), 0, start)]
    parent = {}
    cost = {start: 0}
    visited = set()

    moves = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while queue:
        _, current_cost, current = heapq.heappop(queue)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return path[::-1]

        for dr, dc in moves:
            nxt = (current[0] + dr, current[1] + dc)

            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if occupied[nxt]:
                continue

            step_cost = 1.4142 if dr and dc else 1.0
            new_cost = current_cost + step_cost

            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                parent[nxt] = current
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(queue, (priority, new_cost, nxt))

    return None


def red_obstacle_grid(frame):
    """Convert visible red regions into a coarse obstacle grid."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around both ends of OpenCV's hue scale.
    red_low = cv2.inRange(
        hsv, np.array([0, 100, 70]), np.array([12, 255, 255])
    )
    red_high = cv2.inRange(
        hsv, np.array([168, 100, 70]), np.array([180, 255, 255])
    )
    mask = cv2.bitwise_or(red_low, red_high)

    # Downsample to estimate which cells contain red pixels.
    small = cv2.resize(
        mask, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA
    )
    occupied = small > 35

    # Expand obstacles slightly to avoid routes grazing detected regions.
    occupied_u8 = occupied.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    occupied = cv2.dilate(occupied_u8, kernel, iterations=1).astype(bool)

    return occupied, mask


def draw_route(frame, path, occupied):
    """Overlay grid, detected obstacles, start, goal, and A* route."""
    output = frame.copy()
    height, width = output.shape[:2]
    cell_w = width / GRID_SIZE
    cell_h = height / GRID_SIZE

    # Red tint shows the detected obstacle mask.
    _, mask = red_obstacle_grid(frame)
    red_layer = np.zeros_like(output)
    red_layer[:, :, 2] = 255
    selected = mask > 0
    output[selected] = (
        0.45 * output[selected] + 0.55 * red_layer[selected]
    ).astype(np.uint8)

    # Draw the planned route as a green line.
    if path:
        points = np.array([
            [
                int((col + 0.5) * cell_w),
                int((row + 0.5) * cell_h),
            ]
            for row, col in path
        ], dtype=np.int32)

        if len(points) >= 2:
            cv2.polylines(
                output, [points], False, (0, 255, 0), 3,
                lineType=cv2.LINE_AA,
            )

        cv2.circle(
            output, tuple(points[0]), 7, (255, 255, 0), -1
        )
        cv2.circle(
            output, tuple(points[-1]), 7, (255, 0, 255), -1
        )

    return output



def make_3d_route_figure(path, occupied, rover_pos=None):
    """Render the image-grid route as a lightweight 3D-style navigation scene."""
    fig = go.Figure()

    # Ground plane grid, shown as subtle lines for depth cues.
    for k in range(0, GRID_SIZE, 5):
        fig.add_trace(go.Scatter3d(
            x=[0, GRID_SIZE - 1], y=[k, k], z=[0, 0],
            mode="lines", line=dict(color="rgba(80,170,190,0.20)", width=1),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[k, k], y=[0, GRID_SIZE - 1], z=[0, 0],
            mode="lines", line=dict(color="rgba(80,170,190,0.20)", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    # Detected obstacles as red raised blocks (limited for smooth rendering).
    obstacle_rows, obstacle_cols = np.where(occupied)
    if len(obstacle_rows):
        stride = max(1, int(np.ceil(len(obstacle_rows) / 180)))
        obstacle_rows = obstacle_rows[::stride]
        obstacle_cols = obstacle_cols[::stride]
        fig.add_trace(go.Scatter3d(
            x=obstacle_cols, y=GRID_SIZE - 1 - obstacle_rows,
            z=np.full(len(obstacle_rows), 0.55),
            mode="markers",
            marker=dict(size=5, symbol="square", color="#ff365e",
                        line=dict(color="#ff8a9e", width=1)),
            name="Detected obstacles",
            hovertemplate="Obstacle estimate<extra></extra>",
        ))

    if path:
        xs = [cell[1] for cell in path]
        ys = [GRID_SIZE - 1 - cell[0] for cell in path]
        # Soft underglow plus bright raised route ribbon.
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=[0.18] * len(path),
            mode="lines",
            line=dict(color="#00ff85", width=10),
            name="Planned route",
            hovertemplate="Route cell<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=[0.22] * len(path),
            mode="lines",
            line=dict(color="#baffd8", width=3),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=[xs[0]], y=[ys[0]], z=[0.7],
            mode="markers+text",
            marker=dict(size=8, color="#00e5ff", symbol="circle",
                        line=dict(color="white", width=1)),
            text=["START"], textposition="top center",
            textfont=dict(color="#b9f8ff", size=10),
            name="Start",
        ))
        fig.add_trace(go.Scatter3d(
            x=[xs[-1]], y=[ys[-1]], z=[0.7],
            mode="markers+text",
            marker=dict(size=9, color="#ff4dff", symbol="diamond",
                        line=dict(color="white", width=1)),
            text=["GOAL"], textposition="top center",
            textfont=dict(color="#ffd2ff", size=10),
            name="Destination",
        ))

    # TERRAINLENS_VIRTUAL_ROVER: simulated vehicle marker.
    if rover_pos is not None:
        rover_row, rover_col = rover_pos
        fig.add_trace(go.Scatter3d(
            x=[rover_col],
            y=[GRID_SIZE - 1 - rover_row],
            z=[0.95],
            mode="markers+text",
            marker=dict(
                size=10,
                color="#ffe66d",
                symbol="diamond",
                line=dict(color="white", width=2),
            ),
            text=["VIRTUAL ROVER"],
            textposition="top center",
            textfont=dict(color="#ffe66d", size=10),
            name="Virtual rover",
            hovertemplate="Simulated rover<extra></extra>",
        ))

    fig.update_layout(
        height=470,
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text="TERRAINLENS // 3D ROUTE PREVIEW", x=0.03,
                   font=dict(color="#c8fff0", size=15)),
        paper_bgcolor="#0a1017",
        plot_bgcolor="#0a1017",
        font=dict(color="#b7cbd4", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=0.01,
                    xanchor="left", x=0.02,
                    bgcolor="rgba(8,16,24,0.65)"),
        scene=dict(
            bgcolor="#0a1017",
            xaxis=dict(title="", range=[0, GRID_SIZE - 1],
                       showbackground=True, backgroundcolor="#101c25",
                       gridcolor="rgba(80,170,190,0.15)", zeroline=False,
                       showticklabels=False),
            yaxis=dict(title="", range=[0, GRID_SIZE - 1],
                       showbackground=True, backgroundcolor="#101c25",
                       gridcolor="rgba(80,170,190,0.15)", zeroline=False,
                       showticklabels=False),
            zaxis=dict(title="", range=[0, 5], visible=False),
            aspectmode="manual",
            aspectratio=dict(x=1.25, y=1.0, z=0.35),
            camera=dict(eye=dict(x=1.55, y=-1.8, z=1.25),
                        center=dict(x=0, y=0, z=-0.15)),
        ),
        showlegend=True,
    )
    return fig

def valid_local_stream_url(url):
    """Only accept an HTTP(S) URL with a host and no embedded credentials."""
    try:
        parsed = urlparse(url.strip())
        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


st.title("TerrainLens | Live Camera Navigation Demo")
st.caption("Android camera → obstacle estimate → A* route")

st.warning(
    "Classroom prototype only. Red-colour detection is not general object "
    "recognition, depth estimation, terrain understanding, or real-vehicle "
    "navigation. The planned route is in image-grid coordinates."
)

with st.sidebar:
    st.header("Phone camera")
    camera_url = st.text_input(
        "Local camera stream URL",
        placeholder="http://192.168.1.100:8080/video",
        help="Enter the video stream URL shown by your phone camera app.",
    )
    st.caption("Both devices must be on the same trusted Wi-Fi network.")
    start = st.button("Start / reconnect camera", type="primary")
    stop = st.button("Stop camera")

if stop:
    reader = st.session_state.pop("terrainlens_camera_reader", None)
    if reader is not None:
        reader.stop()
    cap = st.session_state.pop("terrainlens_camera_cap", None)
    if cap is not None and reader is None:
        cap.release()
    st.session_state["terrainlens_camera_stopped"] = True

if start:
    old_reader = st.session_state.pop("terrainlens_camera_reader", None)
    if old_reader is not None:
        old_reader.stop()
    old_cap = st.session_state.pop("terrainlens_camera_cap", None)
    if old_cap is not None and old_reader is None:
        old_cap.release()

    if not valid_local_stream_url(camera_url):
        st.error("Enter a valid HTTP(S) stream URL without credentials.")
    else:
        with st.spinner("Connecting to the phone camera..."):
            cap = cv2.VideoCapture(camera_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Give the stream a brief chance to provide a frame.
            frame_ok = False
            frame = None
            for _ in range(10):
                frame_ok, frame = cap.read()
                if frame_ok and frame is not None:
                    break
                time.sleep(0.15)

            if frame_ok:
                reader = LatestFrameReader(cap)
                st.session_state["terrainlens_camera_reader"] = reader
                st.session_state["terrainlens_camera_cap"] = cap
                st.session_state["terrainlens_camera_url"] = camera_url
                st.session_state["terrainlens_camera_stopped"] = False
                st.success("Phone camera connected.")
            else:
                cap.release()
                st.error(
                    "Could not read a frame. Check the phone stream URL, "
                    "phone server status, Wi-Fi, and firewall."
                )

reader = st.session_state.get("terrainlens_camera_reader")

if reader is None:
    st.info(
        "Start the phone camera server, enter its stream URL in the sidebar, "
        "then click Start / reconnect camera."
    )
    st.stop()

# Virtual rover controls and persistent simulation state.
if "terrainlens_rover_running" not in st.session_state:
    st.session_state.terrainlens_rover_running = False
if "terrainlens_rover_position" not in st.session_state:
    st.session_state.terrainlens_rover_position = None
if "terrainlens_rover_last_tick" not in st.session_state:
    st.session_state.terrainlens_rover_last_tick = time.monotonic()
if "terrainlens_rover_distance" not in st.session_state:
    st.session_state.terrainlens_rover_distance = 0.0

st.subheader("03 // VIRTUAL ROVER SIMULATION")
control_start, control_pause, control_reset = st.columns(3)

with control_start:
    if st.button("▶ Start", key="terrainlens_rover_start", width="stretch"):
        if st.session_state.terrainlens_rover_position is None:
            st.session_state.terrainlens_rover_position = [
                float(GRID_SIZE // 2), 1.0
            ]
            st.session_state.terrainlens_rover_distance = 0.0
        st.session_state.terrainlens_rover_running = True
        st.session_state.terrainlens_rover_last_tick = time.monotonic()

with control_pause:
    if st.button("⏸ Pause", key="terrainlens_rover_pause", width="stretch"):
        st.session_state.terrainlens_rover_running = False
        st.session_state.terrainlens_rover_last_tick = time.monotonic()

with control_reset:
    if st.button("↺ Reset", key="terrainlens_rover_reset", width="stretch"):
        st.session_state.terrainlens_rover_running = False
        st.session_state.terrainlens_rover_position = None
        st.session_state.terrainlens_rover_distance = 0.0
        st.session_state.terrainlens_rover_last_tick = time.monotonic()


@st.fragment(run_every=0.2)
def live_view():
    reader = st.session_state.get("terrainlens_camera_reader")

    if reader is None:
        st.info("Camera stopped.")
        return

    frame = reader.get_frame()
    if frame is None:
        st.error(
            "The camera stream stopped providing frames. Try reconnecting."
        )
        return

    occupied, _ = red_obstacle_grid(frame)

    start_cell = (GRID_SIZE // 2, 1)
    goal_cell = (GRID_SIZE // 2, GRID_SIZE - 2)

    # Use the rover's current grid cell as the next planning start.
    position = st.session_state.terrainlens_rover_position
    if position is not None:
        start_cell = (
            max(0, min(GRID_SIZE - 1, int(round(position[0])))),
            max(0, min(GRID_SIZE - 1, int(round(position[1])))),
        )

    # Keep the rover's current cell and destination free in this demo.
    occupied[start_cell] = False
    occupied[goal_cell] = False

    path = find_path(occupied, start_cell, goal_cell)

    now = time.monotonic()
    last_tick = st.session_state.terrainlens_rover_last_tick
    dt = max(0.0, min(now - last_tick, 0.5))
    st.session_state.terrainlens_rover_last_tick = now

    position = st.session_state.terrainlens_rover_position

    if st.session_state.terrainlens_rover_running:
        if not path:
            st.session_state.terrainlens_rover_running = False
            st.warning("No route available. The virtual rover has paused.")
        else:
            if position is None:
                position = [float(start_cell[0]), float(start_cell[1])]

            # Follow the next A* waypoint at a simulated 2 grid cells/second.
            if len(path) > 1:
                target_row, target_col = path[1]
                dr = float(target_row) - position[0]
                dc = float(target_col) - position[1]
                distance = math.hypot(dr, dc)
                step = 2.0 * dt

                if distance > 0:
                    travel = min(step, distance)
                    position[0] += dr / distance * travel
                    position[1] += dc / distance * travel
                    st.session_state.terrainlens_rover_distance += travel

            st.session_state.terrainlens_rover_position = position

            # Stop when the destination is reached.
            if (
                math.hypot(
                    position[0] - goal_cell[0],
                    position[1] - goal_cell[1],
                ) < 0.12
            ):
                st.session_state.terrainlens_rover_running = False
                st.success("Virtual rover reached the destination.")
    elif position is None:
        # Show where the rover will start before simulation begins.
        position = [float(GRID_SIZE // 2), 1.0]

    visual = draw_route(frame, path, occupied)

    left, right = st.columns([1, 1], gap="medium")

    with left:
        st.subheader("01 // LIVE CAMERA")
        st.image(
            cv2.cvtColor(visual, cv2.COLOR_BGR2RGB),
            caption=(
                "Red regions = detected obstacles · "
                "green = planned route"
            ),
            width="stretch",
        )

    with right:
        st.subheader("02 // 3D ROUTE SCENE")
        st.plotly_chart(
            make_3d_route_figure(path, occupied, rover_pos=position),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
            key="terrainlens_live_3d_scene",
        )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Grid", f"{GRID_SIZE} × {GRID_SIZE}")
    metric2.metric("Occupied cells", f"{int(occupied.sum())}")
    metric3.metric("Planned path", f"{len(path)} cells" if path else "No route")

    rover_status = (
        "MOVING"
        if st.session_state.terrainlens_rover_running
        else "PAUSED"
    )
    st.caption(
        f"Virtual rover: {rover_status} · "
        f"Simulated distance: "
        f"{st.session_state.terrainlens_rover_distance:.1f} grid cells"
    )

    if path:
        st.success(
            "A* route available. The virtual rover follows the route "
            "when the simulation is running."
        )
    else:
        st.error("No route found through the current obstacle grid.")

    st.caption(
        "This is a 2D grid-navigation simulation displayed in a 3D scene. "
        "Obstacle detection is based on red image regions, not general "
        "object recognition. It is not safe for real vehicle control."
    )


live_view()

