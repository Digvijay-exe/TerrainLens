"""Software-only TerrainLens drone and ground-rover simulation."""

import heapq
import math

import numpy as np


def generate_terrain(size=45, seed=7):
    """Create synthetic elevation, roughness, and obstacle data."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]

    elevation = (
        12 * np.exp(-((x - size * 0.30) ** 2 + (y - size * 0.35) ** 2) / 100)
        + 9 * np.exp(-((x - size * 0.72) ** 2 + (y - size * 0.68) ** 2) / 130)
        + 3 * np.sin(x / 5) * np.cos(y / 7)
        + rng.normal(0, 0.35, (size, size))
    )

    gy, gx = np.gradient(elevation)
    slope = np.hypot(gx, gy)
    roughness = np.abs(
        elevation
        - (
            np.roll(elevation, 1, axis=0)
            + np.roll(elevation, -1, axis=0)
            + np.roll(elevation, 1, axis=1)
            + np.roll(elevation, -1, axis=1)
        ) / 4
    )

    obstacles = np.zeros((size, size), dtype=bool)
    for _ in range(size // 2):
        cx, cy = rng.integers(4, size - 4, size=2)
        radius = int(rng.integers(1, 3))
        obstacles |= (x - cx) ** 2 + (y - cy) ** 2 <= radius**2

    # Keep the start and destination clear.
    obstacles[2:5, 2:5] = False
    obstacles[size - 5:size - 2, size - 5:size - 2] = False

    return elevation, slope, roughness, obstacles


def reveal_area(known, center, radius):
    """Reveal a circular area of the simulated terrain map."""
    size = known.shape[0]
    cx, cy = center
    y, x = np.ogrid[:size, :size]
    known |= (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def astar(start, goal, elevation, slope, roughness, obstacles, known):
    """A* route planner; unknown terrain has an additional cost."""
    size = elevation.shape[0]
    sx, sy = start
    gx, gy = goal

    frontier = [(0.0, start)]
    came_from = {}
    cost_so_far = {start: 0.0}
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),              (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        cx, cy = current
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            if obstacles[ny, nx]:
                continue

            step = math.sqrt(2) if dx and dy else 1.0
            terrain_cost = (
                1.0
                + 1.8 * float(slope[ny, nx])
                + 0.8 * float(roughness[ny, nx])
            )
            if not known[ny, nx]:
                terrain_cost += 3.0

            new_cost = cost_so_far[current] + step * terrain_cost
            neighbor = (nx, ny)

            if new_cost < cost_so_far.get(neighbor, float("inf")):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heuristic = math.hypot(gx - nx, gy - ny)
                heapq.heappush(
                    frontier, (new_cost + heuristic, neighbor)
                )

    if goal not in cost_so_far:
        return []

    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def run_simulation(size=45, ticks=65, seed=7):
    """Run a repeatable drone-scout and rover-navigation scenario."""
    elevation, slope, roughness, truth_obstacles = generate_terrain(size, seed)

    start = (2, 2)
    goal = (size - 3, size - 3)
    known = np.zeros((size, size), dtype=bool)

    # The drone sweeps back and forth across the map.
    waypoints = []
    for row in range(3, size - 2, 6):
        xs = range(3, size - 3) if len(waypoints) % 2 == 0 else range(size - 4, 2, -1)
        waypoints.extend((x, row) for x in xs)

    drone_position = waypoints[0]
    drone_trace = []
    rover_trace = [start]
    rover_position = start
    reached = False
    replans = 0

    reveal_area(known, drone_position, 6)
    reveal_area(known, rover_position, 2)

    for tick in range(ticks):
        drone_position = waypoints[min(tick, len(waypoints) - 1)]
        drone_trace.append(drone_position)
        reveal_area(known, drone_position, 6)

        # Simulated local sensing around the rover.
        reveal_area(known, rover_position, 2)

        # Only obstacles discovered by the simulated sensors are known.
        observed_obstacles = truth_obstacles & known
        path = astar(
            rover_position, goal, elevation, slope, roughness,
            observed_obstacles, known
        )
        replans += 1

        if not path:
            break
        if rover_position == goal:
            reached = True
            break

        if len(path) > 1:
            next_position = path[1]
            nx, ny = next_position

            # If an unknown obstacle is encountered, the rover stops.
            if truth_obstacles[ny, nx]:
                reveal_area(known, next_position, 2)
            else:
                rover_position = next_position
                rover_trace.append(rover_position)

        if rover_position == goal:
            reached = True
            break

    # Final route for display, based on what the rover has observed.
    final_path = astar(
        rover_position, goal, elevation, slope, roughness,
        truth_obstacles & known, known
    )

    return {
        "elevation": elevation,
        "slope": slope,
        "roughness": roughness,
        "truth_obstacles": truth_obstacles,
        "known": known,
        "start": start,
        "goal": goal,
        "drone_trace": drone_trace,
        "rover_trace": rover_trace,
        "rover_position": rover_position,
        "final_path": final_path,
        "replans": replans,
        "reached": reached,
        "ticks": ticks,
    }
