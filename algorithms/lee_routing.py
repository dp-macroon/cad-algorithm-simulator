"""
Lee's Maze Routing — Algorithm Module
======================================

Provides:
  - lee_route_4dir  : BFS (Lee) for 4-directional movement (unit cost, provably optimal).
  - lee_route_8dir  : Dijkstra for 8-directional movement (diagonal cost = √2).
  - record_steps    : Returns ordered list of step dicts for step-by-step playback.
  - validate_grid   : Input validation.

All functions are pure and deterministic.
"""

from __future__ import annotations

import math
import heapq
import time
from collections import deque
from typing import Any

MAX_ROWS = 50
MAX_COLS = 80
MAX_CELLS_FOR_STEP_RECORD = 2000  # above this, skip per-cell step recording

FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]
DIAGONAL_COST = math.sqrt(2)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_grid(
    rows: int,
    cols: int,
    source: tuple[int, int],
    destination: tuple[int, int],
    obstacles: set[tuple[int, int]],
) -> list[str]:
    """Return list of error/warning strings. Empty = valid."""
    errors = []
    if rows < 2 or cols < 2:
        errors.append(f"Grid must be at least 2×2 (got {rows}×{cols}).")
    if rows > MAX_ROWS:
        errors.append(f"Row count {rows} exceeds maximum {MAX_ROWS}.")
    if cols > MAX_COLS:
        errors.append(f"Column count {cols} exceeds maximum {MAX_COLS}.")
    if not (0 <= source[0] < rows and 0 <= source[1] < cols):
        errors.append(f"Source {source} is outside the grid ({rows}×{cols}).")
    if not (0 <= destination[0] < rows and 0 <= destination[1] < cols):
        errors.append(f"Destination {destination} is outside the grid ({rows}×{cols}).")
    if source == destination:
        errors.append("Source and destination are the same cell.")
    if source in obstacles:
        errors.append(f"Source {source} is blocked by an obstacle.")
    if destination in obstacles:
        errors.append(f"Destination {destination} is blocked by an obstacle.")
    out_of_bounds = [(r, c) for r, c in obstacles if not (0 <= r < rows and 0 <= c < cols)]
    if out_of_bounds:
        errors.append(f"{len(out_of_bounds)} obstacle(s) are outside the grid and will be ignored.")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Core BFS (4-directional, unit cost)
# ─────────────────────────────────────────────────────────────────────────────

def lee_route_4dir(
    rows: int,
    cols: int,
    source: tuple[int, int],
    destination: tuple[int, int],
    obstacles: set[tuple[int, int]],
) -> dict[str, Any]:
    """
    Lee's BFS maze routing (4-directional, unit cost).

    Returns a dict with:
      found, path, distance_map, visited_order, steps, path_length,
      cells_explored, runtime_ms
    """
    obstacles = {(r, c) for r, c in obstacles if 0 <= r < rows and 0 <= c < cols}

    queue: deque = deque([source])
    distance: dict[tuple, int] = {source: 0}
    parent: dict[tuple, tuple | None] = {source: None}
    visited_order: list[tuple] = []
    steps: list[dict] = []

    record_steps_flag = rows * cols <= MAX_CELLS_FOR_STEP_RECORD

    step_num = 0
    found = False

    # Step 0 — initialise source
    step_num += 1
    steps.append({
        "step_index": step_num,
        "phase": "init",
        "current": source,
        "wavefront": [source],
        "explored": [],
        "distance_map": {source: 0},
        "path": [],
        "description": f"Initialise. Source {source} placed in queue with distance 0.",
    })

    t0 = time.perf_counter()

    while queue:
        current = queue.popleft()
        visited_order.append(current)

        if current == destination:
            found = True
            break

        r, c = current
        newly_queued = []
        for dr, dc in FOUR_DIRS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if nxt in obstacles or nxt in distance:
                continue
            distance[nxt] = distance[current] + 1
            parent[nxt] = current
            queue.append(nxt)
            newly_queued.append(nxt)

        if record_steps_flag:
            step_num += 1
            steps.append({
                "step_index": step_num,
                "phase": "expand",
                "current": current,
                "wavefront": list(queue),
                "explored": list(visited_order),
                "newly_queued": newly_queued,
                "distance_map": dict(distance),
                "path": [],
                "description": (
                    f"Expand cell {current} (distance={distance.get(current, '?')}). "
                    f"Queued {len(newly_queued)} new neighbour(s): {newly_queued}."
                ),
            })

    runtime_ms = (time.perf_counter() - t0) * 1000

    if not found:
        step_num += 1
        steps.append({
            "step_index": step_num,
            "phase": "no_route",
            "current": None,
            "wavefront": [],
            "explored": list(visited_order),
            "distance_map": dict(distance),
            "path": [],
            "description": f"Queue exhausted. No path from {source} to {destination}.",
        })
        return {
            "found": False,
            "path": [],
            "distance_map": distance,
            "visited_order": visited_order,
            "steps": steps,
            "path_length": None,
            "cells_explored": len(visited_order),
            "runtime_ms": runtime_ms,
        }

    # Backtrace
    step_num += 1
    steps.append({
        "step_index": step_num,
        "phase": "found_destination",
        "current": destination,
        "wavefront": [],
        "explored": list(visited_order),
        "distance_map": dict(distance),
        "path": [],
        "description": (
            f"Destination {destination} reached! "
            f"Distance = {distance[destination]}. "
            f"Beginning backtracing..."
        ),
    })

    path: list[tuple] = []
    node: tuple | None = destination
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()

    # Add backtrace steps
    for i in range(len(path)):
        partial = path[:i+1]
        step_num += 1
        steps.append({
            "step_index": step_num,
            "phase": "backtrace",
            "current": path[i],
            "wavefront": [],
            "explored": list(visited_order),
            "distance_map": dict(distance),
            "path": list(partial),
            "description": (
                f"Backtracing step {i+1}/{len(path)}: "
                f"at cell {path[i]}, distance={distance.get(path[i], 0)}. "
                f"Path so far: {list(partial)}."
            ),
        })

    step_num += 1
    steps.append({
        "step_index": step_num,
        "phase": "complete",
        "current": None,
        "wavefront": [],
        "explored": list(visited_order),
        "distance_map": dict(distance),
        "path": list(path),
        "description": (
            f"Routing complete. Shortest path found: {len(path)-1} moves, "
            f"{len(path)} cells. Explored {len(visited_order)} cells total."
        ),
    })

    return {
        "found": True,
        "path": path,
        "distance_map": distance,
        "visited_order": visited_order,
        "steps": steps,
        "path_length": len(path) - 1,
        "cells_explored": len(visited_order),
        "runtime_ms": runtime_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dijkstra (8-directional, diagonal cost = √2)
# ─────────────────────────────────────────────────────────────────────────────

def lee_route_8dir(
    rows: int,
    cols: int,
    source: tuple[int, int],
    destination: tuple[int, int],
    obstacles: set[tuple[int, int]],
) -> dict[str, Any]:
    """
    8-directional routing using Dijkstra's algorithm.
    Cardinal moves cost 1.0; diagonal moves cost √2 ≈ 1.414.

    Returns same dict structure as lee_route_4dir.
    Note: path_length is reported as the total *move count* (number of edges),
    and total_cost is the sum of move costs.
    """
    obstacles = {(r, c) for r, c in obstacles if 0 <= r < rows and 0 <= c < cols}

    dist: dict[tuple, float] = {source: 0.0}
    parent: dict[tuple, tuple | None] = {source: None}
    heap: list = [(0.0, source)]
    visited_order: list[tuple] = []
    visited_set: set = set()
    steps: list[dict] = []
    record_steps_flag = rows * cols <= MAX_CELLS_FOR_STEP_RECORD

    step_num = 0
    found = False

    step_num += 1
    steps.append({
        "step_index": step_num,
        "phase": "init",
        "current": source,
        "wavefront": [source],
        "explored": [],
        "distance_map": {source: 0.0},
        "path": [],
        "description": (
            f"Initialise Dijkstra (8-dir). "
            f"Source {source} placed in priority queue with cost 0."
        ),
    })

    t0 = time.perf_counter()

    while heap:
        cost, current = heapq.heappop(heap)
        if current in visited_set:
            continue
        visited_set.add(current)
        visited_order.append(current)

        if current == destination:
            found = True
            break

        r, c = current
        newly_queued = []
        for dr, dc in EIGHT_DIRS:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if nxt in obstacles or nxt in visited_set:
                continue
            move_cost = DIAGONAL_COST if (dr != 0 and dc != 0) else 1.0
            new_cost = cost + move_cost
            if nxt not in dist or new_cost < dist[nxt]:
                dist[nxt] = new_cost
                parent[nxt] = current
                heapq.heappush(heap, (new_cost, nxt))
                newly_queued.append(nxt)

        if record_steps_flag:
            step_num += 1
            steps.append({
                "step_index": step_num,
                "phase": "expand",
                "current": current,
                "wavefront": [item[1] for item in heap],
                "explored": list(visited_order),
                "newly_queued": newly_queued,
                "distance_map": dict(dist),
                "path": [],
                "description": (
                    f"Dijkstra: expand cell {current} (cost={cost:.3f}). "
                    f"Relaxed {len(newly_queued)} neighbour(s)."
                ),
            })

    runtime_ms = (time.perf_counter() - t0) * 1000

    if not found:
        step_num += 1
        steps.append({
            "step_index": step_num,
            "phase": "no_route",
            "current": None,
            "wavefront": [],
            "explored": list(visited_order),
            "distance_map": dict(dist),
            "path": [],
            "description": f"No path found from {source} to {destination}.",
        })
        return {
            "found": False,
            "path": [],
            "distance_map": dist,
            "visited_order": visited_order,
            "steps": steps,
            "path_length": None,
            "total_cost": None,
            "cells_explored": len(visited_order),
            "runtime_ms": runtime_ms,
        }

    # Backtrace
    step_num += 1
    steps.append({
        "step_index": step_num,
        "phase": "found_destination",
        "current": destination,
        "wavefront": [],
        "explored": list(visited_order),
        "distance_map": dict(dist),
        "path": [],
        "description": (
            f"Destination {destination} reached! "
            f"Total cost = {dist[destination]:.3f}. Backtracing..."
        ),
    })

    path: list[tuple] = []
    node: tuple | None = destination
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()

    for i in range(len(path)):
        partial = path[:i+1]
        step_num += 1
        steps.append({
            "step_index": step_num,
            "phase": "backtrace",
            "current": path[i],
            "wavefront": [],
            "explored": list(visited_order),
            "distance_map": dict(dist),
            "path": list(partial),
            "description": (
                f"Backtracing {i+1}/{len(path)}: cell {path[i]} "
                f"(cost={dist.get(path[i], 0):.3f})."
            ),
        })

    step_num += 1
    steps.append({
        "step_index": step_num,
        "phase": "complete",
        "current": None,
        "wavefront": [],
        "explored": list(visited_order),
        "distance_map": dict(dist),
        "path": list(path),
        "description": (
            f"8-dir routing complete. Moves = {len(path)-1}, "
            f"total cost = {dist[destination]:.3f}. "
            f"Cells explored: {len(visited_order)}."
        ),
    })

    return {
        "found": True,
        "path": path,
        "distance_map": dist,
        "visited_order": visited_order,
        "steps": steps,
        "path_length": len(path) - 1,
        "total_cost": dist[destination],
        "cells_explored": len(visited_order),
        "runtime_ms": runtime_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Random maze generator
# ─────────────────────────────────────────────────────────────────────────────

def random_obstacles(
    rows: int,
    cols: int,
    source: tuple[int, int],
    destination: tuple[int, int],
    density: float = 0.25,
    seed: int = 42,
) -> set[tuple[int, int]]:
    """Generate random obstacle set with given density (0–1). Never blocks source or destination."""
    import random
    rng = random.Random(seed)
    obstacles: set[tuple] = set()
    protected = {source, destination}
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in protected and rng.random() < density:
                obstacles.add((r, c))
    return obstacles


def example_maze() -> tuple[int, int, tuple, tuple, set]:
    """Return (rows, cols, source, destination, obstacles) for the built-in example."""
    rows, cols = 15, 20
    source = (1, 1)
    destination = (13, 18)
    obstacles: set[tuple] = set()
    # Two vertical walls with gaps
    for r in range(2, 12):
        for c in (5, 6):
            if (r, c) not in {(5, 5), (8, 6)}:
                obstacles.add((r, c))
    for r in range(4, 13):
        for c in (11, 12):
            if (r, c) not in {(7, 11), (10, 12)}:
                obstacles.add((r, c))
    # Horizontal segment
    for c in range(7, 11):
        obstacles.add((8, c))
    return rows, cols, source, destination, obstacles
