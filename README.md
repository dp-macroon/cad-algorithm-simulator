# VLSI CAD Lab — Professional Edition

**Interactive Placement & Routing Algorithm Explorer**

A professional-grade educational tool for demonstrating and exploring VLSI placement and routing algorithms with step-by-step visualisation, editable graphs and grids, and real algorithm execution.

---

## Features

### Module A — Min-Cut Graph Partitioning
- **Editable graph**: add/remove vertices and edges, edit weights, load example, generate random graph
- **Exact balanced search**: exhaustive bipartition, guaranteed optimal for ≤ 20 vertices
- **Kernighan–Lin heuristic**: near-optimal for larger graphs, with swap-step logging
- **Step-by-step playback**: navigate every candidate evaluated with Previous/Next/Play/Restart
- **Metrics**: cut cost, partition balance, runtime, candidates examined, optimality guarantee
- **Export**: JSON and CSV download of results
- **Pseudocode panel** and **Glossary**

### Module B — Lee's Maze Routing
- **Editable grid**: configurable rows/columns, source, destination, obstacle placement/removal
- **4-directional Lee BFS**: guarantees shortest path (unit cost)
- **8-directional Dijkstra**: diagonal moves at cost √2, correct optimal routing
- **Step-by-step playback**: wave expansion, individual cell exploration, backtracing
- **Metrics**: path length (moves), cells explored, runtime, route status
- **Export**: JSON with path, obstacles, and metrics

---

## Technology Stack

| Component | Library |
|---|---|
| UI / Browser app | Streamlit ≥ 1.35 |
| Interactive graphs | Plotly ≥ 5.18 |
| Graph layout | NetworkX ≥ 3.2 |
| Algorithms | Pure Python (stdlib only) |
| Tests | pytest ≥ 7.4 |

---

## Installation

```bash
# 1. Navigate to the project directory
cd vlsi_cad_lab

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The application opens automatically in your browser at `http://localhost:8501`.

---

## Running Tests

```bash
# Run with pytest (recommended)
pytest tests/ -v

# Or run directly
python tests/test_algorithms.py
```

Expected output: **all tests passing** with a summary of passed/failed counts.

---

## Project Structure

```
vlsi_cad_lab/
├── app.py                        # Streamlit application entry point
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── algorithms/
│   ├── __init__.py
│   ├── min_cut.py                # Min-cut: exact search + KL heuristic
│   └── lee_routing.py           # Lee BFS + Dijkstra 8-dir routing
│
└── tests/
    └── test_algorithms.py        # Automated test suite (pytest)
```

---

## Algorithm Overview

### Min-Cut Graph Partitioning

**Objective**: Given graph G = (V, E), divide V into two balanced groups A and B such that the sum of weights of edges between A and B (the cut cost) is minimised.

**Exact Search**
- Fixes one vertex in group A to eliminate mirrored duplicates.
- Enumerates all C(n−1, ⌊n/2⌋−1) balanced bipartitions.
- Evaluates cut cost for each.
- **Complexity**: O(C(n,n/2) · |E|) — exponential in n.
- **Optimality**: guaranteed global minimum.
- **Limit**: ≤ 20 vertices (configurable via `EXACT_SEARCH_LIMIT`).

**Kernighan–Lin Heuristic (KL)**
- Starts with a random balanced partition.
- In each pass, tentatively swaps pairs of vertices to maximise cumulative gain.
- Applies the best prefix of swaps; repeats until no improvement.
- **Complexity**: O(|V|² · passes) per iteration.
- **Optimality**: near-optimal heuristic — not guaranteed global minimum.

### Lee's Maze Routing

**Objective**: Find the shortest path from source S to destination D on a grid, respecting obstacle cells.

**Lee BFS (4-directional)**
- Initialises a BFS queue at the source with distance 0.
- Expands outward, assigning distance labels to newly reachable cells.
- Stops when the destination is reached.
- Backtraces via parent pointers to reconstruct the shortest path.
- **Complexity**: O(R × C).
- **Optimality**: shortest path guaranteed (unit costs).

**Dijkstra (8-directional)**
- Uses a min-heap priority queue.
- Cardinal moves cost 1.0; diagonal moves cost √2.
- Settles each cell at minimum cost.
- BFS is not used here because diagonal costs differ — it would not give the true optimum.
- **Complexity**: O(R·C·log(R·C)).

---

## Demonstration Guide (for Faculty Evaluators)

### Module A — Min-Cut

1. Launch the app. The example 8-vertex circuit netlist loads automatically.
2. Click **▶ Run Partitioning** to execute exact balanced min-cut.
3. Observe the graph coloured by partition (indigo = A, teal = B), with cut edges shown in red.
4. Inspect the metrics panel: cut cost, balance ratio, runtime, candidates examined.
5. Use **▶ Next / ◀ Prev** to step through every candidate bipartition evaluated.
6. Each step shows: candidate assignment, crossing edges, cut cost, best cost, improvement flag, and a plain-English explanation.
7. Change the balance slider to 0.4 and rerun. Observe how the optimal partition changes.
8. Use **🎲 Random Graph** to generate a fresh graph and rerun.
9. Add a vertex and edges using the **Graph Editor** panel, then rerun.
10. Switch to **Kernighan–Lin Heuristic** for a larger graph (add vertices beyond 20).

### Module B — Lee Routing

1. Switch to the **Lee's Maze Routing** tab. The built-in maze loads automatically.
2. Click **▶ Run Routing**.
3. Observe the BFS wave expansion (wavefront in teal, explored in dark blue, path in amber).
4. Step through with **▶ Next / ◀ Prev**: see wave expansion, individual cell labels, destination discovery, and backtracing.
5. Check **Show distance labels on grid** to see BFS distance values on each cell.
6. Add obstacles manually or click **🎲 Random Obs.**
7. Move the source/destination using the coordinate inputs and rerun.
8. Block all paths and observe the "No route found" message.
9. Switch to **8-directional (Dijkstra)** mode. Observe shorter diagonal paths and cost labels.
10. Resize the grid to 20×30 and test routing on a larger area.

---

## Performance Notes

| Scenario | Expected Behaviour |
|---|---|
| Min-cut, n ≤ 20 | Exact search, completes in < 1 second |
| Min-cut, n > 20 | KL heuristic activated automatically |
| Grid ≤ 50×80 | Full step recording, responsive playback |
| Large grids | Step recording may be skipped for speed (see `MAX_CELLS_FOR_STEP_RECORD`) |

---

## Limitations

- The graph visualiser uses NetworkX spring layout. Dragging nodes is not yet supported in Plotly — node positions recalculate on each render.
- KL heuristic uses a fixed random seed (42) for reproducibility. Change `seed_partition` argument for different starting points.
- Auto-play uses `time.sleep` in a single Streamlit thread — for true async playback, a custom component or WebSocket would be needed.
- Exact search is limited to 20 vertices to avoid exponential hang times in a browser session.

---

## Methodology (for CAD Assignment Report)

**Objective**: Implement and visualise two fundamental VLSI CAD algorithms — balanced graph partitioning (placement) and Lee maze routing (routing) — in an interactive educational tool.

**Algorithm Description**: See "Algorithm Overview" section above.

**Implementation**: Python algorithms separated from Streamlit UI. Session state preserves user edits across interactions. Plotly provides interactive graph and grid visualisations.

**Simulation Procedure**: Users configure inputs via UI controls, run the algorithm, inspect results via metrics, and step through execution states via the playback panel.

**Performance Metrics**: Cut cost, runtime, candidates examined (min-cut); path length, cells explored, runtime (routing).

**Results**: Both algorithms produce correct, verifiable results confirmed by automated tests including cross-verification against independent reference implementations.

**Limitations**: Exact min-cut is exponential; step recording uses O(R·C·steps) memory.

**Conclusion**: The simulator successfully demonstrates both algorithms with correct step-by-step visualisation, editable inputs, and professional-grade UI design suitable for educational and demonstration purposes.

---

## License

Educational use. Original Tkinter code adapted and significantly upgraded.
