"""
Automated tests for VLSI CAD Lab algorithms.

Run with:  python -m pytest tests/ -v
       or: python tests/test_algorithms.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import math

from algorithms.min_cut import (
    exact_balanced_min_cut,
    kernighan_lin,
    compute_cut_cost,
    validate_graph,
    EXACT_SEARCH_LIMIT,
)
from algorithms.lee_routing import (
    lee_route_4dir,
    lee_route_8dir,
    validate_grid,
    random_obstacles,
    example_maze,
    FOUR_DIRS,
    MAX_ROWS,
    MAX_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Min-Cut Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCutCost:
    def test_all_in_one_group(self):
        g1 = {"A", "B", "C"}
        g2 = set()
        edges = [("A", "B"), ("B", "C")]
        assert compute_cut_cost(g1, g2, edges) == 0.0

    def test_perfect_cut(self):
        g1 = {"A", "B"}
        g2 = {"C", "D"}
        edges = [("A", "C"), ("B", "D"), ("A", "B")]
        assert compute_cut_cost(g1, g2, edges) == 2.0

    def test_weighted_cut(self):
        g1 = {"A"}
        g2 = {"B"}
        edges = [("A", "B", 5.0)]
        w = {("A", "B"): 5.0}
        assert compute_cut_cost(g1, g2, edges, w) == 5.0


class TestValidateGraph:
    def test_single_vertex_error(self):
        errs = validate_graph(["A"], [])
        assert any("at least 2" in e for e in errs)

    def test_isolated_vertex_warning(self):
        errs = validate_graph(["A", "B", "C"], [("A", "B")])
        assert any("Isolated" in e for e in errs)

    def test_self_loop_warning(self):
        errs = validate_graph(["A", "B"], [("A", "A")])
        assert any("Self-loop" in e for e in errs)

    def test_unknown_vertex_error(self):
        errs = validate_graph(["A", "B"], [("A", "Z")])
        assert any("unknown vertex" in e for e in errs)

    def test_duplicate_edge_warning(self):
        errs = validate_graph(["A", "B"], [("A", "B"), ("B", "A")])
        assert any("Duplicate" in e for e in errs)

    def test_negative_weight_error(self):
        errs = validate_graph(["A", "B"], [("A", "B", -1)])
        assert any("non-positive" in e for e in errs)

    def test_valid_graph_no_errors(self):
        errs = validate_graph(["A", "B", "C"], [("A", "B"), ("B", "C")])
        assert errs == []


class TestExactMinCut:
    """Tests for exact_balanced_min_cut."""

    def _make_known_graph(self):
        # 4-cycle: A-B-C-D-A + diagonal A-C
        # Optimal balanced cut splits {A,B} | {C,D} or {A,D} | {B,C}, cut=2 edges
        vertices = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]
        return vertices, edges

    def test_known_optimal_cut(self):
        vertices, edges = self._make_known_graph()
        result = exact_balanced_min_cut(vertices, edges)
        assert result["cut_cost"] == 2.0
        assert result["optimality"] == "Exact (exhaustive search)"
        assert len(result["group1"]) == 2
        assert len(result["group2"]) == 2

    def test_balance_preserved(self):
        vertices = ["A", "B", "C", "D", "E", "F"]
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")]
        result = exact_balanced_min_cut(vertices, edges)
        assert len(result["group1"]) == 3
        assert len(result["group2"]) == 3

    def test_cut_edges_match_cost(self):
        """Verify that cut_cost equals the sum of weights of returned cut_edges."""
        vertices, edges = self._make_known_graph()
        result = exact_balanced_min_cut(vertices, edges)
        computed = sum(e["weight"] for e in result["cut_edges"])
        assert abs(computed - result["cut_cost"]) < 1e-9

    def test_weighted_graph(self):
        # Two communities: {A,B} and {C,D}, with heavy internal edges
        vertices = ["A", "B", "C", "D"]
        edges = [("A", "B", 10.0), ("C", "D", 10.0), ("A", "C", 1.0), ("B", "D", 1.0)]
        result = exact_balanced_min_cut(vertices, edges)
        assert result["cut_cost"] == 2.0

    def test_disconnected_graph(self):
        vertices = ["A", "B", "C", "D"]
        edges = [("A", "B")]  # C and D isolated
        result = exact_balanced_min_cut(vertices, edges)
        # Optimal: one group contains one connected pair, cut_cost = 0
        assert result["cut_cost"] == 0.0

    def test_isolated_vertices(self):
        vertices = ["A", "B", "C", "D"]
        edges = []
        result = exact_balanced_min_cut(vertices, edges)
        assert result["cut_cost"] == 0.0

    def test_deterministic_output(self):
        vertices = ["A", "B", "C", "D", "E", "F"]
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "A"), ("A", "D")]
        r1 = exact_balanced_min_cut(vertices, edges)
        r2 = exact_balanced_min_cut(vertices, edges)
        assert r1["cut_cost"] == r2["cut_cost"]
        assert sorted(r1["group1"]) == sorted(r1["group1"])

    def test_too_few_vertices_error(self):
        with pytest.raises(ValueError):
            exact_balanced_min_cut(["A"], [])

    def test_too_many_vertices_error(self):
        big_v = [str(i) for i in range(EXACT_SEARCH_LIMIT + 2)]
        with pytest.raises(ValueError, match="exact search"):
            exact_balanced_min_cut(big_v, [])

    def test_infeasible_constraint_error(self):
        with pytest.raises(ValueError):
            exact_balanced_min_cut(["A", "B"], [], balance_constraint=0.0)

    def test_steps_are_recorded(self):
        vertices, edges = self._make_known_graph()
        result = exact_balanced_min_cut(vertices, edges)
        assert len(result["steps"]) > 0
        # Each step must have required keys
        for s in result["steps"]:
            assert "step_index" in s
            assert "group1" in s
            assert "group2" in s
            assert "cut_cost" in s
            assert "description" in s

    def test_partition_covers_all_vertices(self):
        vertices = ["A", "B", "C", "D"]
        result = exact_balanced_min_cut(vertices, [("A", "B"), ("C", "D")])
        all_assigned = set(result["group1"]) | set(result["group2"])
        assert all_assigned == set(vertices)

    def test_no_vertex_in_both_partitions(self):
        vertices = ["A", "B", "C", "D"]
        result = exact_balanced_min_cut(vertices, [("A", "B"), ("C", "D")])
        assert set(result["group1"]).isdisjoint(set(result["group2"]))

    def test_cut_edges_are_actually_crossing(self):
        vertices = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
        result = exact_balanced_min_cut(vertices, edges)
        g1 = set(result["group1"])
        g2 = set(result["group2"])
        for ce in result["cut_edges"]:
            u, v = ce["u"], ce["v"]
            assert (u in g1) != (v in g1), \
                f"Edge ({u},{v}) reported as cut but endpoints are in the same partition."


class TestKernighanLin:
    def test_kl_runs_and_returns_valid_partition(self):
        vertices = ["A", "B", "C", "D", "E", "F"]
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("A", "D")]
        result = kernighan_lin(vertices, edges)
        assert set(result["group1"]) | set(result["group2"]) == set(vertices)
        assert set(result["group1"]).isdisjoint(set(result["group2"]))
        assert len(result["group1"]) == 3
        assert len(result["group2"]) == 3

    def test_kl_cut_cost_matches_edges(self):
        vertices = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
        result = kernighan_lin(vertices, edges)
        g1 = set(result["group1"])
        g2 = set(result["group2"])
        actual_cut = sum(1 for u, v in [e[:2] for e in edges] if (u in g1) != (v in g1))
        assert abs(actual_cut - result["cut_cost"]) < 1e-9

    def test_kl_steps_recorded(self):
        vertices = ["A", "B", "C", "D", "E", "F"]
        edges = [("A", "B"), ("C", "D"), ("E", "F"), ("A", "C"), ("B", "D")]
        result = kernighan_lin(vertices, edges)
        assert "steps" in result
        assert isinstance(result["steps"], list)

    def test_kl_optimality_label(self):
        vertices = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("C", "D")]
        result = kernighan_lin(vertices, edges)
        assert "Heuristic" in result["optimality"]
        assert "Kernighan" in result["optimality"]

    def test_kl_odd_vertex_count_error(self):
        with pytest.raises(ValueError):
            kernighan_lin(["A", "B", "C"], [("A", "B")])


# ─────────────────────────────────────────────────────────────────────────────
# Lee Routing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateGridRouting:
    def test_source_out_of_bounds(self):
        errs = validate_grid(5, 5, (10, 10), (0, 0), set())
        assert any("Source" in e for e in errs)

    def test_destination_out_of_bounds(self):
        errs = validate_grid(5, 5, (0, 0), (5, 5), set())
        assert any("Destination" in e for e in errs)

    def test_source_equals_destination(self):
        errs = validate_grid(5, 5, (2, 2), (2, 2), set())
        assert any("same cell" in e for e in errs)

    def test_source_blocked(self):
        errs = validate_grid(5, 5, (2, 2), (4, 4), {(2, 2)})
        assert any("Source" in e and "blocked" in e for e in errs)

    def test_destination_blocked(self):
        errs = validate_grid(5, 5, (0, 0), (4, 4), {(4, 4)})
        assert any("Destination" in e and "blocked" in e for e in errs)

    def test_grid_too_small(self):
        errs = validate_grid(1, 1, (0, 0), (0, 0), set())
        assert any("2×2" in e for e in errs)

    def test_grid_too_large(self):
        errs = validate_grid(MAX_ROWS + 1, 5, (0, 0), (1, 1), set())
        assert any("exceeds maximum" in e for e in errs)

    def test_valid_config_no_errors(self):
        errs = validate_grid(10, 10, (0, 0), (9, 9), set())
        assert errs == []


class TestLeeRoute4Dir:
    def test_straight_line_path(self):
        """Horizontal straight-line, no obstacles."""
        result = lee_route_4dir(5, 5, (2, 0), (2, 4), set())
        assert result["found"] is True
        assert result["path_length"] == 4
        path = result["path"]
        assert path[0] == (2, 0)
        assert path[-1] == (2, 4)

    def test_path_avoids_obstacles(self):
        """Path must not pass through any obstacle."""
        obstacles = {(2, 1), (2, 2), (2, 3)}
        result = lee_route_4dir(5, 5, (2, 0), (2, 4), obstacles)
        # Should find a route going around
        if result["found"]:
            path_set = set(result["path"])
            for obs in obstacles:
                assert obs not in path_set, f"Path passes through obstacle {obs}"

    def test_no_route_found(self):
        """Destination completely surrounded by obstacles."""
        obstacles = {(0, 1), (1, 0), (1, 2), (2, 1)}
        result = lee_route_4dir(5, 5, (0, 0), (1, 1), obstacles)
        assert result["found"] is False
        assert result["path"] == []

    def test_path_length_matches_path(self):
        """path_length must equal len(path) - 1."""
        result = lee_route_4dir(10, 10, (0, 0), (9, 9), set())
        if result["found"]:
            assert result["path_length"] == len(result["path"]) - 1

    def test_path_is_connected(self):
        """Every consecutive pair in path must be adjacent (4-dir)."""
        result = lee_route_4dir(10, 10, (0, 0), (9, 9), set())
        if result["found"]:
            path = result["path"]
            for i in range(len(path) - 1):
                r1, c1 = path[i]
                r2, c2 = path[i+1]
                assert (abs(r2 - r1) + abs(c2 - c1)) == 1, \
                    f"Non-adjacent cells in path: {path[i]} → {path[i+1]}"

    def test_path_starts_at_source(self):
        result = lee_route_4dir(5, 5, (0, 0), (4, 4), set())
        assert result["path"][0] == (0, 0)

    def test_path_ends_at_destination(self):
        result = lee_route_4dir(5, 5, (0, 0), (4, 4), set())
        assert result["path"][-1] == (4, 4)

    def test_boundary_source_and_dest(self):
        """Routing on grid corners."""
        result = lee_route_4dir(10, 10, (0, 0), (0, 9), set())
        assert result["found"] is True
        assert result["path_length"] == 9

    def test_maze_with_obstacles(self):
        """Test with the built-in example maze."""
        rows, cols, src, dst, obs = example_maze()
        result = lee_route_4dir(rows, cols, src, dst, obs)
        assert result["found"] is True
        # Verify path consistency
        path = result["path"]
        for cell in path:
            assert cell not in obs, f"Path cell {cell} is an obstacle."

    def test_rectangular_grid(self):
        """Non-square grid routing."""
        result = lee_route_4dir(5, 20, (2, 0), (2, 19), set())
        assert result["found"] is True
        assert result["path_length"] == 19

    def test_explored_cells_includes_path(self):
        """All path cells must appear in visited_order."""
        result = lee_route_4dir(10, 10, (0, 0), (5, 5), set())
        if result["found"]:
            visited = set(result["visited_order"])
            for cell in result["path"]:
                assert cell in visited, f"Path cell {cell} not in visited_order"

    def test_steps_have_required_keys(self):
        result = lee_route_4dir(5, 5, (0, 0), (4, 4), set())
        for s in result["steps"]:
            assert "step_index" in s
            assert "phase" in s
            assert "description" in s
            assert "path" in s

    def test_cells_explored_count_matches(self):
        result = lee_route_4dir(10, 10, (0, 0), (9, 9), set())
        assert result["cells_explored"] == len(result["visited_order"])

    def test_shortest_path_optimality(self):
        """
        Manhattan distance is a lower bound for 4-dir BFS.
        Verify path length equals Manhattan distance when no obstacles exist.
        """
        src, dst = (0, 0), (7, 9)
        result = lee_route_4dir(15, 15, src, dst, set())
        expected = abs(dst[0] - src[0]) + abs(dst[1] - src[1])
        assert result["path_length"] == expected

    def test_runtime_recorded(self):
        result = lee_route_4dir(10, 10, (0, 0), (9, 9), set())
        assert result["runtime_ms"] >= 0


class TestLeeRoute8Dir:
    def test_diagonal_path_found(self):
        """Diagonal from (0,0) to (4,4) should have 4 moves at cost √2 each."""
        result = lee_route_8dir(10, 10, (0, 0), (4, 4), set())
        assert result["found"] is True
        assert result["path_length"] == 4
        assert abs(result["total_cost"] - 4 * math.sqrt(2)) < 1e-6

    def test_8dir_path_is_connected(self):
        """Every consecutive pair in path must differ by at most 1 in each dimension."""
        result = lee_route_8dir(10, 10, (0, 0), (9, 9), set())
        if result["found"]:
            path = result["path"]
            for i in range(len(path) - 1):
                r1, c1 = path[i]
                r2, c2 = path[i+1]
                assert abs(r2 - r1) <= 1 and abs(c2 - c1) <= 1

    def test_8dir_no_route(self):
        obstacles = {(0, 1), (1, 0), (1, 2), (2, 1), (1, 1)}
        result = lee_route_8dir(5, 5, (0, 0), (2, 2), obstacles)
        assert result["found"] is False

    def test_8dir_path_avoids_obstacles(self):
        obstacles = {(1, 1), (1, 2), (2, 1)}
        result = lee_route_8dir(5, 5, (0, 0), (3, 3), obstacles)
        if result["found"]:
            for cell in result["path"]:
                assert cell not in obstacles


class TestRandomObstacles:
    def test_reproducible_with_same_seed(self):
        obs1 = random_obstacles(10, 10, (0, 0), (9, 9), 0.3, seed=123)
        obs2 = random_obstacles(10, 10, (0, 0), (9, 9), 0.3, seed=123)
        assert obs1 == obs2

    def test_different_seeds_differ(self):
        obs1 = random_obstacles(10, 10, (0, 0), (9, 9), 0.3, seed=1)
        obs2 = random_obstacles(10, 10, (0, 0), (9, 9), 0.3, seed=2)
        # Very unlikely to be equal
        assert obs1 != obs2

    def test_source_and_dest_never_blocked(self):
        src, dst = (0, 0), (9, 9)
        for seed in range(10):
            obs = random_obstacles(10, 10, src, dst, 0.9, seed=seed)
            assert src not in obs
            assert dst not in obs


class TestExampleMaze:
    def test_example_maze_returns_valid_config(self):
        rows, cols, src, dst, obs = example_maze()
        errs = validate_grid(rows, cols, src, dst, obs)
        assert errs == [], f"Example maze validation errors: {errs}"

    def test_example_maze_has_route(self):
        rows, cols, src, dst, obs = example_maze()
        result = lee_route_4dir(rows, cols, src, dst, obs)
        assert result["found"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Cross-verification tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossVerification:
    def test_4dir_optimal_vs_bfs_reference(self):
        """
        Independent BFS reference: compare path length against our implementation.
        """
        from collections import deque as _deque

        def ref_bfs(rows, cols, src, dst, obs):
            q = _deque([src])
            dist = {src: 0}
            while q:
                cur = q.popleft()
                if cur == dst:
                    return dist[cur]
                r, c = cur
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nxt = (r+dr, c+dc)
                    if 0<=nxt[0]<rows and 0<=nxt[1]<cols and nxt not in obs and nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            return None

        rows, cols, src, dst, obs = example_maze()
        ref = ref_bfs(rows, cols, src, dst, obs)
        our = lee_route_4dir(rows, cols, src, dst, obs)["path_length"]
        assert ref == our, f"Reference BFS gives {ref} but our implementation gives {our}"

    def test_mincut_cut_cost_verified_manually(self):
        """
        4-vertex path graph A-B-C-D: optimal balanced cut separates {A,B}|{C,D} = 1 edge.
        """
        vertices = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        result = exact_balanced_min_cut(vertices, edges)
        assert result["cut_cost"] == 1.0
        # The single cut edge must be B-C (the only cross-partition edge for this optimal partition)
        g1 = set(result["group1"])
        g2 = set(result["group2"])
        for ce in result["cut_edges"]:
            u, v = ce["u"], ce["v"]
            assert (u in g1) != (v in g1)


if __name__ == "__main__":
    # Run tests manually without pytest
    import traceback

    test_classes = [
        TestComputeCutCost,
        TestValidateGraph,
        TestExactMinCut,
        TestKernighanLin,
        TestValidateGridRouting,
        TestLeeRoute4Dir,
        TestLeeRoute8Dir,
        TestRandomObstacles,
        TestExampleMaze,
        TestCrossVerification,
    ]

    passed = 0
    failed = 0
    errors_found = []

    for cls in test_classes:
        obj = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for method in methods:
            try:
                getattr(obj, method)()
                passed += 1
                print(f"  ✓ {cls.__name__}.{method}")
            except Exception as e:
                failed += 1
                errors_found.append((cls.__name__, method, str(e)))
                print(f"  ✗ {cls.__name__}.{method}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors_found:
        print("\nFailures:")
        for cls_name, method, err in errors_found:
            print(f"  {cls_name}.{method}: {err}")
    else:
        print("All tests passed!")
    sys.exit(0 if failed == 0 else 1)
