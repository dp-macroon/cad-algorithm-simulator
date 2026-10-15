import sys
import os
import traceback
import math
from collections import deque as _deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

passed = 0
failed = 0
failures = []


def ok(name):
    global passed
    passed += 1
    print(f"  PASS  {name}")


def fail(name, reason):
    global failed
    failed += 1
    failures.append((name, reason))
    print(f"  FAIL  {name}: {reason}")


def run(name, fn):
    try:
        fn()
        ok(name)
    except AssertionError as e:
        fail(name, f"AssertionError: {e}")
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# compute_cut_cost
# ─────────────────────────────────────────────────────────────────────────────
print("\n[compute_cut_cost]")

def t_all_in_one_group():
    g1 = {"A", "B", "C"}; g2 = set()
    assert compute_cut_cost(g1, g2, [("A","B"),("B","C")]) == 0.0
run("all_in_one_group", t_all_in_one_group)

def t_perfect_cut():
    g1 = {"A","B"}; g2 = {"C","D"}
    assert compute_cut_cost(g1, g2, [("A","C"),("B","D"),("A","B")]) == 2.0
run("perfect_cut", t_perfect_cut)

def t_weighted_cut():
    g1 = {"A"}; g2 = {"B"}
    assert compute_cut_cost(g1, g2, [("A","B",5.0)], {("A","B"):5.0}) == 5.0
run("weighted_cut", t_weighted_cut)


# ─────────────────────────────────────────────────────────────────────────────
# validate_graph
# ─────────────────────────────────────────────────────────────────────────────
print("\n[validate_graph]")

def t_single_vertex():
    errs = validate_graph(["A"], [])
    assert any("at least 2" in e for e in errs)
run("single_vertex_error", t_single_vertex)

def t_isolated():
    errs = validate_graph(["A","B","C"], [("A","B")])
    assert any("Isolated" in e for e in errs)
run("isolated_vertex_warning", t_isolated)

def t_self_loop():
    errs = validate_graph(["A","B"], [("A","A")])
    assert any("Self-loop" in e for e in errs)
run("self_loop_warning", t_self_loop)

def t_unknown_vertex():
    errs = validate_graph(["A","B"], [("A","Z")])
    assert any("unknown vertex" in e for e in errs)
run("unknown_vertex_error", t_unknown_vertex)

def t_duplicate_edge():
    errs = validate_graph(["A","B"], [("A","B"),("B","A")])
    assert any("Duplicate" in e for e in errs)
run("duplicate_edge_warning", t_duplicate_edge)

def t_negative_weight():
    errs = validate_graph(["A","B"], [("A","B",-1)])
    assert any("non-positive" in e for e in errs)
run("negative_weight_error", t_negative_weight)

def t_valid_graph():
    errs = validate_graph(["A","B","C"], [("A","B"),("B","C")])
    assert errs == []
run("valid_graph_no_errors", t_valid_graph)


# ─────────────────────────────────────────────────────────────────────────────
# exact_balanced_min_cut
# ─────────────────────────────────────────────────────────────────────────────
print("\n[exact_balanced_min_cut]")

def t_known_optimal():
    # Path graph A-B-C-D: optimal balanced cut {A,B}|{C,D} or {A,C}|{B,D}.
    # Only 1 edge crosses ({B},{C} or similar), so cut_cost=1.
    # Use a graph where optimum is clearly 2:
    # Triangle-pair: {A-B-C fully connected} + {D-E-F fully connected} + bridge C-D.
    # Optimal split {A,B,C}|{D,E,F} = 1 crossing edge (C-D).
    # For balanced bipartition with n=6, target=3:
    v = ["A","B","C","D","E","F"]
    e = [("A","B"),("B","C"),("C","A"),("D","E"),("E","F"),("F","D"),("C","D")]
    r = exact_balanced_min_cut(v, e)
    assert r["cut_cost"] == 1.0, f"Expected 1.0, got {r['cut_cost']}"
    assert len(r["group1"]) == 3 and len(r["group2"]) == 3
run("known_optimal_cut", t_known_optimal)

def t_balance_preserved():
    v = ["A","B","C","D","E","F"]
    e = [("A","B"),("B","C"),("C","D"),("D","E"),("E","F")]
    r = exact_balanced_min_cut(v, e)
    assert len(r["group1"]) == 3 and len(r["group2"]) == 3
run("balance_preserved", t_balance_preserved)

def t_cut_edges_match_cost():
    v = ["A","B","C","D"]
    e = [("A","B"),("B","C"),("C","D"),("D","A"),("A","C")]
    r = exact_balanced_min_cut(v, e)
    computed = sum(ce["weight"] for ce in r["cut_edges"])
    assert abs(computed - r["cut_cost"]) < 1e-9
run("cut_edges_match_cost", t_cut_edges_match_cost)

def t_weighted_graph():
    v = ["A","B","C","D"]
    e = [("A","B",10.0),("C","D",10.0),("A","C",1.0),("B","D",1.0)]
    r = exact_balanced_min_cut(v, e)
    assert r["cut_cost"] == 2.0
run("weighted_graph", t_weighted_graph)

def t_disconnected():
    v = ["A","B","C","D"]
    e = [("A","B")]
    r = exact_balanced_min_cut(v, e)
    assert r["cut_cost"] == 0.0
run("disconnected_graph", t_disconnected)

def t_isolated_vertices():
    v = ["A","B","C","D"]; e = []
    r = exact_balanced_min_cut(v, e)
    assert r["cut_cost"] == 0.0
run("isolated_vertices", t_isolated_vertices)

def t_deterministic():
    v = ["A","B","C","D","E","F"]
    e = [("A","B"),("B","C"),("C","D"),("D","E"),("E","F"),("F","A"),("A","D")]
    r1 = exact_balanced_min_cut(v, e)
    r2 = exact_balanced_min_cut(v, e)
    assert r1["cut_cost"] == r2["cut_cost"]
run("deterministic_output", t_deterministic)

def t_too_few_vertices():
    try:
        exact_balanced_min_cut(["A"], [])
        assert False, "should have raised"
    except ValueError:
        pass
run("too_few_vertices_error", t_too_few_vertices)

def t_too_many_vertices():
    big = [str(i) for i in range(EXACT_SEARCH_LIMIT + 2)]
    try:
        exact_balanced_min_cut(big, [])
        assert False, "should have raised"
    except ValueError:
        pass
run("too_many_vertices_error", t_too_many_vertices)

def t_infeasible_constraint():
    try:
        exact_balanced_min_cut(["A","B"], [], balance_constraint=0.0)
        assert False, "should have raised"
    except ValueError:
        pass
run("infeasible_constraint_error", t_infeasible_constraint)

def t_steps_recorded():
    v = ["A","B","C","D"]
    e = [("A","B"),("B","C"),("C","D"),("D","A"),("A","C")]
    r = exact_balanced_min_cut(v, e)
    assert len(r["steps"]) > 0
    for s in r["steps"]:
        for k in ["step_index","group1","group2","cut_cost","description"]:
            assert k in s
run("steps_are_recorded", t_steps_recorded)

def t_partition_covers_all():
    v = ["A","B","C","D"]
    r = exact_balanced_min_cut(v, [("A","B"),("C","D")])
    assert set(r["group1"]) | set(r["group2"]) == set(v)
run("partition_covers_all_vertices", t_partition_covers_all)

def t_no_vertex_in_both():
    v = ["A","B","C","D"]
    r = exact_balanced_min_cut(v, [("A","B"),("C","D")])
    assert set(r["group1"]).isdisjoint(set(r["group2"]))
run("no_vertex_in_both_partitions", t_no_vertex_in_both)

def t_cut_edges_actually_crossing():
    v = ["A","B","C","D"]
    e = [("A","B"),("B","C"),("C","D"),("D","A")]
    r = exact_balanced_min_cut(v, e)
    g1 = set(r["group1"]); g2 = set(r["group2"])
    for ce in r["cut_edges"]:
        u, v2 = ce["u"], ce["v"]
        assert (u in g1) != (v2 in g1)
run("cut_edges_actually_crossing", t_cut_edges_actually_crossing)

def t_path_graph_optimal():
    v = ["A","B","C","D"]
    e = [("A","B"),("B","C"),("C","D")]
    r = exact_balanced_min_cut(v, e)
    assert r["cut_cost"] == 1.0
run("path_graph_known_optimal", t_path_graph_optimal)


# ─────────────────────────────────────────────────────────────────────────────
# kernighan_lin
# ─────────────────────────────────────────────────────────────────────────────
print("\n[kernighan_lin]")

def t_kl_valid_partition():
    v = ["A","B","C","D","E","F"]
    e = [("A","B"),("B","C"),("C","D"),("D","E"),("E","F"),("A","D")]
    r = kernighan_lin(v, e)
    assert set(r["group1"]) | set(r["group2"]) == set(v)
    assert set(r["group1"]).isdisjoint(set(r["group2"]))
    assert len(r["group1"]) == 3 and len(r["group2"]) == 3
run("kl_valid_partition", t_kl_valid_partition)

def t_kl_cost_matches_edges():
    v = ["A","B","C","D"]
    e = [("A","B"),("B","C"),("C","D"),("D","A")]
    r = kernighan_lin(v, e)
    g1 = set(r["group1"]); g2 = set(r["group2"])
    actual = sum(1 for u2,v2 in [(x[0],x[1]) for x in e] if (u2 in g1) != (v2 in g1))
    assert abs(actual - r["cut_cost"]) < 1e-9
run("kl_cost_matches_edges", t_kl_cost_matches_edges)

def t_kl_steps():
    v = ["A","B","C","D","E","F"]
    e = [("A","B"),("C","D"),("E","F"),("A","C"),("B","D")]
    r = kernighan_lin(v, e)
    assert isinstance(r["steps"], list)
run("kl_steps_recorded", t_kl_steps)

def t_kl_optimality_label():
    v = ["A","B","C","D"]; e = [("A","B"),("C","D")]
    r = kernighan_lin(v, e)
    assert "Heuristic" in r["optimality"] and "Kernighan" in r["optimality"]
run("kl_optimality_label", t_kl_optimality_label)

def t_kl_odd_error():
    try:
        kernighan_lin(["A","B","C"], [("A","B")])
        assert False
    except ValueError:
        pass
run("kl_odd_vertex_count_error", t_kl_odd_error)


# ─────────────────────────────────────────────────────────────────────────────
# validate_grid
# ─────────────────────────────────────────────────────────────────────────────
print("\n[validate_grid]")

def t_src_oob():
    errs = validate_grid(5,5,(10,10),(0,0),set())
    assert any("Source" in e for e in errs)
run("source_out_of_bounds", t_src_oob)

def t_dst_oob():
    errs = validate_grid(5,5,(0,0),(5,5),set())
    assert any("Destination" in e for e in errs)
run("destination_out_of_bounds", t_dst_oob)

def t_src_eq_dst():
    errs = validate_grid(5,5,(2,2),(2,2),set())
    assert any("same cell" in e for e in errs)
run("source_equals_destination", t_src_eq_dst)

def t_src_blocked():
    errs = validate_grid(5,5,(2,2),(4,4),{(2,2)})
    assert any("Source" in e and "blocked" in e for e in errs)
run("source_blocked", t_src_blocked)

def t_dst_blocked():
    errs = validate_grid(5,5,(0,0),(4,4),{(4,4)})
    assert any("Destination" in e and "blocked" in e for e in errs)
run("destination_blocked", t_dst_blocked)

def t_grid_too_small():
    errs = validate_grid(1,1,(0,0),(0,0),set())
    assert any("2" in e for e in errs)
run("grid_too_small", t_grid_too_small)

def t_grid_too_large():
    errs = validate_grid(MAX_ROWS+1,5,(0,0),(1,1),set())
    assert any("exceeds maximum" in e for e in errs)
run("grid_too_large", t_grid_too_large)

def t_valid_grid():
    errs = validate_grid(10,10,(0,0),(9,9),set())
    assert errs == []
run("valid_grid_no_errors", t_valid_grid)


# ─────────────────────────────────────────────────────────────────────────────
# lee_route_4dir
# ─────────────────────────────────────────────────────────────────────────────
print("\n[lee_route_4dir]")

def t_straight_line():
    r = lee_route_4dir(5,5,(2,0),(2,4),set())
    assert r["found"] and r["path_length"] == 4
    assert r["path"][0] == (2,0) and r["path"][-1] == (2,4)
run("straight_line_path", t_straight_line)

def t_avoids_obstacles():
    obs = {(2,1),(2,2),(2,3)}
    r = lee_route_4dir(5,5,(2,0),(2,4),obs)
    if r["found"]:
        for c in r["path"]: assert c not in obs
run("path_avoids_obstacles", t_avoids_obstacles)

def t_no_route():
    obs = {(0,1),(1,0),(1,2),(2,1)}
    r = lee_route_4dir(5,5,(0,0),(1,1),obs)
    assert not r["found"] and r["path"] == []
run("no_route_found", t_no_route)

def t_path_length_matches():
    r = lee_route_4dir(10,10,(0,0),(9,9),set())
    if r["found"]: assert r["path_length"] == len(r["path"]) - 1
run("path_length_matches_path", t_path_length_matches)

def t_path_connected():
    r = lee_route_4dir(10,10,(0,0),(9,9),set())
    if r["found"]:
        path = r["path"]
        for i in range(len(path)-1):
            r1,c1 = path[i]; r2,c2 = path[i+1]
            assert abs(r2-r1)+abs(c2-c1) == 1
run("path_is_connected", t_path_connected)

def t_starts_at_source():
    r = lee_route_4dir(5,5,(0,0),(4,4),set())
    assert r["path"][0] == (0,0)
run("path_starts_at_source", t_starts_at_source)

def t_ends_at_destination():
    r = lee_route_4dir(5,5,(0,0),(4,4),set())
    assert r["path"][-1] == (4,4)
run("path_ends_at_destination", t_ends_at_destination)

def t_boundary():
    r = lee_route_4dir(10,10,(0,0),(0,9),set())
    assert r["found"] and r["path_length"] == 9
run("boundary_cells", t_boundary)

def t_example_maze_4dir():
    rows,cols,src,dst,obs = example_maze()
    r = lee_route_4dir(rows,cols,src,dst,obs)
    assert r["found"]
    for cell in r["path"]: assert cell not in obs
run("example_maze_route_found", t_example_maze_4dir)

def t_rectangular_grid():
    r = lee_route_4dir(5,20,(2,0),(2,19),set())
    assert r["found"] and r["path_length"] == 19
run("rectangular_grid", t_rectangular_grid)

def t_explored_includes_path():
    r = lee_route_4dir(10,10,(0,0),(5,5),set())
    if r["found"]:
        visited = set(r["visited_order"])
        for cell in r["path"]: assert cell in visited
run("explored_includes_path", t_explored_includes_path)

def t_steps_keys():
    r = lee_route_4dir(5,5,(0,0),(4,4),set())
    for s in r["steps"]:
        for k in ["step_index","phase","description","path"]: assert k in s
run("steps_have_required_keys", t_steps_keys)

def t_cells_explored_count():
    r = lee_route_4dir(10,10,(0,0),(9,9),set())
    assert r["cells_explored"] == len(r["visited_order"])
run("cells_explored_count_matches", t_cells_explored_count)

def t_optimality_manhattan():
    src,dst = (0,0),(7,9)
    r = lee_route_4dir(15,15,src,dst,set())
    expected = abs(dst[0]-src[0]) + abs(dst[1]-src[1])
    assert r["path_length"] == expected
run("shortest_path_optimality_manhattan", t_optimality_manhattan)

def t_runtime_recorded():
    r = lee_route_4dir(10,10,(0,0),(9,9),set())
    assert r["runtime_ms"] >= 0
run("runtime_recorded", t_runtime_recorded)


# ─────────────────────────────────────────────────────────────────────────────
# lee_route_8dir
# ─────────────────────────────────────────────────────────────────────────────
print("\n[lee_route_8dir]")

def t_diagonal_path():
    r = lee_route_8dir(10,10,(0,0),(4,4),set())
    assert r["found"] and r["path_length"] == 4
    assert abs(r["total_cost"] - 4*math.sqrt(2)) < 1e-6
run("diagonal_path_found", t_diagonal_path)

def t_8dir_connected():
    r = lee_route_8dir(10,10,(0,0),(9,9),set())
    if r["found"]:
        path = r["path"]
        for i in range(len(path)-1):
            r1,c1 = path[i]; r2,c2 = path[i+1]
            assert abs(r2-r1)<=1 and abs(c2-c1)<=1
run("8dir_path_is_connected", t_8dir_connected)

def t_8dir_no_route():
    obs = {(0,1),(1,0),(1,2),(2,1),(1,1)}
    r = lee_route_8dir(5,5,(0,0),(2,2),obs)
    assert not r["found"]
run("8dir_no_route", t_8dir_no_route)

def t_8dir_avoids_obs():
    obs = {(1,1),(1,2),(2,1)}
    r = lee_route_8dir(5,5,(0,0),(3,3),obs)
    if r["found"]:
        for cell in r["path"]: assert cell not in obs
run("8dir_avoids_obstacles", t_8dir_avoids_obs)


# ─────────────────────────────────────────────────────────────────────────────
# random_obstacles
# ─────────────────────────────────────────────────────────────────────────────
print("\n[random_obstacles]")

def t_reproducible():
    o1 = random_obstacles(10,10,(0,0),(9,9),0.3,seed=123)
    o2 = random_obstacles(10,10,(0,0),(9,9),0.3,seed=123)
    assert o1 == o2
run("reproducible_same_seed", t_reproducible)

def t_diff_seeds_differ():
    o1 = random_obstacles(10,10,(0,0),(9,9),0.3,seed=1)
    o2 = random_obstacles(10,10,(0,0),(9,9),0.3,seed=2)
    assert o1 != o2
run("different_seeds_differ", t_diff_seeds_differ)

def t_no_block_src_dst():
    src,dst = (0,0),(9,9)
    for s in range(10):
        obs = random_obstacles(10,10,src,dst,0.9,seed=s)
        assert src not in obs and dst not in obs
run("src_dst_never_blocked", t_no_block_src_dst)


# ─────────────────────────────────────────────────────────────────────────────
# example_maze
# ─────────────────────────────────────────────────────────────────────────────
print("\n[example_maze]")

def t_example_valid():
    rows,cols,src,dst,obs = example_maze()
    errs = validate_grid(rows,cols,src,dst,obs)
    assert errs == [], f"Validation errors: {errs}"
run("example_maze_valid_config", t_example_valid)

def t_example_has_route():
    rows,cols,src,dst,obs = example_maze()
    r = lee_route_4dir(rows,cols,src,dst,obs)
    assert r["found"]
run("example_maze_has_route", t_example_has_route)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-verification
# ─────────────────────────────────────────────────────────────────────────────
print("\n[cross_verification]")

def t_vs_reference_bfs():
    def ref_bfs(rows,cols,src,dst,obs):
        q = _deque([src]); dist = {src:0}
        while q:
            cur = q.popleft()
            if cur == dst: return dist[cur]
            r2,c2 = cur
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nxt = (r2+dr,c2+dc)
                if 0<=nxt[0]<rows and 0<=nxt[1]<cols and nxt not in obs and nxt not in dist:
                    dist[nxt]=dist[cur]+1; q.append(nxt)
        return None
    rows,cols,src,dst,obs = example_maze()
    ref = ref_bfs(rows,cols,src,dst,obs)
    our = lee_route_4dir(rows,cols,src,dst,obs)["path_length"]
    assert ref == our, f"ref={ref}, ours={our}"
run("4dir_optimal_vs_reference_bfs", t_vs_reference_bfs)

def t_mincut_path_graph_manual():
    v=["A","B","C","D"]; e=[("A","B"),("B","C"),("C","D")]
    r=exact_balanced_min_cut(v,e)
    assert r["cut_cost"]==1.0
    g1=set(r["group1"]); g2=set(r["group2"])
    for ce in r["cut_edges"]:
        assert (ce["u"] in g1) != (ce["v"] in g1)
run("mincut_path_graph_manual", t_mincut_path_graph_manual)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests")
if failures:
    print("\nFAILURES:")
    for name, reason in failures:
        print(f"  FAIL {name}")
        print(f"    {reason}")
else:
    print("All tests passed!")
sys.exit(0 if failed == 0 else 1)
