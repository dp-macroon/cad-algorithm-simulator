"""
Min-Cut Graph Partitioning — Algorithm Module
==============================================

Provides:
  - exact_balanced_min_cut  : Exhaustive balanced bipartition (optimal, exponential time).
  - kernighan_lin            : Kernighan–Lin heuristic (polynomial, near-optimal).
  - compute_cut_cost         : Shared cost helper.
  - record_exact_steps       : Returns a list of step records for step-by-step playback.
  - record_kl_steps          : Returns a list of step records for KL playback.

All functions are pure (no side effects) and fully deterministic given the same seed.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

EXACT_SEARCH_LIMIT = 20   # vertices — above this, exact search is too slow


def _edge_set(edges: list[tuple]) -> list[tuple]:
    """Normalise edges to (min, max) tuples so direction doesn't matter."""
    return [(min(u, v), max(u, v)) for u, v in edges]


def compute_cut_cost(
    group1: set,
    group2: set,
    edges: list[tuple],
    weights: dict[tuple, float] | None = None,
) -> float:
    """
    Return the total cost of edges whose endpoints span the two partitions.

    For unweighted graphs, cost = number of crossing edges.
    For weighted graphs, cost = sum of weights of crossing edges.
    """
    total = 0.0
    for u, v, *_ in edges:
        in1_u = u in group1
        in1_v = v in group1
        if in1_u != in1_v:
            if weights:
                key = (min(u, v), max(u, v))
                total += weights.get(key, 1.0)
            else:
                total += 1.0
    return total


def _cut_edges_list(
    group1: set,
    group2: set,
    edges: list[tuple],
    weights: dict | None = None,
) -> list[dict]:
    """Return detailed list of crossing edge dicts."""
    result = []
    for edge in edges:
        u, v = edge[0], edge[1]
        w = edge[2] if len(edge) > 2 else (weights.get((min(u, v), max(u, v)), 1.0) if weights else 1.0)
        if (u in group1) != (v in group1):
            result.append({"u": u, "v": v, "weight": w})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Exact balanced bipartition search
# ─────────────────────────────────────────────────────────────────────────────

def exact_balanced_min_cut(
    vertices: list,
    edges: list[tuple],
    weights: dict | None = None,
    balance_constraint: float = 0.5,
) -> dict[str, Any]:
    """
    Exact balanced min-cut by exhaustive search.

    Parameters
    ----------
    vertices           : list of vertex names (any hashable).
    edges              : list of (u, v) or (u, v, weight) tuples.
    weights            : optional dict mapping (min_v, max_v) → float weight.
    balance_constraint : target fraction of vertices in group 1 (default 0.5 = equal).

    Returns
    -------
    dict with keys:
      group1, group2, cut_edges, cut_cost, runtime_ms,
      candidates_examined, optimality, steps (list of step dicts)
    """
    n = len(vertices)
    if n < 2:
        raise ValueError("Need at least 2 vertices for partitioning.")
    if n > EXACT_SEARCH_LIMIT:
        raise ValueError(
            f"Exact search supports up to {EXACT_SEARCH_LIMIT} vertices. "
            f"Use the Kernighan–Lin heuristic for {n} vertices."
        )

    if balance_constraint <= 0.0 or balance_constraint >= 1.0:
        raise ValueError(
            f"Balance constraint must be strictly between 0 and 1 (got {balance_constraint})."
        )
    target1 = round(n * balance_constraint)
    target2 = n - target1
    if target1 < 1 or target2 < 1:
        raise ValueError(
            f"Balance constraint {balance_constraint} produces an empty partition "
            f"(target sizes {target1}:{target2} for {n} vertices)."
        )

    # Normalise edge weight lookup
    if weights is None:
        weights = {}
    norm_weights: dict[tuple, float] = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        w = edge[2] if len(edge) > 2 else weights.get((min(u, v), max(u, v)), 1.0)
        norm_weights[(min(u, v), max(u, v))] = float(w)

    steps: list[dict] = []
    best_cost: float | None = None
    best_g1: set | None = None
    best_g2: set | None = None
    candidates_examined = 0

    # Fix one vertex to avoid mirrored duplicates
    anchor = vertices[0]
    remaining = vertices[1:]

    t0 = time.perf_counter()

    for chosen in combinations(remaining, target1 - 1):
        g1 = set((anchor,) + chosen)
        g2 = set(vertices) - g1
        candidates_examined += 1

        cut_list = _cut_edges_list(g1, g2, edges, norm_weights)
        cost = sum(e["weight"] for e in cut_list)
        is_best = best_cost is None or cost < best_cost

        if is_best:
            best_cost = cost
            best_g1 = set(g1)
            best_g2 = set(g2)

        steps.append({
            "step_index": candidates_examined,
            "group1": sorted(g1),
            "group2": sorted(g2),
            "cut_edges": cut_list,
            "cut_cost": cost,
            "best_cost": best_cost,
            "is_improvement": is_best,
            "description": (
                f"Candidate #{candidates_examined}: "
                f"A={sorted(g1)}, B={sorted(g2)}. "
                f"Cut cost = {cost:.2f}. "
                + (f"✓ New best!" if is_best else f"Not better than best ({best_cost:.2f}).")
            ),
        })

    runtime_ms = (time.perf_counter() - t0) * 1000

    final_cut = _cut_edges_list(best_g1, best_g2, edges, norm_weights)

    return {
        "group1": sorted(best_g1),
        "group2": sorted(best_g2),
        "cut_edges": final_cut,
        "cut_cost": best_cost,
        "runtime_ms": runtime_ms,
        "candidates_examined": candidates_examined,
        "optimality": "Exact (exhaustive search)",
        "steps": steps,
        "balance": f"{len(best_g1)}:{len(best_g2)}",
        "balance_ratio": len(best_g1) / len(best_g2) if best_g2 else float("inf"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Kernighan–Lin heuristic
# ─────────────────────────────────────────────────────────────────────────────

def _kl_gain(a: str, b: str, g1: set, g2: set, adj: dict, w: dict) -> float:
    """
    Compute the gain from swapping vertex a (in g1) with vertex b (in g2).
    Gain = reduction in cut cost after the swap.
    """
    d_a = sum(
        w.get((min(a, nb), max(a, nb)), 1.0)
        for nb in adj.get(a, [])
        if (nb in g2)  # edges crossing before swap from a's side
    ) - sum(
        w.get((min(a, nb), max(a, nb)), 1.0)
        for nb in adj.get(a, [])
        if (nb in g1 and nb != a)
    )
    d_b = sum(
        w.get((min(b, nb), max(b, nb)), 1.0)
        for nb in adj.get(b, [])
        if (nb in g1)
    ) - sum(
        w.get((min(b, nb), max(b, nb)), 1.0)
        for nb in adj.get(b, [])
        if (nb in g2 and nb != b)
    )
    c_ab = w.get((min(a, b), max(a, b)), 0.0)
    return d_a + d_b - 2 * c_ab


def kernighan_lin(
    vertices: list,
    edges: list[tuple],
    weights: dict | None = None,
    max_passes: int = 20,
    seed_partition: tuple[list, list] | None = None,
) -> dict[str, Any]:
    """
    Kernighan–Lin heuristic balanced graph bipartition.

    Performs iterative vertex-swap passes until no improvement is found.

    Returns a dict with the same keys as exact_balanced_min_cut plus:
      passes_completed, swap_log
    """
    import random

    n = len(vertices)
    if n < 2:
        raise ValueError("Need at least 2 vertices.")
    if n % 2 != 0:
        raise ValueError("KL requires an even number of vertices for balanced partition.")

    # Build weight lookup
    if weights is None:
        weights = {}
    norm_w: dict[tuple, float] = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        ew = edge[2] if len(edge) > 2 else weights.get((min(u, v), max(u, v)), 1.0)
        norm_w[(min(u, v), max(u, v))] = float(ew)

    # Build adjacency
    adj: dict[str, list] = {v: [] for v in vertices}
    for edge in edges:
        u, v = edge[0], edge[1]
        adj[u].append(v)
        adj[v].append(u)

    # Initial partition
    if seed_partition:
        g1 = set(seed_partition[0])
        g2 = set(seed_partition[1])
    else:
        rng = random.Random(42)
        shuffled = list(vertices)
        rng.shuffle(shuffled)
        half = n // 2
        g1 = set(shuffled[:half])
        g2 = set(shuffled[half:])

    steps: list[dict] = []
    passes_completed = 0
    t0 = time.perf_counter()

    step_num = 0

    for pass_i in range(max_passes):
        locked1: set = set()
        locked2: set = set()
        pass_swaps: list[tuple] = []
        gains: list[float] = []
        current_g1 = set(g1)
        current_g2 = set(g2)

        for _ in range(n // 2):
            best_gain = None
            best_pair = None
            for a in current_g1 - locked1:
                for b in current_g2 - locked2:
                    gain = _kl_gain(a, b, current_g1, current_g2, adj, norm_w)
                    if best_gain is None or gain > best_gain:
                        best_gain = gain
                        best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            gains.append(best_gain)
            pass_swaps.append((a, b, best_gain))

            # Apply tentative swap
            current_g1.remove(a)
            current_g1.add(b)
            current_g2.remove(b)
            current_g2.add(a)
            locked1.add(b)
            locked2.add(a)

            cut_cost_now = compute_cut_cost(current_g1, current_g2, edges, norm_w)
            step_num += 1
            steps.append({
                "step_index": step_num,
                "pass": pass_i + 1,
                "swap": (a, b),
                "gain": best_gain,
                "group1": sorted(current_g1),
                "group2": sorted(current_g2),
                "cut_edges": _cut_edges_list(current_g1, current_g2, edges, norm_w),
                "cut_cost": cut_cost_now,
                "best_cost": None,  # filled below
                "is_improvement": None,
                "description": (
                    f"Pass {pass_i+1}, swap #{len(pass_swaps)}: "
                    f"Move '{a}' → B and '{b}' → A. "
                    f"Gain = {best_gain:.2f}. "
                    f"Current cut cost = {cut_cost_now:.2f}."
                ),
            })

        # Find best prefix of swaps
        if not gains:
            break

        cum = 0.0
        best_cum = None
        best_k = 0
        for k, g in enumerate(gains):
            cum += g
            if best_cum is None or cum > best_cum:
                best_cum = cum
                best_k = k + 1

        if best_cum is None or best_cum <= 0:
            passes_completed = pass_i + 1
            break

        # Apply only the best prefix
        g1_new = set(g1)
        g2_new = set(g2)
        for i in range(best_k):
            a_s, b_s, _ = pass_swaps[i]
            if a_s in g1_new and b_s in g2_new:
                g1_new.remove(a_s)
                g1_new.add(b_s)
                g2_new.remove(b_s)
                g2_new.add(a_s)

        g1 = g1_new
        g2 = g2_new
        passes_completed = pass_i + 1

    # Fill best_cost into steps
    best_final = compute_cut_cost(g1, g2, edges, norm_w)
    for s in steps:
        s["best_cost"] = best_final
        s["is_improvement"] = s["cut_cost"] <= best_final

    runtime_ms = (time.perf_counter() - t0) * 1000
    final_cut = _cut_edges_list(g1, g2, edges, norm_w)

    return {
        "group1": sorted(g1),
        "group2": sorted(g2),
        "cut_edges": final_cut,
        "cut_cost": best_final,
        "runtime_ms": runtime_ms,
        "candidates_examined": len(steps),
        "optimality": f"Heuristic (Kernighan–Lin, {passes_completed} pass(es))",
        "steps": steps,
        "passes_completed": passes_completed,
        "balance": f"{len(g1)}:{len(g2)}",
        "balance_ratio": len(g1) / len(g2) if g2 else float("inf"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_graph(vertices: list, edges: list[tuple], weights: dict | None = None) -> list[str]:
    """
    Return a list of warning/error strings for the given graph.
    Empty list means the graph is valid.
    """
    errors = []
    if len(vertices) < 2:
        errors.append("Graph must have at least 2 vertices.")
        return errors

    vset = set(vertices)
    seen_edges: set[tuple] = set()
    for e in edges:
        if len(e) < 2:
            errors.append(f"Malformed edge: {e}")
            continue
        u, v = e[0], e[1]
        if u not in vset:
            errors.append(f"Edge references unknown vertex '{u}'.")
        if v not in vset:
            errors.append(f"Edge references unknown vertex '{v}'.")
        if u == v:
            errors.append(f"Self-loop detected on vertex '{u}' (will be ignored).")
        key = (min(u, v), max(u, v))
        if key in seen_edges:
            errors.append(f"Duplicate edge ({u}, {v}) detected.")
        seen_edges.add(key)
        if len(e) > 2:
            try:
                w = float(e[2])
                if w <= 0:
                    errors.append(f"Edge ({u}, {v}) has non-positive weight {e[2]}.")
            except (TypeError, ValueError):
                errors.append(f"Edge ({u}, {v}) has invalid weight '{e[2]}'.")

    isolated = [v for v in vertices if not any(v in (e[0], e[1]) for e in edges)]
    if isolated:
        errors.append(f"Isolated vertices (no edges): {isolated}. They will count toward partition size.")

    return errors
