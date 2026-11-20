"""Safe-path search: BFS, DFS, IDDFS and A* (unit cost), and the min-cost search
(Dijkstra vs A* with h_count vs A* with h_pair).

    python benchmarks/bench_search.py   # -> reports/tables/search_*.csv

Problems are random 5x5 positions. A *reachable* problem's goal is produced by
k random safe moves from the start, so a safe path is known to exist. An
*unreachable* problem's goal clears k random cells of the start and is safe
itself, but no safe path leads to it; proving that requires exhausting the space.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from _util import median, write_csv

from hinger.scenarios import random_path_problem, random_position
from hinger.search import is_safe_path, min_safe, path_astar, path_bfs, path_cost, path_dfs, path_iddfs
from hinger.state import State

ROWS, COLS = 5, 5
UNIT_K = [2, 4, 6, 8, 10, 12, 14]
COST_K = [4, 6, 8, 10, 12, 14, 16]
PER_K = 30
UNPRUNED_K = [2, 3, 4, 5, 6]


def unreachable_problem(k: int, rng: random.Random) -> tuple[State, State]:
    while True:
        start = random_position(ROWS, COLS, rng, removed=(0.0, 0.3))
        active = [i for i, v in enumerate(start.cells) if v]
        if len(active) < k:
            continue
        cells = list(start.cells)
        for i in rng.sample(active, k):
            cells[i] = 0
        goal = State._from_cells(ROWS, COLS, tuple(cells))
        if goal.is_safe() and not path_dfs(start, goal).found:
            return start, goal


def run_unit(task):
    kind, k, seed = task
    rng = random.Random(seed)
    if kind == "reachable":
        start, goal = random_path_problem(ROWS, COLS, k, rng)
    else:
        start, goal = unreachable_problem(k, rng)
    out = []
    for fn in (path_bfs, path_dfs, path_iddfs, path_astar):
        r = fn(start, goal)
        assert r.found == (kind == "reachable")
        assert not r.found or (is_safe_path(start, goal, r.path) and len(r.path) == k)
        out.append((r.method, r.expanded, r.max_frontier, r.elapsed))
    return kind, k, out


def run_unpruned(task):
    k, seed = task
    rng = random.Random(seed)
    start, goal = random_path_problem(ROWS, COLS, k, rng)
    a, b = path_bfs(start, goal, prune=True), path_bfs(start, goal, prune=False)
    assert a.found and b.found
    return k, a.expanded, b.expanded


def run_cost(task):
    piles, k, seed = task
    rng = random.Random(seed)
    start, goal = random_path_problem(ROWS, COLS, k, rng, max_counters=piles)
    out, costs = [], set()
    for h in ("none", "count", "pair"):
        r = min_safe(start, goal, heuristic=h)
        assert r.found and is_safe_path(start, goal, r.path)
        costs.add(path_cost(start, r.path))
        out.append((r.method, r.expanded, r.max_frontier, r.elapsed))
    assert len(costs) == 1, "heuristics disagree on the optimal cost"
    bfs_len_path = path_astar(start, goal)  # an arbitrary (unit-cost) safe path
    return piles, k, out, costs.pop(), path_cost(start, bfs_len_path.path)


def summarise(groups, extra_keys):
    rows = []
    for key, runs in groups.items():
        by_method = defaultdict(list)
        for method, expanded, frontier, elapsed in runs:
            by_method[method].append((expanded, frontier, elapsed))
        for method, xs in by_method.items():
            rows.append(
                {
                    **dict(zip(extra_keys, key)),
                    "method": method,
                    "problems": len(xs),
                    "expanded_median": median([x[0] for x in xs]),
                    "expanded_mean": round(sum(x[0] for x in xs) / len(xs), 1),
                    "frontier_median": median([x[1] for x in xs]),
                    "time_ms_median": round(median([x[2] for x in xs]) * 1000, 3),
                }
            )
    return rows


def main() -> None:
    workers = min(12, max(1, (sys.modules["os"].cpu_count() or 2) // 2))
    seed = iter(range(10_000_000))
    unit_tasks = [(kind, k, next(seed)) for kind in ("reachable", "unreachable") for k in UNIT_K for _ in range(PER_K)]
    cost_tasks = [(piles, k, next(seed)) for piles in (1, 2) for k in COST_K for _ in range(PER_K)]
    unpruned_tasks = [(k, next(seed)) for k in UNPRUNED_K for _ in range(PER_K)]

    with ProcessPoolExecutor(workers) as pool:
        unit = list(pool.map(run_unit, unit_tasks, chunksize=4))
        cost = list(pool.map(run_cost, cost_tasks, chunksize=4))
        unpruned = list(pool.map(run_unpruned, unpruned_tasks, chunksize=4))

    groups = defaultdict(list)
    for kind, k, out in unit:
        groups[(kind, k)].extend(out)
    write_csv("search_unit_cost.csv", summarise(groups, ("problem", "k")))

    groups = defaultdict(list)
    savings = defaultdict(list)
    for piles, k, out, best, naive in cost:
        groups[(piles, k)].extend(out)
        savings[(piles, k)].append((naive - best) / naive)
    rows = summarise(groups, ("max_counters", "k"))
    for r in rows:
        r["cost_saving_vs_arbitrary_path_pct"] = round(100 * median(savings[(r["max_counters"], r["k"])]), 1)
    write_csv("search_min_cost.csv", rows)

    by_k = defaultdict(list)
    for k, pruned, raw in unpruned:
        by_k[k].append((pruned, raw))
    write_csv(
        "search_pruning.csv",
        [
            {
                "k": k,
                "problems": len(v),
                "bfs_pruned_expanded_median": median([a for a, _ in v]),
                "bfs_unpruned_expanded_median": median([b for _, b in v]),
                "reduction_x": round(median([b / max(a, 1) for a, b in v]), 1),
            }
            for k, v in sorted(by_k.items())
        ],
    )


if __name__ == "__main__":
    main()
