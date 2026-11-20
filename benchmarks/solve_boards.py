"""Solve the game exactly on full rectangular boards (every cell holds one counter).

python benchmarks/solve_boards.py      # -> reports/tables/solved_boards.csv
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

from _util import write_csv

from hinger.solver import Solver
from hinger.state import State

SIZES = [(r, c) for r in range(1, 6) for c in range(r, 9) if r * c <= 25]
WORDS = {1: "first player wins", 0: "draw", -1: "second player wins"}


def solve(size):
    r, c = size
    solver = Solver()
    t0 = time.perf_counter()
    v = solver.value(State.full(r, c))
    return r, c, v, solver.nodes, time.perf_counter() - t0


def main() -> None:
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(solve, SIZES))
    rows = [
        {
            "rows": r,
            "cols": c,
            "counters": r * c,
            "result": WORDS[v],
            "positions_solved": nodes,
            "seconds": round(t, 2),
        }
        for r, c, v, nodes, t in sorted(results)
    ]
    write_csv("solved_boards.csv", rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
