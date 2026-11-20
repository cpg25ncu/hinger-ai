"""Hinger detection: articulation points (O(N)) vs remove-and-recount (O(N^2)).

python benchmarks/bench_hinger.py      # -> reports/tables/hinger_detection.csv
"""

from __future__ import annotations

import random
import time

from _util import median, write_csv

from hinger.state import State, hingers_bruteforce

SIZES = [3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40]
BOARDS_PER_SIZE = 20


def time_call(fn, boards, repeat: int) -> float:
    """Median seconds per call; each call gets a fresh State so no cache is hit."""
    per_board = []
    for b in boards:
        fresh = [State._from_cells(b.rows, b.cols, b.cells) for _ in range(repeat)]
        t0 = time.perf_counter()
        for s in fresh:
            fn(s)
        per_board.append((time.perf_counter() - t0) / repeat)
    return median(per_board)


def main() -> None:
    rng = random.Random(2025)
    rows = []
    for n in SIZES:
        # Dense random boards with some piles: many single-counter cells to test.
        boards = [State.random(n, n, density=0.75, max_counters=2, rng=rng) for _ in range(BOARDS_PER_SIZE)]
        for b in boards:
            assert b.hingers() == hingers_bruteforce(b), "implementations disagree"
        active = median([b.num_active() for b in boards])
        fast = time_call(State.hingers, boards, repeat=20)
        slow = time_call(hingers_bruteforce, boards, repeat=max(1, 400 // (n * n)))
        rows.append(
            {
                "board": f"{n}x{n}",
                "cells": n * n,
                "active_cells_median": active,
                "tarjan_ms": round(fast * 1000, 4),
                "bruteforce_ms": round(slow * 1000, 4),
                "speedup": round(slow / fast, 1),
            }
        )
        print(rows[-1], flush=True)
    write_csv("hinger_detection.csv", rows)


if __name__ == "__main__":
    main()
