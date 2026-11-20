"""Random positions and search problems for tests, benchmarks and demos."""

from __future__ import annotations

import random

from .state import State

# Starting boards (all hinger-free). The comment gives the exact result for the
# player who moves first, from the solver.
PRESETS: dict[str, str] = {
    "cross": "01110/11111/11111/11111/01110",  # first player wins
    "diamond": "00100/01110/11111/01110/00100",  # first player wins
    "staircase": "11000/11100/01110/00111/00011",  # first player wins
    "ring": "1111/1001/1111",  # first player loses
    "piles": "2112/1221/2112",  # draw; piles of two counters
    "bars": "11011/11011/11011/11011",  # draw; two separate regions
    "full4x4": "1111/1111/1111/1111",  # first player loses
    "full4x5": "11111/11111/11111/11111",  # first player loses
    "full5x5": "11111/11111/11111/11111/11111",  # first player wins
}


def preset(name: str) -> State:
    try:
        return State.parse(PRESETS[name])
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None


def safe_walk(state: State, steps: int, rng: random.Random) -> State | None:
    """Make ``steps`` random moves that never create a hinger.

    Returns ``None`` if the walk gets stuck (every move would create a hinger,
    or the board runs out of counters) before ``steps`` moves are made.
    """
    if not state.is_safe():
        return None
    s = state
    for _ in range(steps):
        moves = s.legal_moves()
        rng.shuffle(moves)
        for mv in moves:
            nxt = s.apply(mv)
            if nxt.is_safe():
                s = nxt
                break
        else:
            return None
    return s


def random_position(
    rows: int,
    cols: int,
    rng: random.Random,
    *,
    removed: tuple[float, float] = (0.0, 0.6),
    max_counters: int = 1,
) -> State:
    """A random hinger-free position, reached by safe moves from a full board.

    ``removed`` is the range for the fraction of counters taken off before the
    position is returned. Positions produced this way are typical mid-game
    positions, unlike uniformly random boards, which usually contain hingers.
    """
    while True:
        start = (
            State.full(rows, cols, 1)
            if max_counters == 1
            else State._from_cells(rows, cols, tuple(rng.randint(1, max_counters) for _ in range(rows * cols)))
        )
        total = start.total_counters()
        k = int(round(rng.uniform(*removed) * total))
        s = safe_walk(start, k, rng)
        if s is not None:
            return s


def random_path_problem(
    rows: int,
    cols: int,
    k: int,
    rng: random.Random,
    *,
    max_counters: int = 1,
    removed: tuple[float, float] = (0.0, 0.3),
) -> tuple[State, State]:
    """A ``(start, goal)`` pair where ``goal`` is ``k`` safe moves after ``start``,
    so at least one safe path is guaranteed to exist."""
    while True:
        start = random_position(rows, cols, rng, removed=removed, max_counters=max_counters)
        goal = safe_walk(start, k, rng)
        if goal is not None:
            return start, goal
