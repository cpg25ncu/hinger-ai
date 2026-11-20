"""Exact game-theoretic solver for Hinger.

Values are from the point of view of the player to move:
``+1`` win, ``0`` draw, ``-1`` loss.

The rules make the recursion short:

* if the position contains a hinger, the player to move takes it and wins;
* if the board is empty, the game is a draw;
* otherwise the value is the best of ``-value(child)`` over all moves.

Every move removes a counter, so the game tree is finite and acyclic; positions
are memoised in a transposition table (many move orders reach the same board),
and the search stops at the first winning move. Positions related by a symmetry
of the board (rotations and reflections) have the same value, so the table is
keyed by a canonical form, which cuts its size by up to 8x on square boards.

The solver is exponential in the number of counters: it is meant for small
boards, where it gives ground truth for testing and grading the heuristic
agents, and for answering ``win(state)`` exactly.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from .state import Move, State

WIN, DRAW, LOSS = 1, 0, -1


@lru_cache(maxsize=None)
def _symmetries(rows: int, cols: int) -> tuple[tuple[int, ...], ...]:
    """Index permutations for the board's symmetry group (4 or 8 elements)."""

    def perm(fn):
        return tuple(fn(r, c) for r in range(rows) for c in range(cols))

    sym = [
        perm(lambda r, c: r * cols + c),  # identity
        perm(lambda r, c: r * cols + (cols - 1 - c)),  # mirror left-right
        perm(lambda r, c: (rows - 1 - r) * cols + c),  # mirror top-bottom
        perm(lambda r, c: (rows - 1 - r) * cols + (cols - 1 - c)),  # rotate 180
    ]
    if rows == cols:
        n = rows
        sym += [
            perm(lambda r, c: c * n + r),  # transpose
            perm(lambda r, c: (n - 1 - c) * n + (n - 1 - r)),  # anti-transpose
            perm(lambda r, c: c * n + (n - 1 - r)),  # rotate 90
            perm(lambda r, c: (n - 1 - c) * n + r),  # rotate 270
        ]
    return tuple(sym)


def canonical(state: State) -> tuple[int, ...]:
    """Smallest image of the board under its symmetries, used as a table key."""
    cells = state.cells
    return min(tuple(cells[i] for i in p) for p in _symmetries(state.rows, state.cols))


class Solver:
    """Memoised negamax over win/draw/loss values."""

    def __init__(self, use_symmetry: bool = True):
        self.use_symmetry = use_symmetry
        self.table: dict[object, int] = {}
        self.nodes = 0

    def _key(self, state: State) -> object:
        return (state.rows, state.cols, canonical(state)) if self.use_symmetry else state

    def value(self, state: State) -> int:
        """+1 if the player to move can force a win, 0 a draw, -1 a loss."""
        depth_needed = state.total_counters() + 100
        if sys.getrecursionlimit() < depth_needed:
            sys.setrecursionlimit(depth_needed)
        return self._value(state)

    def _value(self, state: State) -> int:
        key = self._key(state)
        v = self.table.get(key)
        if v is not None:
            return v
        self.nodes += 1
        if state.has_hinger():
            v = WIN
        elif state.is_empty():
            v = DRAW
        else:
            v = LOSS
            # Try moves that do not hand the opponent a hinger first: they are
            # the only ones that can be wins, so a win is found sooner.
            children = list(state.successors())
            children.sort(key=lambda mc: mc[1].has_hinger())
            for _mv, child in children:
                cv = -self._value(child)
                if cv > v:
                    v = cv
                    if v == WIN:
                        break
        self.table[key] = v
        return v

    def move_values(self, state: State) -> dict[Move, int]:
        """Value of every legal move, from the mover's point of view."""
        self.value(state)  # also raises the recursion limit if needed
        return {mv: -self._value(child) for mv, child in state.successors()}

    def best_move(self, state: State) -> Move | None:
        """A move that achieves the position's value (a winning hinger if there is one)."""
        if state.is_empty():
            return None
        if state.has_hinger():
            return min(state.hingers())
        values = self.move_values(state)
        best = max(values.values())
        return min(mv for mv, v in values.items() if v == best)


def win(state: State, solver: Solver | None = None) -> bool:
    """True if the player to move in ``state`` can force a win (any board, binary or not)."""
    return (solver or Solver()).value(state) == WIN
