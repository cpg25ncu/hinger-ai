"""Board representation and rules of Hinger.

Hinger is played on an ``rows x cols`` board. Every cell holds a pile of zero or
more counters; a cell with at least one counter is *active*. Two cells are
adjacent when they touch horizontally, vertically or diagonally (8-neighbourhood),
and an *active region* is a connected component of active cells.

A move removes one counter from an active cell. A *hinger* is an active cell with
exactly one counter whose removal splits its region, i.e. increases the number of
active regions. The first player to move on a hinger wins; if the board is
cleared without that happening, the game is a draw.

Hinger detection is the operation every search and agent in this package calls
most often. A direct implementation removes each candidate cell in turn and
recounts the regions, which costs O(N^2) on a board with N active cells. Here it
is reduced to a single O(N) pass: removing a cell increases the number of
connected components exactly when the cell is an *articulation point* (cut
vertex) of the 8-connected graph of active cells, so the hingers are the
articulation points that hold one counter. Articulation points are found with
Tarjan's low-link algorithm, written iteratively so large boards cannot hit
Python's recursion limit.
"""

from __future__ import annotations

import random
from collections import deque
from functools import lru_cache
from typing import Iterator, Sequence

Move = tuple[int, int]
"""A move is the ``(row, col)`` of the cell a counter is removed from."""


@lru_cache(maxsize=None)
def neighbour_table(rows: int, cols: int) -> tuple[tuple[int, ...], ...]:
    """Flat indices of the 8-neighbours of every cell, cached per board size."""
    table = []
    for r in range(rows):
        for c in range(cols):
            nbrs = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        nbrs.append(nr * cols + nc)
            table.append(tuple(nbrs))
    return tuple(table)


def articulation_points(cells: Sequence[int], nbrs: Sequence[Sequence[int]]) -> set[int]:
    """Return the articulation points of the graph of active cells.

    ``cells`` is the flattened board and ``nbrs`` its neighbour table. Runs an
    iterative version of Tarjan's algorithm in O(V + E); on a grid with
    8-adjacency E <= 4V, so this is linear in the number of active cells.
    """
    n = len(cells)
    disc = [0] * n  # discovery time, 0 = not visited yet
    low = [0] * n  # lowest discovery time reachable through the DFS subtree
    points: set[int] = set()
    clock = 0
    for root in range(n):
        if cells[root] == 0 or disc[root]:
            continue
        clock += 1
        disc[root] = low[root] = clock
        root_children = 0
        # Each frame: (vertex, DFS parent, iterator over the vertex's neighbours).
        stack = [(root, -1, iter(nbrs[root]))]
        while stack:
            v, parent, it = stack[-1]
            descended = False
            for w in it:
                if cells[w] == 0:
                    continue
                if not disc[w]:
                    clock += 1
                    disc[w] = low[w] = clock
                    stack.append((w, v, iter(nbrs[w])))
                    descended = True
                    break
                if w != parent and disc[w] < low[v]:
                    low[v] = disc[w]  # back edge
            if descended:
                continue
            stack.pop()
            if not stack:
                break
            p = stack[-1][0]
            if low[v] < low[p]:
                low[p] = low[v]
            if p == root:
                root_children += 1
            elif low[v] >= disc[p]:
                # No vertex in v's subtree reaches above p: removing p cuts it off.
                points.add(p)
        if root_children > 1:
            points.add(root)
    return points


class State:
    """An immutable Hinger board.

    States are hashable, so they can be used directly as keys in visited sets
    and transposition tables. Region and hinger information is computed lazily
    and cached on the instance.
    """

    __slots__ = ("rows", "cols", "cells", "_hash", "_hingers", "_regions")

    def __init__(self, grid: Sequence[Sequence[int]]):
        if not isinstance(grid, Sequence) or len(grid) == 0:
            raise ValueError("grid must be a non-empty 2D sequence of non-negative ints")
        rows = len(grid)
        if not isinstance(grid[0], Sequence) or len(grid[0]) == 0:
            raise ValueError("grid rows must be non-empty sequences")
        cols = len(grid[0])
        cells = []
        for r, row in enumerate(grid):
            if not isinstance(row, Sequence) or len(row) != cols:
                raise ValueError(f"row {r} has a different length from row 0 ({cols})")
            for c, v in enumerate(row):
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise ValueError(f"cell ({r},{c}) = {v!r}; counters must be non-negative ints")
                cells.append(v)
        self._init(rows, cols, tuple(cells))

    def _init(self, rows: int, cols: int, cells: tuple[int, ...]) -> None:
        self.rows = rows
        self.cols = cols
        self.cells = cells
        self._hash = hash((rows, cols, cells))
        self._hingers: frozenset[Move] | None = None
        self._regions: int | None = None

    @classmethod
    def _from_cells(cls, rows: int, cols: int, cells: tuple[int, ...]) -> "State":
        """Fast constructor for internal use (no validation)."""
        s = object.__new__(cls)
        s._init(rows, cols, cells)
        return s

    # ------------------------------------------------------------------ factories
    @classmethod
    def parse(cls, text: str) -> "State":
        """Build a state from text.

        Rows are separated by newlines or ``/``. Within a row, cells are separated
        by whitespace or commas; a row without separators is read one digit per
        cell. ``"110/011"`` and ``"1 1 0\\n0 1 1"`` give the same board.
        """
        rows = [line.strip() for line in text.replace("/", "\n").splitlines() if line.strip()]
        grid = []
        for line in rows:
            line = line.replace(",", " ")
            parts = line.split() if " " in line else list(line)
            try:
                grid.append([int(p) for p in parts])
            except ValueError as exc:
                raise ValueError(f"cannot parse board row {line!r}") from exc
        return cls(grid)

    @classmethod
    def full(cls, rows: int, cols: int, counters: int = 1) -> "State":
        """A board with ``counters`` counters on every cell."""
        return cls._from_cells(rows, cols, (counters,) * (rows * cols))

    @classmethod
    def random(
        cls,
        rows: int,
        cols: int,
        *,
        density: float = 0.6,
        max_counters: int = 1,
        rng: random.Random | None = None,
    ) -> "State":
        """A random board: each cell is active with probability ``density`` and
        then holds between 1 and ``max_counters`` counters."""
        rng = rng or random.Random()
        cells = tuple(rng.randint(1, max_counters) if rng.random() < density else 0 for _ in range(rows * cols))
        return cls._from_cells(rows, cols, cells)

    # ------------------------------------------------------------------ basics
    @property
    def grid(self) -> list[list[int]]:
        """The board as a fresh 2D list (row-major)."""
        c = self.cols
        return [list(self.cells[r * c : (r + 1) * c]) for r in range(self.rows)]

    @property
    def size(self) -> tuple[int, int]:
        return self.rows, self.cols

    def __getitem__(self, move: Move) -> int:
        r, c = move
        return self.cells[r * self.cols + c]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.rows == other.rows and self.cols == other.cols and self.cells == other.cells

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other: "State") -> bool:
        # Lets states sit in heap tuples without an explicit tie-breaker.
        return self.cells < other.cells

    def __repr__(self) -> str:
        rows = "/".join("".join(map(str, row)) for row in self.grid)
        return f"State.parse({rows!r})"

    def __str__(self) -> str:
        return "\n".join(" ".join(str(v) for v in row) for row in self.grid)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def total_counters(self) -> int:
        return sum(self.cells)

    def num_active(self) -> int:
        return sum(1 for v in self.cells if v)

    def is_empty(self) -> bool:
        return not any(self.cells)

    def is_binary(self) -> bool:
        return all(v <= 1 for v in self.cells)

    # ------------------------------------------------------------------ moves
    def is_legal(self, move: object) -> bool:
        """True if ``move`` is an in-bounds ``(row, col)`` of an active cell."""
        if not (isinstance(move, tuple) and len(move) == 2):
            return False
        r, c = move
        if not (isinstance(r, int) and isinstance(c, int)) or isinstance(r, bool) or isinstance(c, bool):
            return False
        return self.in_bounds(r, c) and self.cells[r * self.cols + c] > 0

    def legal_moves(self) -> list[Move]:
        cols = self.cols
        return [divmod(i, cols) for i, v in enumerate(self.cells) if v]

    def apply(self, move: Move) -> "State":
        """Return the state after removing one counter at ``move``."""
        if not self.is_legal(move):
            raise ValueError(f"illegal move {move!r}")
        r, c = move
        return self._apply_index(r * self.cols + c)

    def _apply_index(self, i: int) -> "State":
        cells = self.cells
        return State._from_cells(self.rows, self.cols, cells[:i] + (cells[i] - 1,) + cells[i + 1 :])

    def successors(self) -> Iterator[tuple[Move, "State"]]:
        """Yield ``(move, next_state)`` for every legal move."""
        cols = self.cols
        for i, v in enumerate(self.cells):
            if v:
                yield divmod(i, cols), self._apply_index(i)

    def neighbours(self, r: int, c: int) -> list[Move]:
        """In-bounds 8-neighbours of ``(r, c)``, active or not."""
        cols = self.cols
        return [divmod(j, cols) for j in neighbour_table(self.rows, cols)[r * cols + c]]

    def active_neighbours(self, r: int, c: int) -> int:
        cells = self.cells
        return sum(1 for j in neighbour_table(self.rows, self.cols)[r * self.cols + c] if cells[j])

    def move_cost(self, move: Move) -> int:
        """Cost of a move: one plus the number of active cells adjacent to it."""
        r, c = move
        return 1 + self.active_neighbours(r, c)

    # ------------------------------------------------------------------ regions and hingers
    def num_regions(self) -> int:
        """Number of active regions (8-connected components of active cells)."""
        if self._regions is None:
            cells = self.cells
            nbrs = neighbour_table(self.rows, self.cols)
            seen = [False] * len(cells)
            regions = 0
            for start, v in enumerate(cells):
                if not v or seen[start]:
                    continue
                regions += 1
                seen[start] = True
                queue = deque([start])
                while queue:
                    i = queue.popleft()
                    for j in nbrs[i]:
                        if cells[j] and not seen[j]:
                            seen[j] = True
                            queue.append(j)
            self._regions = regions
        return self._regions

    def hingers(self) -> frozenset[Move]:
        """All hinger cells, found in one O(N) articulation-point pass."""
        if self._hingers is None:
            cells = self.cells
            cut = articulation_points(cells, neighbour_table(self.rows, self.cols))
            cols = self.cols
            self._hingers = frozenset(divmod(i, cols) for i in cut if cells[i] == 1)
        return self._hingers

    def num_hingers(self) -> int:
        return len(self.hingers())

    def has_hinger(self) -> bool:
        return bool(self.hingers())

    def is_hinger(self, move: Move) -> bool:
        return move in self.hingers()

    def is_safe(self) -> bool:
        """A state is safe when it contains no hinger."""
        return not self.hingers()


def hingers_bruteforce(state: State) -> frozenset[Move]:
    """Reference hinger detection: remove each single-counter cell and recount.

    O(N) region counts of O(N) each, so O(N^2) overall. Kept to cross-check the
    articulation-point version in the tests and to benchmark against it.
    """
    base = state.num_regions()
    found = set()
    for i, v in enumerate(state.cells):
        if v == 1 and state._apply_index(i).num_regions() > base:
            found.add(divmod(i, state.cols))
    return frozenset(found)
