"""Safe-path search between two Hinger states.

A *safe path* from ``start`` to ``goal`` is a sequence of moves in which no state
along the way, ``start`` and ``goal`` included, contains a hinger.

Two properties of the state space shape every algorithm here:

* **It is a DAG with fixed depths.** Every move removes exactly one counter, so a
  state ``s`` can only be reached from ``start`` after ``total(start) - total(s)``
  moves, whatever the route. Cycles are impossible and a visited set never needs
  to remember the depth a state was reached at.
* **Dead states can be pruned.** Counters are never added, so a state with fewer
  counters than ``goal`` in some cell can never reach it. Successor generation
  therefore only touches cells that still hold more counters than the goal
  (``prune=True``). On binary boards this shrinks the space from every subset of
  the active cells to the ``2^k`` states between ``start`` and ``goal``, where
  ``k`` is the number of counters to remove.

Consequently, with unit move costs **every** safe path has the same length
``k``. BFS, DFS, IDDFS and A* all return a path of that length when one exists;
what separates them is how much of the space they touch, which is what
``SearchResult`` reports.

With the real move costs (``min_safe``) the same holds on *binary* boards: of
two adjacent cells that are both cleared, exactly one pays for the other
(whichever goes first), so the total cost does not depend on the order. Costs
only differ between paths once piles hold two or more counters. ``h_pair`` is
exact on binary boards, which is why A* with it runs straight to the goal there.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from typing import Callable, Iterator

from .state import Move, State, neighbour_table


@dataclass
class SearchResult:
    """Outcome of a search plus the effort it took."""

    method: str
    path: list[Move] | None
    cost: int | None = None  # number of moves, or total move cost for min_safe
    expanded: int = 0  # states taken off the frontier and expanded
    generated: int = 0  # successor states created
    max_frontier: int = 0  # peak size of the frontier (memory proxy)
    elapsed: float = 0.0  # seconds
    reason: str = field(default="")  # why the search stopped without a path

    @property
    def found(self) -> bool:
        return self.path is not None


# ---------------------------------------------------------------------- helpers
def _precheck(start: State, goal: State) -> str:
    """Return a reason why no safe path can exist, or '' if search is needed."""
    if start.size != goal.size:
        raise ValueError(f"board sizes differ: {start.size} vs {goal.size}")
    if any(g > s for s, g in zip(start.cells, goal.cells)):
        return "goal has more counters than start in some cell"
    if not start.is_safe():
        return "start contains a hinger"
    if not goal.is_safe():
        return "goal contains a hinger"
    return ""


def _children(state: State, goal: State, prune: bool) -> Iterator[tuple[Move, State]]:
    """Successors of ``state``; with ``prune`` only moves that keep the goal reachable."""
    cols = state.cols
    gcells = goal.cells
    for i, v in enumerate(state.cells):
        if v and (not prune or v > gcells[i]):
            yield divmod(i, cols), state._apply_index(i)


def _reconstruct(parent: dict[State, tuple[State | None, Move | None]], end: State) -> list[Move]:
    moves = []
    node = end
    while True:
        prev, mv = parent[node]
        if prev is None:
            break
        moves.append(mv)
        node = prev
    moves.reverse()
    return moves


def is_safe_path(start: State, goal: State, moves: list[Move]) -> bool:
    """Check that ``moves`` leads from ``start`` to ``goal`` through safe states only."""
    s = start
    if not s.is_safe():
        return False
    for mv in moves:
        if not s.is_legal(mv):
            return False
        s = s.apply(mv)
        if not s.is_safe():
            return False
    return s == goal


def path_cost(start: State, moves: list[Move]) -> int:
    """Total move cost of ``moves`` played from ``start`` (1 + active neighbours each)."""
    total = 0
    s = start
    for mv in moves:
        total += s.move_cost(mv)
        s = s.apply(mv)
    return total


def _finish(res: SearchResult, t0: float, path: list[Move] | None, reason: str = "") -> SearchResult:
    res.path = path
    res.cost = len(path) if path is not None and res.cost is None else res.cost
    res.reason = "" if path is not None else (reason or "no safe path")
    res.elapsed = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------- uninformed search
def path_bfs(start: State, goal: State, *, prune: bool = True) -> SearchResult:
    """Breadth-first graph search (goal test on generation)."""
    t0 = time.perf_counter()
    res = SearchResult("BFS", None)
    why = _precheck(start, goal)
    if why:
        return _finish(res, t0, None, why)
    if start == goal:
        return _finish(res, t0, [])
    parent: dict[State, tuple[State | None, Move | None]] = {start: (None, None)}
    frontier = deque([start])
    res.max_frontier = 1
    while frontier:
        cur = frontier.popleft()
        res.expanded += 1
        for mv, nxt in _children(cur, goal, prune):
            res.generated += 1
            if nxt in parent:
                continue
            parent[nxt] = (cur, mv)  # unsafe states are recorded too, so they are never re-checked
            if not nxt.is_safe():
                continue
            if nxt == goal:
                return _finish(res, t0, _reconstruct(parent, nxt))
            frontier.append(nxt)
        res.max_frontier = max(res.max_frontier, len(frontier))
    return _finish(res, t0, None)


def path_dfs(start: State, goal: State, *, prune: bool = True) -> SearchResult:
    """Depth-first graph search with an explicit stack (no recursion limit)."""
    t0 = time.perf_counter()
    res = SearchResult("DFS", None)
    why = _precheck(start, goal)
    if why:
        return _finish(res, t0, None, why)
    parent: dict[State, tuple[State | None, Move | None]] = {start: (None, None)}
    stack = [start]
    res.max_frontier = 1
    while stack:
        cur = stack.pop()
        if cur == goal:
            return _finish(res, t0, _reconstruct(parent, cur))
        res.expanded += 1
        children = []
        for mv, nxt in _children(cur, goal, prune):
            res.generated += 1
            if nxt in parent:
                continue
            parent[nxt] = (cur, mv)
            if nxt.is_safe():
                children.append(nxt)
        stack.extend(reversed(children))  # explore the first child first
        res.max_frontier = max(res.max_frontier, len(stack))
    return _finish(res, t0, None)


def path_iddfs(start: State, goal: State, *, prune: bool = True) -> SearchResult:
    """Iterative deepening DFS.

    Each iteration is a depth-limited DFS with its own visited set. That is sound
    here because every route to a state has the same length (see the module
    docstring), so a state first met within the limit never needs revisiting
    with more depth to spare. Memory stays linear in the depth limit plus the
    visited set; the price is re-expanding the shallow levels on every
    iteration, and because all solutions sit at depth ``k`` only the last
    iteration can succeed.
    """
    t0 = time.perf_counter()
    res = SearchResult("IDDFS", None)
    why = _precheck(start, goal)
    if why:
        return _finish(res, t0, None, why)
    max_depth = start.total_counters() - goal.total_counters()
    for limit in range(max_depth + 1):
        parent: dict[State, tuple[State | None, Move | None]] = {start: (None, None)}
        # Frames: (state, depth, iterator over its children).
        stack = [(start, 0, None)]
        while stack:
            cur, depth, it = stack[-1]
            if it is None:
                if cur == goal:
                    return _finish(res, t0, _reconstruct(parent, cur))
                if depth == limit:
                    stack.pop()
                    continue
                res.expanded += 1
                it = _children(cur, goal, prune)
                stack[-1] = (cur, depth, it)
            for mv, nxt in it:
                res.generated += 1
                if nxt in parent:
                    continue
                parent[nxt] = (cur, mv)
                if nxt.is_safe():
                    stack.append((nxt, depth + 1, None))
                    break
            else:
                stack.pop()
            res.max_frontier = max(res.max_frontier, len(stack))
    return _finish(res, t0, None)


# ---------------------------------------------------------------------- heuristics
def h_count(state: State, goal: State) -> int:
    """Counters still to remove: sum over cells of ``state - goal``.

    Every move removes exactly one counter and costs at least 1, so this never
    overestimates, under unit or real move costs. A move changes it by exactly
    1, so it is also consistent. Under unit costs on the pruned space it is
    *exact*: every safe path to the goal has this length.
    """
    return sum(s - g for s, g in zip(state.cells, goal.cells))


@dataclass(frozen=True)
class _PairPlan:
    both: tuple[tuple[int, int], ...]  # adjacent cells that both stay active in the goal
    one: tuple[int, ...]  # cell that empties, next to a cell that stays active
    zero: tuple[tuple[int, int], ...]  # adjacent cells that both empty


_pair_cache: dict[State, _PairPlan] = {}


def _pair_plan(goal: State) -> _PairPlan:
    plan = _pair_cache.get(goal)
    if plan is None:
        g = goal.cells
        both, one, zero = [], [], []
        for i, nb in enumerate(neighbour_table(goal.rows, goal.cols)):
            for j in nb:
                if j <= i:
                    continue
                if g[i] and g[j]:
                    both.append((i, j))
                elif g[i]:
                    one.append(j)
                elif g[j]:
                    one.append(i)
                else:
                    zero.append((i, j))
        plan = _PairPlan(tuple(both), tuple(one), tuple(zero))
        if len(_pair_cache) > 64:
            _pair_cache.clear()
        _pair_cache[goal] = plan
    return plan


def h_pair(state: State, goal: State) -> int:
    """Pairwise lower bound on the remaining *move cost* (used by ``min_safe``).

    A move at x costs 1 plus the number of active neighbours y, so the remaining
    cost splits into ``h_count`` plus, for every adjacent pair {x, y}, the
    number of removals at one cell made while the other is still active. With
    ``d = state - goal`` the pair term is bounded below as follows:

    * both cells keep counters in the goal: both stay active throughout, so
      every removal at either pays for the other: ``d_x + d_y``;
    * only y keeps counters: y is always active, so each of x's removals pays:
      ``d_x``;
    * both empty: whichever cell empties first, all of its removals happen while
      the other still has counters: ``min(d_x, d_y)``.

    Summing lower bounds over disjoint parts of the cost gives an admissible
    heuristic. It is also consistent: a move at x lowers ``h_count`` by 1 and
    each pair term containing x by at most 1, and in every case where a pair
    term drops the neighbour is active, so the move pays for that drop too.
    """
    d = [s - g for s, g in zip(state.cells, goal.cells)]
    plan = _pair_plan(goal)
    total = sum(d)
    for i, j in plan.both:
        total += d[i] + d[j]
    for i in plan.one:
        total += d[i]
    for i, j in plan.zero:
        a, b = d[i], d[j]
        total += a if a < b else b
    return total


def _zero(_s: State, _g: State) -> int:
    return 0


# ---------------------------------------------------------------------- best-first search
def _best_first(
    start: State,
    goal: State,
    method: str,
    step_cost: Callable[[State, Move], int],
    heuristic: Callable[[State, State], int],
) -> SearchResult:
    """A* / uniform-cost search on the pruned safe-state graph.

    Ties on f are broken towards larger g (deeper nodes). With unit costs and
    the exact ``h_count`` every node has the same f, so this tie-break is what
    lets A* run straight down to the goal instead of sweeping level by level.
    """
    t0 = time.perf_counter()
    res = SearchResult(method, None)
    why = _precheck(start, goal)
    if why:
        return _finish(res, t0, None, why)
    tie = count()
    g_best: dict[State, int] = {start: 0}
    parent: dict[State, tuple[State | None, Move | None]] = {start: (None, None)}
    heap = [(heuristic(start, goal), 0, next(tie), start)]
    closed: set[State] = set()
    while heap:
        _f, neg_g, _, cur = heappop(heap)
        if cur in closed:
            continue
        g = -neg_g
        if cur == goal:
            res.cost = g
            return _finish(res, t0, _reconstruct(parent, cur))
        closed.add(cur)
        res.expanded += 1
        for mv, nxt in _children(cur, goal, prune=True):
            res.generated += 1
            if nxt in closed:
                continue
            ng = g + step_cost(cur, mv)
            if ng >= g_best.get(nxt, ng + 1):
                continue
            if not nxt.is_safe():
                closed.add(nxt)  # never enterable; skip any later rediscovery cheaply
                continue
            g_best[nxt] = ng
            parent[nxt] = (cur, mv)
            heappush(heap, (ng + heuristic(nxt, goal), -ng, next(tie), nxt))
        res.max_frontier = max(res.max_frontier, len(heap))
    return _finish(res, t0, None)


def _unit(_s: State, _mv: Move) -> int:
    return 1


def path_astar(start: State, goal: State) -> SearchResult:
    """A* with unit move costs and the ``h_count`` heuristic."""
    return _best_first(start, goal, "A*", _unit, h_count)


HEURISTICS: dict[str, Callable[[State, State], int]] = {
    "none": _zero,
    "count": h_count,
    "pair": h_pair,
}


def min_safe(start: State, goal: State, *, heuristic: str = "pair") -> SearchResult:
    """Minimum-cost safe path, where a move costs 1 + its active neighbours.

    ``heuristic="none"`` is Dijkstra's algorithm (uniform-cost search);
    ``"count"`` and ``"pair"`` run A* with ``h_count`` or ``h_pair``. All three
    return a path of the same, optimal cost; ``"pair"`` expands the fewest
    states. Works on non-binary boards.
    """
    try:
        h = HEURISTICS[heuristic]
    except KeyError:
        raise ValueError(f"unknown heuristic {heuristic!r}; choose from {sorted(HEURISTICS)}") from None
    label = {"none": "Dijkstra", "count": "A* (h_count)", "pair": "A* (h_pair)"}[heuristic]
    return _best_first(start, goal, label, State.move_cost, h)


SEARCHES: dict[str, Callable[..., SearchResult]] = {
    "bfs": path_bfs,
    "dfs": path_dfs,
    "iddfs": path_iddfs,
    "astar": path_astar,
}


def find_path(start: State, goal: State, method: str = "astar") -> SearchResult:
    """Dispatch to one of the unit-cost searches, or ``"min_safe"``."""
    if method == "min_safe":
        return min_safe(start, goal)
    try:
        return SEARCHES[method](start, goal)
    except KeyError:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(SEARCHES) + ['min_safe']}") from None
