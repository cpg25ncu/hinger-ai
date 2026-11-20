import itertools
import random

import pytest

from hinger.scenarios import random_path_problem
from hinger.search import (
    find_path,
    h_count,
    h_pair,
    is_safe_path,
    min_safe,
    path_astar,
    path_bfs,
    path_cost,
    path_dfs,
    path_iddfs,
)
from hinger.state import State

UNIT = [path_bfs, path_dfs, path_iddfs, path_astar]


@pytest.mark.parametrize("fn", UNIT)
def test_unit_cost_searches_find_valid_paths(fn):
    rng = random.Random(1)
    for k in range(0, 9):
        start, goal = random_path_problem(4, 4, k, rng)
        res = fn(start, goal)
        assert res.found
        assert len(res.path) == k  # every safe path removes exactly k counters
        assert is_safe_path(start, goal, res.path)


@pytest.mark.parametrize("fn", UNIT + [min_safe])
def test_impossible_problems_return_no_path(fn):
    start = State.parse("11/11")
    assert not fn(start, State.parse("11/12")).found  # goal needs a counter added
    hinger_goal = State.parse("100/010/001")
    assert not fn(State.full(3, 3), hinger_goal).found  # goal itself is unsafe
    res = fn(State.parse("100/010/001"), State.parse("000/000/001"))
    assert not res.found and "start" in res.reason


def test_unreachable_goal_needs_exhaustive_search():
    # The goal is safe, but every route to it goes through a hinger state.
    start = State.parse("111/111")
    goal = State.parse("101/000")
    for fn in UNIT + [min_safe]:
        res = fn(start, goal)
        assert not res.found, fn.__name__


def test_size_mismatch_raises():
    with pytest.raises(ValueError):
        path_bfs(State.parse("11"), State.parse("1"))


def test_pruning_does_not_change_the_answer():
    rng = random.Random(2)
    for _ in range(20):
        start, goal = random_path_problem(3, 4, 4, rng)
        a, b = path_bfs(start, goal, prune=True), path_bfs(start, goal, prune=False)
        assert a.found and b.found and len(a.path) == len(b.path)
        assert a.expanded <= b.expanded


def _all_safe_path_costs(start, goal):
    """Brute force: cost of every safe ordering of the removals (tiny problems only)."""
    removals = []
    for i, (s, g) in enumerate(zip(start.cells, goal.cells)):
        removals += [divmod(i, start.cols)] * (s - g)
    costs = set()
    for order in set(itertools.permutations(removals)):
        if is_safe_path(start, goal, list(order)):
            costs.add(path_cost(start, list(order)))
    return costs


def test_min_safe_is_optimal_against_brute_force():
    rng = random.Random(3)
    checked = 0
    while checked < 25:
        start, goal = random_path_problem(3, 3, rng.randint(1, 5), rng, max_counters=2)
        costs = _all_safe_path_costs(start, goal)
        for h in ("none", "count", "pair"):
            res = min_safe(start, goal, heuristic=h)
            assert res.found and res.cost == min(costs) == path_cost(start, res.path)
        checked += 1


def test_binary_boards_every_safe_path_costs_the_same():
    rng = random.Random(6)
    for _ in range(15):
        start, goal = random_path_problem(3, 3, rng.randint(1, 5), rng)
        costs = _all_safe_path_costs(start, goal)
        assert len(costs) == 1
        assert h_pair(start, goal) == costs.pop()  # h_pair is exact on binary boards


def test_heuristics_are_admissible_and_consistent():
    rng = random.Random(4)
    for _ in range(40):
        start, goal = random_path_problem(4, 4, rng.randint(2, 7), rng, max_counters=2)
        exact = min_safe(start, goal, heuristic="none").cost
        assert h_count(start, goal) <= h_pair(start, goal) <= exact
        # Consistency on every edge out of the start state.
        for mv, child in start.successors():
            if child.cells[mv[0] * child.cols + mv[1]] >= goal.cells[mv[0] * goal.cols + mv[1]]:
                c = start.move_cost(mv)
                assert h_pair(start, goal) <= c + h_pair(child, goal)
                assert h_count(start, goal) <= c + h_count(child, goal)


def test_pair_heuristic_expands_fewer_states_than_dijkstra():
    rng = random.Random(5)
    dij = pair = 0
    for _ in range(15):
        start, goal = random_path_problem(4, 5, 8, rng, max_counters=2)
        dij += min_safe(start, goal, heuristic="none").expanded
        pair += min_safe(start, goal, heuristic="pair").expanded
    assert pair < dij


def test_find_path_dispatch():
    start, goal = State.full(3, 3), State.parse("011/111/111")
    assert find_path(start, goal, "bfs").found
    assert find_path(start, goal, "min_safe").found
    with pytest.raises(ValueError):
        find_path(start, goal, "nope")
    with pytest.raises(ValueError):
        min_safe(start, goal, heuristic="nope")
