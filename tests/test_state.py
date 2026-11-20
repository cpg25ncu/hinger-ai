import random

import pytest

from hinger.state import State, hingers_bruteforce


def test_parse_formats_agree():
    a = State.parse("110/011")
    b = State.parse("1 1 0\n0 1 1")
    c = State([[1, 1, 0], [0, 1, 1]])
    assert a == b == c
    assert hash(a) == hash(c)
    assert a.grid == [[1, 1, 0], [0, 1, 1]]


def test_str_is_a_plain_matrix():
    assert str(State.parse("102/030")) == "1 0 2\n0 3 0"


def test_repr_round_trips():
    s = State.parse("1102/0031")
    assert eval(repr(s)) == s


@pytest.mark.parametrize(
    "grid",
    [[], [[]], [[1, 2], [1]], [[1, -1]], [[1, 1.5]], [[True, 1]], "12"],
)
def test_invalid_grids_are_rejected(grid):
    with pytest.raises(ValueError):
        State(grid)


def test_state_is_immutable_and_grid_is_a_copy():
    s = State.parse("11/11")
    g = s.grid
    g[0][0] = 9
    assert s[(0, 0)] == 1
    t = s.apply((0, 0))
    assert s[(0, 0)] == 1 and t[(0, 0)] == 0


def test_regions_use_8_connectivity():
    # Diagonal neighbours are connected.
    assert State.parse("100/010/001").num_regions() == 1
    assert State.parse("101/000/101").num_regions() == 4
    assert State.parse("11002/11000/00111/00011").num_regions() == 2
    assert State.parse("000/000").num_regions() == 0


def test_hingers_known_cases():
    # Centre of a diagonal line is a hinger.
    assert State.parse("100/010/001").hingers() == {(1, 1)}
    # A pile of two counters is never a hinger, even at a cut point.
    assert State.parse("100/020/001").hingers() == frozenset()
    # Two blocks joined through one cell: the joints split the board.
    s = State.parse("11002/11000/00111/00011")
    assert s.hingers() == {(1, 1), (2, 2)}
    # A full rectangle (at least 2x2) has no cut vertex.
    assert State.full(4, 5).num_hingers() == 0
    # In a 1xn line every interior cell is a hinger.
    assert State.full(1, 5).hingers() == {(0, 1), (0, 2), (0, 3)}
    # Removing an isolated cell lowers the region count: not a hinger.
    assert State.parse("1").num_hingers() == 0


def test_articulation_points_match_bruteforce_on_random_boards():
    rng = random.Random(0)
    for _ in range(3000):
        s = State.random(
            rng.randint(1, 8), rng.randint(1, 8), density=rng.random(), max_counters=rng.choice([1, 2, 3]), rng=rng
        )
        assert s.hingers() == hingers_bruteforce(s), repr(s)


def test_large_board_does_not_hit_recursion_limit():
    s = State.full(60, 60)  # a 3600-cell DFS would overflow a recursive version
    assert s.num_hingers() == 0
    snake = State([[1] * 400])  # a 400-cell line: every interior cell is a hinger
    assert snake.num_hingers() == 398


def test_moves_and_costs():
    s = State.parse("11002/11000/00111/00011")
    assert len(s.legal_moves()) == 10
    assert len(list(s.successors())) == 10
    assert s.move_cost((0, 4)) == 1  # isolated pile
    assert s.move_cost((2, 2)) == 4  # (1,1), (2,3), (3,3) are active neighbours
    assert s.is_legal((0, 4)) and not s.is_legal((0, 2)) and not s.is_legal((9, 9))
    assert not s.is_legal("0 4") and not s.is_legal((True, 0))
    with pytest.raises(ValueError):
        s.apply((0, 2))
    t = s.apply((0, 4))
    assert t[(0, 4)] == 1 and t.total_counters() == s.total_counters() - 1


def test_successor_states_cache_independently():
    s = State.parse("100/010/001")
    assert s.has_hinger()
    child = s.apply((1, 1))
    assert child.num_regions() == 2 and not child.has_hinger()
