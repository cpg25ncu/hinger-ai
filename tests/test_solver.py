import random

from hinger.scenarios import random_position
from hinger.solver import DRAW, LOSS, WIN, Solver, canonical, win
from hinger.state import State


def _plain_negamax(s: State) -> int:
    """Reference solver: no memo, no symmetry, no cut-offs."""
    if s.has_hinger():
        return WIN
    if s.is_empty():
        return DRAW
    return max(-_plain_negamax(c) for _m, c in s.successors())


def test_basic_positions():
    assert Solver().value(State.parse("100/010/001")) == WIN  # take the hinger
    assert Solver().value(State.parse("1")) == DRAW  # last counter: board cleared
    assert Solver().value(State.parse("11")) == DRAW
    assert Solver().value(State.full(3, 4)) == LOSS
    assert win(State.full(3, 3)) is True
    assert win(State.full(4, 4)) is False


def test_matches_plain_negamax_on_small_random_positions():
    rng = random.Random(0)
    solver = Solver()
    checked = 0
    while checked < 120:
        s = random_position(
            rng.randint(2, 3), rng.randint(2, 4), rng, removed=(0.0, 0.8), max_counters=rng.choice([1, 2])
        )
        if s.total_counters() > 7:  # the reference has no memo: keep the tree small
            continue
        assert solver.value(s) == _plain_negamax(s), repr(s)
        checked += 1


def test_symmetry_does_not_change_values():
    rng = random.Random(1)
    a, b = Solver(use_symmetry=True), Solver(use_symmetry=False)
    for _ in range(60):
        s = random_position(3, 3, rng, removed=(0.0, 0.5))
        assert a.value(s) == b.value(s)


def test_canonical_form_is_shared_by_symmetric_boards():
    s = State.parse("110/011/000")
    mirrored = State.parse("011/110/000")
    rotated = State.parse("001/011/010")
    assert canonical(s) == canonical(mirrored) == canonical(rotated)


def test_best_move_achieves_the_value():
    rng = random.Random(2)
    solver = Solver()
    for _ in range(40):
        s = random_position(3, 4, rng, removed=(0.0, 0.5))
        if s.is_empty():
            continue
        mv = solver.best_move(s)
        if s.has_hinger():
            assert s.is_hinger(mv)
        else:
            assert -solver.value(s.apply(mv)) == solver.value(s)
