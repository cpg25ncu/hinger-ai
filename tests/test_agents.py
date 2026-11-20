import random

import pytest

from hinger.agents import (
    AGENTS,
    MATE_BOUND,
    AlphaBetaAgent,
    GreedyAgent,
    MCTSAgent,
    MinimaxAgent,
    make_agent,
)
from hinger.scenarios import random_position
from hinger.solver import Solver
from hinger.state import State

FAST = {
    "random": {"seed": 0},
    "greedy": {"seed": 0},
    "minimax": {"depth": 2},
    "alphabeta": {"time_limit": 0.05},
    "mcts": {"iterations": 60, "time_limit": None, "seed": 0},
    "solver": {},
}


@pytest.mark.parametrize("kind", sorted(AGENTS))
def test_every_agent_returns_legal_moves(kind):
    rng = random.Random(0)
    agent = make_agent(kind, **FAST[kind])
    for _ in range(8):
        s = random_position(3, 4, rng, removed=(0.0, 0.7))
        mv = agent.choose(s)
        assert s.is_legal(mv)
    assert agent.choose(State.parse("000/000")) is None


@pytest.mark.parametrize("kind", sorted(set(AGENTS) - {"random"}))
def test_agents_take_an_immediate_win(kind):
    s = State.parse("11000/11000/00111/00011")  # (1,1) and (2,2) are hingers
    mv = make_agent(kind, **FAST[kind]).choose(s)
    assert s.is_hinger(mv)


@pytest.mark.parametrize("kind", ["greedy", "minimax", "alphabeta", "mcts", "solver"])
def test_agents_avoid_handing_over_a_hinger(kind):
    # Removing (0,0) or (2,2) keeps the board safe; removing (1,1) creates hingers.
    s = State.parse("110/111/011")
    assert not s.has_hinger()
    mv = make_agent(kind, **FAST[kind]).choose(s)
    assert not s.apply(mv).has_hinger()


def test_minimax_and_alphabeta_variants_agree():
    rng = random.Random(1)
    for _ in range(25):
        s = random_position(4, 4, rng, removed=(0.1, 0.6))
        depth = rng.randint(1, 3)
        expected = MinimaxAgent(depth=depth).search(s)[0]
        for kw in (dict(use_tt=False, ordering=False), dict(use_tt=False), dict()):
            assert AlphaBetaAgent(time_limit=None, **kw).search(s, depth)[0] == expected


def test_alphabeta_searches_fewer_nodes_than_minimax():
    rng = random.Random(2)
    mm_nodes = ab_nodes = 0
    for _ in range(10):
        s = random_position(4, 5, rng, removed=(0.2, 0.5))
        mm = MinimaxAgent(depth=3)
        mm.search(s)
        ab = AlphaBetaAgent(time_limit=None)
        ab.search(s, 3)
        mm_nodes += mm.nodes
        ab_nodes += ab.nodes
    assert ab_nodes < mm_nodes / 2


def test_unlimited_alphabeta_matches_the_solver():
    rng = random.Random(3)
    solver = Solver()
    for _ in range(30):
        s = random_position(3, 4, rng, removed=(0.0, 0.5))
        if s.is_empty() or s.has_hinger():
            continue
        agent = AlphaBetaAgent(time_limit=None)
        agent.choose(s)
        score = agent.stats["score"]
        value = 1 if score >= MATE_BOUND else -1 if score <= -MATE_BOUND else 0
        assert value == solver.value(s)


def test_alphabeta_respects_its_time_limit():
    agent = AlphaBetaAgent(time_limit=0.1)
    agent.choose(State.full(6, 6))
    assert agent.stats["time"] < 0.5
    assert agent.stats["depth"] >= 1


def test_mcts_is_reproducible_with_a_seed():
    s = State.full(4, 4)
    a = MCTSAgent(iterations=200, time_limit=None, seed=42).choose(s)
    b = MCTSAgent(iterations=200, time_limit=None, seed=42).choose(s)
    assert a == b


def test_mcts_needs_a_budget():
    with pytest.raises(ValueError):
        MCTSAgent(time_limit=None, iterations=None)


def test_unknown_agent():
    with pytest.raises(ValueError):
        make_agent("oracle")


def test_greedy_plays_a_losing_move_only_when_forced():
    s = State.parse("111")  # every move either creates a hinger or... the middle is a hinger
    assert GreedyAgent(seed=0).choose(s) == (0, 1)
