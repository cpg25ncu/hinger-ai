import json

import pytest

from hinger.agents import Agent, GreedyAgent, make_agent
from hinger.cli import main
from hinger.game import GameResult, parse_move, play, render, replay
from hinger.state import State


class Scripted(Agent):
    kind = "scripted"

    def __init__(self, moves, name="scripted"):
        super().__init__(name)
        self.moves = list(moves)

    def choose(self, state):
        return self.moves.pop(0)


def test_hinger_move_wins():
    s = State.parse("100/010/001")
    res = play(s, Scripted([(1, 1)], "alice"), Scripted([], "bob"), output=None)
    assert res.winner == "alice" and res.winner_side == "A" and res.reason == "hinger"
    assert res.moves[0].hinger and res.moves[0].cost == 3


def test_clearing_the_board_is_a_draw():
    s = State.parse("1001")
    res = play(s, Scripted([(0, 0)]), Scripted([(0, 3)]), output=None)
    assert res.winner is None and res.reason == "draw" and len(res.moves) == 2


@pytest.mark.parametrize("bad", [(0, 1), (5, 5), "x", None, (0,)])
def test_illegal_move_loses(bad):
    s = State.parse("1001")
    res = play(s, Scripted([bad], "cheat"), Scripted([], "honest"), output=None)
    assert res.winner == "honest" and res.reason == "illegal move"


def test_crashing_agent_loses():
    class Boom(Agent):
        def choose(self, state):
            raise RuntimeError("boom")

    res = play(State.parse("11"), Boom("boom"), GreedyAgent(seed=0), output=None)
    assert res.winner_side == "B" and res.reason == "error"


def test_time_limit():
    class Slow(Agent):
        def choose(self, state):
            import time

            time.sleep(0.05)
            return state.legal_moves()[0]

    res = play(State.parse("11"), Slow("slow"), GreedyAgent(seed=0), time_limit=0.01, output=None)
    assert res.winner_side == "B" and res.reason == "timeout"


def test_human_input_is_reprompted_on_typos():
    answers = iter(["hello", "1 1"])
    res = play(State.parse("100/010/001"), None, GreedyAgent(seed=0), output=None, input_fn=lambda _p: next(answers))
    assert res.winner == "Human" and res.reason == "hinger"


def test_same_names_are_disambiguated():
    res = play(State.parse("1001"), GreedyAgent(seed=0), GreedyAgent(seed=1), output=None)
    assert {res.player_a, res.player_b} == {"greedy (A)", "greedy (B)"}


def test_save_load_and_replay(tmp_path):
    a, b = make_agent("greedy", seed=1), make_agent("greedy", "other", seed=2)
    res = play(State.full(3, 3), a, b, output=None)
    path = res.save(tmp_path / "g.json")
    data = json.loads(path.read_text())
    assert data["format"] == "hinger-game/1"
    back = GameResult.load(path)
    assert back.winner == res.winner and [m.move for m in back.moves] == [m.move for m in res.moves]
    states = list(back.states())
    assert len(states) == len(back.moves) + 1
    lines = []
    replay(back, lines.append)
    assert res.summary() in lines[-1]


def test_render_and_parse_move():
    out = render(State.parse("100/010/001"), hints=True, last=(0, 0))
    assert "1*" in out and "[1]" in out
    assert parse_move("(2, 3)") == (2, 3) and parse_move("2,3") == (2, 3) and parse_move("a b") is None


def test_cli_commands(tmp_path, capsys):
    assert main(["solve", "111/111/111"]) == 0
    assert "WIN" in capsys.readouterr().out
    assert main(["path", "111/111/111", "011/111/110", "--min-cost"]) == 0
    assert "A* (h_pair)" in capsys.readouterr().out
    game = tmp_path / "g.json"
    args = "play --board ring -a greedy -b minimax --depth 2 --seed 1 --save".split()
    assert main([*args, str(game)]) == 0
    assert main(["replay", str(game)]) == 0
    assert main(["solve", "12/3"]) == 2  # ragged board: clean error, no traceback
