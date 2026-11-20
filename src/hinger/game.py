"""Game loop, move records and replays.

``play(state, player_a, player_b)`` runs one game. A player is an ``Agent`` or
``None`` for a human at the terminal. Player A moves first. The game ends when

* a player moves on a hinger: that player wins (``reason="hinger"``);
* the last counter is removed without a hinger: draw (``reason="draw"``);
* a player makes an illegal move, raises an error or exceeds the per-move time
  limit: the opponent wins (``"illegal move"``, ``"error"``, ``"timeout"``).

Every move is recorded with its cost, the time it took and the board after it,
and a finished game can be saved to JSON and replayed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .agents import Agent
from .state import Move, State

Player = Agent | None  # None = human at the terminal


@dataclass
class MoveRecord:
    ply: int
    side: str  # "A" or "B"
    player: str
    move: Move | None
    hinger: bool = False
    cost: int = 0
    elapsed: float = 0.0
    counters_after: int = 0


@dataclass
class GameResult:
    initial: State
    player_a: str
    player_b: str
    winner: str | None = None  # winner's name; None for a draw
    winner_side: str | None = None  # "A", "B" or None
    reason: str = ""
    moves: list[MoveRecord] = field(default_factory=list)

    @property
    def total_time(self) -> dict[str, float]:
        t = {"A": 0.0, "B": 0.0}
        for m in self.moves:
            t[m.side] += m.elapsed
        return t

    def states(self) -> Iterator[State]:
        """The initial board followed by the board after each legal move."""
        s = self.initial
        yield s
        for m in self.moves:
            if m.move is not None and s.is_legal(m.move):
                s = s.apply(m.move)
                yield s

    def to_dict(self) -> dict:
        return {
            "format": "hinger-game/1",
            "initial": self.initial.grid,
            "player_a": self.player_a,
            "player_b": self.player_b,
            "winner": self.winner,
            "winner_side": self.winner_side,
            "reason": self.reason,
            "moves": [asdict(m) for m in self.moves],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "GameResult":
        moves = []
        for m in data["moves"]:
            m = dict(m)
            m["move"] = tuple(m["move"]) if m["move"] is not None else None
            moves.append(MoveRecord(**m))
        return cls(
            initial=State(data["initial"]),
            player_a=data["player_a"],
            player_b=data["player_b"],
            winner=data["winner"],
            winner_side=data["winner_side"],
            reason=data["reason"],
            moves=moves,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GameResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        n = len(self.moves)
        if self.winner is None:
            return f"Draw after {n} moves ({self.reason})."
        return f"{self.winner} (player {self.winner_side}) wins after {n} moves ({self.reason})."


def render(state: State, *, hints: bool = False, last: Move | None = None) -> str:
    """Board with row/column indices. Empty cells show as '.'.

    With ``hints`` hinger cells are marked ``*``; ``last`` is wrapped in brackets.
    """
    hingers = state.hingers() if hints else frozenset()
    width = max(2, len(str(max(state.cells, default=0))) + 2)
    lines = ["   " + "".join(f"{c:>{width + 1}}" for c in range(state.cols))]
    for r in range(state.rows):
        cells = []
        for c in range(state.cols):
            v = state[(r, c)]
            txt = str(v) if v else "."
            if (r, c) in hingers:
                txt += "*"
            if last == (r, c):
                txt = f"[{txt}]"
            cells.append(f"{txt:>{width + 1}}")
        lines.append(f"{r:>3}" + "".join(cells))
    return "\n".join(lines)


def parse_move(text: str) -> Move | None:
    """Parse ``"r c"``, ``"r,c"`` or ``"(r, c)"``; ``None`` if it is not two integers."""
    parts = text.replace("(", " ").replace(")", " ").replace(",", " ").split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _player_names(a: Player, b: Player) -> tuple[str, str]:
    na = a.name if a is not None else "Human"
    nb = b.name if b is not None else "Human"
    if na == nb:
        na, nb = f"{na} (A)", f"{nb} (B)"
    return na, nb


def play(
    state: State,
    player_a: Player,
    player_b: Player,
    *,
    time_limit: float | None = None,
    output: Callable[[str], None] | None = print,
    input_fn: Callable[[str], str] = input,
    hints: bool = False,
) -> GameResult:
    """Play one game; player A moves first. Returns a ``GameResult``.

    ``time_limit`` is the maximum thinking time per move in seconds (agents and
    humans alike); exceeding it loses the game. Human input that cannot be
    parsed as a move is asked for again, but a well-formed illegal move (an
    empty or off-board cell) loses, as the rules require.
    """
    say = output or (lambda _msg: None)
    names = _player_names(player_a, player_b)
    players = (player_a, player_b)
    for p in players:
        if p is not None:
            p.reset()
    result = GameResult(initial=state, player_a=names[0], player_b=names[1])

    def finish(winner_idx: int | None, reason: str) -> GameResult:
        result.reason = reason
        if winner_idx is not None:
            result.winner = names[winner_idx]
            result.winner_side = "AB"[winner_idx]
        say(result.summary())
        return result

    cur = state
    say(f"{names[0]} (A) vs {names[1]} (B)")
    say(render(cur, hints=hints))
    ply = 0
    turn = 0
    while True:
        if cur.is_empty():
            return finish(None, "draw")
        ply += 1
        player, label, side = players[turn], names[turn], "AB"[turn]
        t0 = time.perf_counter()
        try:
            if player is None:
                while True:
                    raw = input_fn(f"[{label}] your move (row col): ")
                    move = parse_move(raw)
                    if move is not None:
                        break
                    say(f"  could not read {raw!r}; type the row and column, e.g. 2 3")
            else:
                move = player.choose(cur)
        except Exception as exc:  # an agent crashing loses the game
            elapsed = time.perf_counter() - t0
            result.moves.append(
                MoveRecord(ply, side, label, None, elapsed=elapsed, counters_after=cur.total_counters())
            )
            say(f"[{label}] raised {exc!r}")
            return finish(1 - turn, "error")
        elapsed = time.perf_counter() - t0

        if not cur.is_legal(move):
            result.moves.append(
                MoveRecord(ply, side, label, None, elapsed=elapsed, counters_after=cur.total_counters())
            )
            say(f"[{label}] illegal move {move!r}")
            return finish(1 - turn, "illegal move")
        if time_limit is not None and elapsed > time_limit:
            result.moves.append(
                MoveRecord(ply, side, label, move, elapsed=elapsed, counters_after=cur.total_counters())
            )
            say(f"[{label}] took {elapsed:.2f}s, over the {time_limit:.2f}s limit")
            return finish(1 - turn, "timeout")

        hinger = cur.is_hinger(move)
        cost = cur.move_cost(move)
        cur = cur.apply(move)
        result.moves.append(MoveRecord(ply, side, label, move, hinger, cost, elapsed, cur.total_counters()))
        say(f"\n{ply:>3}. {label} -> {move}" + ("  HINGER!" if hinger else "") + f"   ({elapsed:.2f}s)")
        say(render(cur, hints=hints, last=move))
        if hinger:
            return finish(turn, "hinger")
        turn = 1 - turn


def replay(result: GameResult, output: Callable[[str], None] = print, *, hints: bool = True) -> None:
    """Print a recorded game move by move."""
    output(f"{result.player_a} (A) vs {result.player_b} (B)")
    s = result.initial
    output(render(s, hints=hints))
    for m in result.moves:
        if m.move is None or not s.is_legal(m.move):
            output(f"\n{m.ply:>3}. {m.player}: no legal move")
            break
        s = s.apply(m.move)
        tag = "  HINGER!" if m.hinger else ""
        output(f"\n{m.ply:>3}. {m.player} -> {m.move}  cost {m.cost}{tag}")
        output(render(s, hints=hints, last=m.move))
    output("\n" + result.summary())
