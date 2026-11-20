"""Command-line interface: ``hinger <command> ...`` (or ``python -m hinger``)."""

from __future__ import annotations

import argparse
import random
import sys
import time

from .agents import AGENTS, Agent, make_agent
from .game import GameResult, play, render, replay
from .scenarios import PRESETS, preset, random_position
from .search import SEARCHES, min_safe, path_cost
from .solver import Solver
from .state import State

PLAYER_KINDS = ["human", *AGENTS]


def parse_board(text: str) -> State:
    """A preset name, ``random:RxC`` or a board such as ``11002/11000/00111``."""
    if text in PRESETS:
        return preset(text)
    if text.startswith("random:"):
        r, c = (int(x) for x in text.split(":", 1)[1].lower().split("x"))
        return random_position(r, c, random.Random(), removed=(0.0, 0.3))
    return State.parse(text)


def build_player(kind: str, args: argparse.Namespace, seed: int | None) -> Agent | None:
    if kind == "human":
        return None
    kwargs: dict = {}
    if kind == "minimax":
        kwargs["depth"] = args.depth
    if kind in ("alphabeta", "mcts"):
        kwargs["time_limit"] = args.think
    if kind in ("random", "greedy", "mcts"):
        kwargs["seed"] = seed
    return make_agent(kind, **kwargs)


def _add_game_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--board", default="cross", help=f"preset ({', '.join(PRESETS)}), random:RxC or a board like 1100/0110"
    )
    p.add_argument("--depth", type=int, default=3, help="search depth for minimax (default 3)")
    p.add_argument("--think", type=float, default=1.0, help="thinking time per move for alphabeta/mcts, seconds")
    p.add_argument("--seed", type=int, default=None, help="random seed for the stochastic agents")


def cmd_play(args: argparse.Namespace) -> int:
    board = parse_board(args.board)
    a = build_player(args.a, args, args.seed)
    b = build_player(args.b, args, None if args.seed is None else args.seed + 1)
    result = play(board, a, b, time_limit=args.time_limit, hints=args.hints)
    if args.save:
        print(f"saved to {result.save(args.save)}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import run_gui
    except ImportError as exc:  # pygame missing
        print(f"The GUI needs pygame: pip install 'hinger[gui]'  ({exc})", file=sys.stderr)
        return 1
    run_gui(
        parse_board(args.board),
        opponent=args.opponent,
        human_first=not args.ai_first,
        think=args.think,
        depth=args.depth,
        seed=args.seed,
    )
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    s = parse_board(args.board)
    print(render(s, hints=True))
    print(f"\ncounters={s.total_counters()} regions={s.num_regions()} hingers={sorted(s.hingers())}")
    solver = Solver()
    t0 = time.perf_counter()
    value = solver.value(s)
    words = {1: "WIN", 0: "DRAW", -1: "LOSS"}
    print(f"player to move: {words[value]}   ({solver.nodes:,} positions, {time.perf_counter() - t0:.2f}s)")
    if not s.is_empty() and not s.has_hinger():
        values = solver.move_values(s)
        for v in (1, 0, -1):
            moves = sorted(mv for mv, x in values.items() if x == v)
            if moves:
                print(f"  {words[v]:>4}: {moves}")
    return 0


def cmd_path(args: argparse.Namespace) -> int:
    start, goal = parse_board(args.start), parse_board(args.goal)
    methods = list(SEARCHES) if args.method == "all" else [args.method]
    rows = []
    for m in methods:
        rows.append(SEARCHES[m](start, goal))
    if args.min_cost:
        for h in ("none", "count", "pair"):
            rows.append(min_safe(start, goal, heuristic=h))
    print(
        f"{'method':<14}{'found':>6}{'moves':>7}{'cost':>6}{'expanded':>10}{'generated':>11}{'frontier':>10}{'ms':>9}"
    )
    for r in rows:
        cost = path_cost(start, r.path) if r.found else "-"
        print(
            f"{r.method:<14}{'yes' if r.found else 'no':>6}{len(r.path) if r.found else '-':>7}{cost:>6}"
            f"{r.expanded:>10}{r.generated:>11}{r.max_frontier:>10}{r.elapsed * 1000:>9.2f}"
        )
    best = next((r for r in reversed(rows) if r.found), None)
    if best:
        print(f"\n{best.method} path: {best.path}")
    else:
        print(f"\nno safe path: {rows[0].reason}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    replay(GameResult.load(args.file))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hinger", description="Search and game-playing AI for the Hinger board game.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("play", help="play in the terminal (human or AI on either side)")
    _add_game_options(p)
    p.add_argument("-a", choices=PLAYER_KINDS, default="human", help="player A, moves first (default human)")
    p.add_argument("-b", choices=PLAYER_KINDS, default="mcts", help="player B (default mcts)")
    p.add_argument("--time-limit", type=float, default=None, help="per-move limit; going over loses the game")
    p.add_argument("--hints", action="store_true", help="mark hinger cells with *")
    p.add_argument("--save", help="save the game record to this JSON file")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("gui", help="play against the AI in a window (needs pygame)")
    _add_game_options(p)
    p.add_argument("--opponent", choices=[k for k in AGENTS], default="mcts")
    p.add_argument("--ai-first", action="store_true", help="let the AI make the first move")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("solve", help="solve a position exactly (small boards)")
    p.add_argument("board")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("path", help="find a safe path between two positions")
    p.add_argument("start")
    p.add_argument("goal")
    p.add_argument("--method", choices=[*SEARCHES, "all"], default="all")
    p.add_argument("--min-cost", action="store_true", help="also find the minimum-cost safe path")
    p.set_defaults(func=cmd_path)

    p = sub.add_parser("replay", help="replay a saved game")
    p.add_argument("file")
    p.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
