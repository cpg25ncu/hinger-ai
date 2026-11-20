"""Game-playing agents: search efficiency, move quality and a round-robin tournament.

    python benchmarks/bench_agents.py [efficiency|quality|parity|tournament|all]

* efficiency  nodes searched by minimax vs alpha-beta variants at equal depth
              (all variants must return the same value; checked).
* quality     share of positions where an agent picks a game-theoretically
              optimal move, graded by the exact solver.
* parity      how often the player to move wins, by parity of the counters left.
* tournament  every agent plays every other from 12 openings, both colours.
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from _util import median, write_csv

from hinger.agents import AlphaBetaAgent, MinimaxAgent, make_agent
from hinger.game import play
from hinger.scenarios import preset, random_position
from hinger.solver import Solver

WORKERS = min(12, max(1, (os.cpu_count() or 2) // 2))
THINK = 0.2  # seconds per move for the time-limited agents


# ---------------------------------------------------------------------- efficiency
VARIANTS = [
    ("minimax", None),
    ("alpha-beta", dict(use_tt=False, ordering=False)),
    ("+ move ordering", dict(use_tt=False, ordering=True)),
    ("+ transposition table", dict(use_tt=True, ordering=True)),
]


def efficiency_task(args):
    seed, depths = args
    s = random_position(5, 5, random.Random(seed), removed=(0.2, 0.45))
    out = []
    for depth in depths:
        values = set()
        for label, kw in VARIANTS:
            if label == "minimax" and depth > 4:
                continue
            agent = MinimaxAgent(depth=depth) if kw is None else AlphaBetaAgent(time_limit=None, **kw)
            t0 = time.perf_counter()
            v, _ = agent.search(s, depth) if kw is not None else agent.search(s)
            out.append((depth, label, agent.nodes, time.perf_counter() - t0))
            values.add(v)
        assert len(values) == 1, f"variants disagree at depth {depth}: {values}"
    return out


def efficiency(n_positions: int = 40) -> None:
    tasks = [(1000 + i, [1, 2, 3, 4, 5, 6]) for i in range(n_positions)]
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(efficiency_task, tasks))
    groups = defaultdict(list)
    for res in results:
        for depth, label, nodes, t in res:
            groups[(depth, label)].append((nodes, t))
    rows = []
    for (depth, label), xs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], [v[0] for v in VARIANTS].index(kv[0][1]))
    ):
        rows.append(
            {
                "depth": depth,
                "variant": label,
                "positions": len(xs),
                "nodes_mean": round(sum(x[0] for x in xs) / len(xs)),
                "nodes_median": median([x[0] for x in xs]),
                "time_ms_mean": round(1000 * sum(x[1] for x in xs) / len(xs), 2),
            }
        )
    write_csv("agents_efficiency.csv", rows)


# ---------------------------------------------------------------------- move quality
QUALITY_AGENTS = {
    "random": lambda: make_agent("random", seed=0),
    "greedy": lambda: make_agent("greedy", seed=0),
    "minimax d=2": lambda: make_agent("minimax", depth=2),
    "minimax d=3": lambda: make_agent("minimax", depth=3),
    "alphabeta": lambda: make_agent("alphabeta", time_limit=THINK),
    "mcts": lambda: make_agent("mcts", time_limit=THINK, seed=0),
}


def quality_task(args):
    size, seed = args
    rng = random.Random(seed)
    solver = Solver()
    while True:
        s = random_position(*size, rng, removed=(0.1, 0.5))
        if s.is_empty() or s.has_hinger():
            continue
        values = solver.move_values(s)
        best = max(values.values())
        if any(v != best for v in values.values()):
            break  # keep only positions where the choice matters
    out = {}
    for name, factory in QUALITY_AGENTS.items():
        agent = factory()
        t0 = time.perf_counter()
        mv = agent.choose(s)
        out[name] = (values[mv] == best, time.perf_counter() - t0)
    return size, best, out


def quality(n_per_size: int = 120) -> None:
    tasks = [(size, 5000 + i) for size in [(4, 5), (5, 5)] for i in range(n_per_size)]
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(quality_task, tasks))
    rows = []
    for size in [(4, 5), (5, 5)]:
        res = [r for r in results if r[0] == size]
        for name in QUALITY_AGENTS:
            ok = [r[2][name][0] for r in res]
            times = [r[2][name][1] for r in res]
            rows.append(
                {
                    "board": f"{size[0]}x{size[1]}",
                    "agent": name,
                    "positions": len(res),
                    "optimal_move_pct": round(100 * sum(ok) / len(ok), 1),
                    "time_ms_mean": round(1000 * sum(times) / len(times), 1),
                }
            )
    write_csv("agents_move_quality.csv", rows)


# ---------------------------------------------------------------------- parity analysis
def parity(n_per_size: int = 350) -> None:
    """Solver-labelled random positions, split by the parity of counters left."""
    rng = random.Random(3)
    solver = Solver()
    counts = defaultdict(lambda: defaultdict(int))
    for size in [(4, 4), (4, 5), (3, 5), (5, 5)]:
        for _ in range(n_per_size):
            s = random_position(*size, rng, removed=(0.2, 0.7))
            if s.is_empty() or s.has_hinger():
                continue
            counts[s.total_counters() % 2][solver.value(s)] += 1
    rows = []
    for par, label in ((1, "odd"), (0, "even")):
        c = counts[par]
        n = sum(c.values())
        rows.append(
            {
                "counters_left": label,
                "positions": n,
                "mover_wins_pct": round(100 * c[1] / n, 1),
                "draw_pct": round(100 * c[0] / n, 1),
                "mover_loses_pct": round(100 * c[-1] / n, 1),
            }
        )
    write_csv("parity_analysis.csv", rows)


# ---------------------------------------------------------------------- tournament
TOURNAMENT = {
    "random": dict(kind="random"),
    "greedy": dict(kind="greedy"),
    "minimax d=2": dict(kind="minimax", depth=2),
    "alphabeta": dict(kind="alphabeta", time_limit=THINK),
    "mcts": dict(kind="mcts", time_limit=THINK),
}


def openings():
    boards = [preset(n) for n in ("cross", "diamond", "staircase", "full5x5")]
    rng = random.Random(77)
    boards += [random_position(5, 5, rng, removed=(0.0, 0.2)) for _ in range(8)]
    return boards


def game_task(args):
    a_name, b_name, board_idx, seed = args
    board = openings()[board_idx]

    def build(name, s):
        spec = dict(TOURNAMENT[name])
        kind = spec.pop("kind")
        if kind in ("random", "greedy", "mcts"):
            spec["seed"] = s
        return make_agent(kind, name, **spec)

    a, b = build(a_name, seed), build(b_name, seed + 1)
    result = play(board, a, b, output=None)
    return a_name, b_name, board_idx, result.winner_side, len(result.moves)


def tournament() -> None:
    names = list(TOURNAMENT)
    n_open = len(openings())
    tasks = []
    seed = 0
    for x, y in itertools.combinations(names, 2):
        for idx in range(n_open):
            tasks.append((x, y, idx, seed))
            tasks.append((y, x, idx, seed + 2))
            seed += 4
    with ProcessPoolExecutor(WORKERS) as pool:
        games = list(pool.map(game_task, tasks, chunksize=2))

    score = defaultdict(float)
    played = defaultdict(int)
    wdl = defaultdict(lambda: [0, 0, 0])
    pair = defaultdict(float)
    for a, b, _idx, side, _n in games:
        for me, opp, my_side in ((a, b, "A"), (b, a, "B")):
            played[me] += 1
            if side is None:
                s, slot = 0.5, 1
            elif side == my_side:
                s, slot = 1.0, 0
            else:
                s, slot = 0.0, 2
            score[me] += s
            pair[(me, opp)] += s
            wdl[me][slot] += 1
    rows = []
    for name in sorted(names, key=lambda n: -score[n] / played[n]):
        row = {
            "agent": name,
            "games": played[name],
            "wins": wdl[name][0],
            "draws": wdl[name][1],
            "losses": wdl[name][2],
            "score_pct": round(100 * score[name] / played[name], 1),
        }
        for opp in names:
            games_vs = 2 * n_open if opp != name else 0
            row[f"vs {opp}"] = round(100 * pair[(name, opp)] / games_vs, 1) if games_vs else ""
        rows.append(row)
    write_csv("agents_tournament.csv", rows)
    a_wins = sum(1 for g in games if g[3] == "A")
    print(f"{len(games)} games; first player won {a_wins}, draws {sum(1 for g in games if g[3] is None)}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("efficiency", "all"):
        efficiency()
    if what in ("quality", "all"):
        quality()
    if what in ("parity", "all"):
        parity()
    if what in ("tournament", "all"):
        tournament()
