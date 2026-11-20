"""Game-playing agents for Hinger.

Every agent exposes ``choose(state) -> move`` (``None`` when the board is empty)
and a ``name``. All of them take an immediate win when a hinger is on the board.

* ``RandomAgent``     uniform random legal move (baseline).
* ``GreedyAgent``     wins if it can, otherwise a random move that does not hand
                      the opponent a hinger.
* ``MinimaxAgent``    full-width depth-limited minimax (negamax form).
* ``AlphaBetaAgent``  alpha-beta with iterative deepening, a time limit, a
                      transposition table and domain-specific move ordering.
* ``MCTSAgent``       Monte Carlo tree search (UCT) with greedy-safe rollouts.
* ``SolverAgent``     plays perfectly using the exact solver (small boards only).

Scores are from the point of view of the player to move. A win found ``p``
plies from the root scores ``WIN_SCORE - p`` so that faster wins (and slower
losses) are preferred.

**Dominated moves.** A move that creates a hinger hands the opponent an
immediate win, the fastest possible loss. If any other move exists it is at
least as good, so the searching agents drop hinger-creating moves whenever a
safe alternative is available. This never changes a position's minimax value.

**Evaluation.** At the depth limit a position is scored by the parity of the
counters left (``parity_eval``). Labelling 1,400 random mid-game positions with
the exact solver showed the player to move wins 75 % of positions with an odd
number of counters but only 19 % of those with an even number. Parity cannot
rank sibling moves, though: every move removes one counter, so all leaves of a
fixed-depth search have the same parity. Depth-limited alpha-beta is therefore
strong only where it can see forced results, and MCTS, whose rollouts estimate
the outcome directly, is the stronger player in the opening and middle game.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Iterator

from .solver import Solver
from .state import Move, State

WIN_SCORE = 1_000_000
MATE_BOUND = WIN_SCORE - 10_000  # scores beyond this are proven wins/losses
INF = 10 * WIN_SCORE
PARITY_WEIGHT = 100

Evaluator = Callable[[State], int]


def parity_eval(state: State) -> int:
    """+PARITY_WEIGHT if the player to move faces an odd number of counters."""
    return PARITY_WEIGHT if state.total_counters() % 2 else -PARITY_WEIGHT


def zero_eval(state: State) -> int:
    """No knowledge: only proven wins, losses and draws are scored."""
    return 0


def mobility_eval(state: State) -> int:
    """Parity plus a small bonus per safe move available to the player to move."""
    safe = sum(1 for _mv, child in state.successors() if not child.has_hinger())
    return parity_eval(state) + safe


EVALUATORS: dict[str, Evaluator] = {"parity": parity_eval, "zero": zero_eval, "mobility": mobility_eval}


def safe_children(state: State) -> tuple[list[tuple[Move, State]], list[tuple[Move, State]]]:
    """Split successors into (safe, hinger-creating)."""
    safe, unsafe = [], []
    for mv, child in state.successors():
        (unsafe if child.has_hinger() else safe).append((mv, child))
    return safe, unsafe


class SearchTimeout(Exception):
    pass


class Agent:
    """Base class: subclasses implement ``_choose`` for positions without a hinger."""

    kind = "agent"

    def __init__(self, name: str | None = None):
        self.name = name or self.kind
        self.stats: dict[str, object] = {}

    def choose(self, state: State) -> Move | None:
        if state.is_empty():
            return None
        if state.has_hinger():
            self.stats = {"reason": "immediate win"}
            return min(state.hingers())
        return self._choose(state)

    def _choose(self, state: State) -> Move:
        raise NotImplementedError

    def reset(self) -> None:
        """Forget anything cached from a previous game."""

    def __str__(self) -> str:
        return f"{self.name} ({self.kind})"

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class RandomAgent(Agent):
    kind = "random"

    def __init__(self, name: str | None = None, *, seed: int | None = None):
        super().__init__(name)
        self.rng = random.Random(seed)

    def choose(self, state: State) -> Move | None:
        moves = state.legal_moves()
        return self.rng.choice(moves) if moves else None


class GreedyAgent(Agent):
    kind = "greedy"

    def __init__(self, name: str | None = None, *, seed: int | None = None):
        super().__init__(name)
        self.rng = random.Random(seed)

    def _choose(self, state: State) -> Move:
        safe, unsafe = safe_children(state)
        return self.rng.choice(safe or unsafe)[0]


# ---------------------------------------------------------------------- minimax family
def _to_tt(score: int, ply: int) -> int:
    """Store proven scores relative to the node, not the root."""
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


EXACT, LOWER, UPPER = 0, 1, 2


class MinimaxAgent(Agent):
    """Plain depth-limited minimax: every move searched, no pruning."""

    kind = "minimax"

    def __init__(self, name: str | None = None, *, depth: int = 3, evaluator: str = "parity"):
        super().__init__(name)
        self.depth = depth
        self.evaluate = EVALUATORS[evaluator]
        self.nodes = 0

    def search(self, state: State, depth: int | None = None) -> tuple[int, Move | None]:
        """Return (score, best move) for the player to move."""
        self.nodes = 0
        return self._negamax(state, self.depth if depth is None else depth, 0)

    def _negamax(self, s: State, depth: int, ply: int) -> tuple[int, Move | None]:
        self.nodes += 1
        if s.has_hinger():
            return WIN_SCORE - ply, min(s.hingers())
        if s.is_empty():
            return 0, None
        if depth == 0:
            return self.evaluate(s), None
        best, best_mv = -INF, None
        for mv, child in s.successors():
            v = -self._negamax(child, depth - 1, ply + 1)[0]
            if v > best:
                best, best_mv = v, mv
        return best, best_mv

    def _choose(self, state: State) -> Move:
        t0 = time.perf_counter()
        score, mv = self.search(state)
        self.stats = {"score": score, "depth": self.depth, "nodes": self.nodes, "time": time.perf_counter() - t0}
        return mv


class AlphaBetaAgent(Agent):
    """Alpha-beta negamax with iterative deepening.

    ``ordering`` enables safe-first move ordering and dominated-move pruning;
    ``use_tt`` enables the transposition table, whose stored best move is also
    tried first at the next iteration. With ``time_limit`` set, the deepest
    fully completed iteration decides the move.
    """

    kind = "alphabeta"

    def __init__(
        self,
        name: str | None = None,
        *,
        depth: int = 64,
        time_limit: float | None = 1.0,
        use_tt: bool = True,
        ordering: bool = True,
        evaluator: str = "parity",
        max_tt_entries: int = 2_000_000,
    ):
        super().__init__(name)
        self.depth = depth
        self.time_limit = time_limit
        self.use_tt = use_tt
        self.ordering = ordering
        self.evaluate = EVALUATORS[evaluator]
        self.max_tt_entries = max_tt_entries
        self.tt: dict[State, tuple[int, int, int, Move | None]] = {}
        self.nodes = 0
        self._deadline: float | None = None

    def reset(self) -> None:
        self.tt.clear()

    # -- search -------------------------------------------------------------
    def search(self, state: State, depth: int) -> tuple[int, Move | None]:
        """One fixed-depth search (no time limit). Returns (score, best move)."""
        self.nodes = 0
        self._deadline = None
        return self._root(state, depth)

    def _root(self, state: State, depth: int) -> tuple[int, Move | None]:
        best, best_mv = -INF, None
        alpha, beta = -INF, INF
        for mv, child in self._children(state):
            v = -self._ab(child, depth - 1, -beta, -alpha, 1)
            if v > best:
                best, best_mv = v, mv
            if v > alpha:
                alpha = v
        if self.use_tt:
            self._store(state, depth, best, EXACT, best_mv, 0)
        return best, best_mv

    def _children(self, s: State) -> Iterator[tuple[Move, State]]:
        """Children in search order, generated lazily.

        With ordering on: the transposition table's best move first, then the
        safe moves; hinger-creating moves are held back and only searched if no
        safe move exists (they are dominated otherwise). Being lazy matters: a
        child's hinger check is the expensive part of generating it, and after
        a beta cut-off the remaining children are never checked at all.
        """
        if not self.ordering:
            yield from s.successors()
            return
        tt_move = None
        if self.use_tt:
            entry = self.tt.get(s)
            if entry is not None:
                tt_move = entry[3]
        any_safe = False
        held_back: list[tuple[Move, State]] = []
        if tt_move is not None:
            child = s.apply(tt_move)
            if child.has_hinger():
                held_back.append((tt_move, child))
            else:
                any_safe = True
                yield tt_move, child
        for mv, child in s.successors():
            if mv == tt_move:
                continue
            if child.has_hinger():
                held_back.append((mv, child))
            else:
                any_safe = True
                yield mv, child
        if not any_safe:
            yield from held_back

    def _store(self, s: State, depth: int, score: int, flag: int, mv: Move | None, ply: int) -> None:
        if len(self.tt) >= self.max_tt_entries:
            self.tt.clear()
        self.tt[s] = (depth, _to_tt(score, ply), flag, mv)

    def _ab(self, s: State, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self._deadline is not None and not (self.nodes & 1023) and time.perf_counter() > self._deadline:
            raise SearchTimeout
        if s.has_hinger():
            return WIN_SCORE - ply
        if s.is_empty():
            return 0
        if depth <= 0:
            return self.evaluate(s)
        alpha0 = alpha
        if self.use_tt:
            entry = self.tt.get(s)
            if entry is not None and entry[0] >= depth:
                v = _from_tt(entry[1], ply)
                flag = entry[2]
                if flag == EXACT:
                    return v
                if flag == LOWER and v > alpha:
                    alpha = v
                elif flag == UPPER and v < beta:
                    beta = v
                if alpha >= beta:
                    return v
        best, best_mv = -INF, None
        for mv, child in self._children(s):
            v = -self._ab(child, depth - 1, -beta, -alpha, ply + 1)
            if v > best:
                best, best_mv = v, mv
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        if self.use_tt:
            flag = UPPER if best <= alpha0 else LOWER if best >= beta else EXACT
            self._store(s, depth, best, flag, best_mv, ply)
        return best

    def _choose(self, state: State) -> Move:
        t0 = time.perf_counter()
        self.nodes = 0
        self._deadline = t0 + self.time_limit if self.time_limit else None
        best_mv, best_score, reached = None, 0, 0
        max_depth = min(self.depth, state.total_counters())
        for d in range(1, max_depth + 1):
            try:
                score, mv = self._root(state, d)
            except SearchTimeout:
                break
            best_mv, best_score, reached = mv, score, d
            if abs(score) >= MATE_BOUND:
                break  # the result is proven; deeper search cannot change it
        self._deadline = None
        if best_mv is None:  # not even depth 1 finished in time
            safe, unsafe = safe_children(state)
            best_mv = (safe or unsafe)[0][0]
        self.stats = {
            "score": best_score,
            "depth": reached,
            "nodes": self.nodes,
            "time": time.perf_counter() - t0,
            "proven": abs(best_score) >= MATE_BOUND or reached == state.total_counters(),
        }
        return best_mv


# ---------------------------------------------------------------------- Monte Carlo tree search
class _Node:
    __slots__ = ("state", "parent", "move", "children", "untried", "visits", "reward")

    def __init__(self, state: State, parent: "_Node | None", move: Move | None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: list[_Node] = []
        if state.has_hinger() or state.is_empty():
            self.untried: list[tuple[Move, State]] = []  # terminal
        else:
            safe, unsafe = safe_children(state)
            self.untried = safe or unsafe
        self.visits = 0
        self.reward = 0.0  # total reward for the player who moved INTO this node


class MCTSAgent(Agent):
    """UCT Monte Carlo tree search.

    Rollouts play a random *safe* move whenever one exists and take a hinger
    when one appears, which is far more realistic than uniform random play.
    Wins count 1, draws 0.5.
    """

    kind = "mcts"

    def __init__(
        self,
        name: str | None = None,
        *,
        time_limit: float | None = 1.0,
        iterations: int | None = None,
        c: float = 1.4,
        seed: int | None = None,
    ):
        super().__init__(name)
        if time_limit is None and iterations is None:
            raise ValueError("give MCTSAgent a time_limit or an iteration budget")
        self.time_limit = time_limit
        self.iterations = iterations
        self.c = c
        self.rng = random.Random(seed)

    def _rollout(self, state: State) -> float:
        """Reward for the player to move at ``state`` (1 win, 0.5 draw, 0 loss)."""
        rng = self.rng
        s = state
        me = True
        while True:
            if s.has_hinger():
                return 1.0 if me else 0.0
            if s.is_empty():
                return 0.5
            moves = s.legal_moves()
            rng.shuffle(moves)
            nxt = None
            for mv in moves:
                cand = s.apply(mv)
                if not cand.has_hinger():
                    nxt = cand
                    break
            s = nxt if nxt is not None else s.apply(moves[0])
            me = not me

    def _choose(self, state: State) -> Move:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit if self.time_limit else None
        root = _Node(state, None, None)
        if len(root.untried) == 1:
            self.stats = {"iterations": 0, "time": 0.0, "reason": "only one sensible move"}
            return root.untried[0][0]
        c, log = self.c, math.log
        it = 0
        while True:
            if self.iterations is not None and it >= self.iterations:
                break
            if deadline is not None and time.perf_counter() > deadline:
                break
            it += 1
            node = root
            # Selection
            while not node.untried and node.children:
                ln = log(node.visits)
                node = max(
                    node.children,
                    key=lambda ch: ch.reward / ch.visits + c * math.sqrt(ln / ch.visits),
                )
            # Expansion
            if node.untried:
                mv, child_state = node.untried.pop(self.rng.randrange(len(node.untried)))
                child = _Node(child_state, node, mv)
                node.children.append(child)
                node = child
            # Simulation: reward for the player to move at `node`
            reward = self._rollout(node.state)
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.reward += 1.0 - reward
                reward = 1.0 - reward
                node = node.parent
        best = max(root.children, key=lambda ch: ch.visits)
        self.stats = {
            "iterations": it,
            "time": time.perf_counter() - t0,
            "win_rate": best.reward / best.visits,
        }
        return best.move


# ---------------------------------------------------------------------- perfect play
class SolverAgent(Agent):
    """Perfect play from the exact solver. Exponential: small boards only."""

    kind = "solver"

    def __init__(self, name: str | None = None):
        super().__init__(name)
        self.solver = Solver()

    def reset(self) -> None:
        self.solver = Solver()

    def _choose(self, state: State) -> Move:
        t0 = time.perf_counter()
        mv = self.solver.best_move(state)
        self.stats = {"value": self.solver.value(state), "time": time.perf_counter() - t0}
        return mv


AGENTS: dict[str, type[Agent]] = {
    cls.kind: cls for cls in (RandomAgent, GreedyAgent, MinimaxAgent, AlphaBetaAgent, MCTSAgent, SolverAgent)
}


def make_agent(kind: str, name: str | None = None, **kwargs) -> Agent:
    """Build an agent by kind (``random``, ``greedy``, ``minimax``, ``alphabeta``, ``mcts``, ``solver``)."""
    try:
        cls = AGENTS[kind]
    except KeyError:
        raise ValueError(f"unknown agent {kind!r}; choose from {sorted(AGENTS)}") from None
    return cls(name, **kwargs)
