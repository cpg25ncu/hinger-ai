"""Search and game-playing AI for Hinger, a two-player combinatorial board game."""

from .agents import (
    AlphaBetaAgent,
    GreedyAgent,
    MCTSAgent,
    MinimaxAgent,
    RandomAgent,
    SolverAgent,
    make_agent,
)
from .game import GameResult, play
from .search import SearchResult, find_path, min_safe, path_astar, path_bfs, path_dfs, path_iddfs
from .solver import Solver, win
from .state import Move, State

__version__ = "1.0.0"

__all__ = [
    "AlphaBetaAgent",
    "GameResult",
    "GreedyAgent",
    "MCTSAgent",
    "MinimaxAgent",
    "Move",
    "RandomAgent",
    "SearchResult",
    "Solver",
    "SolverAgent",
    "State",
    "find_path",
    "make_agent",
    "min_safe",
    "path_astar",
    "path_bfs",
    "path_dfs",
    "path_iddfs",
    "play",
    "win",
]
