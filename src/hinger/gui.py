"""Pygame front end: play Hinger against any agent, or watch two agents play.

Controls: click a cell to remove a counter. ``H`` toggles hints (hinger cells
glow amber, moves that would hand the opponent a hinger are marked red), ``U``
undoes your last move, ``N`` starts a new game, ``Esc`` quits.

The AI thinks on a background thread so the window stays responsive. Drawing
is separated from the event loop (``HingerView.draw``), which lets
``scripts/make_demo_gif.py`` render frames without opening a window.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from .agents import Agent, make_agent  # noqa: E402
from .state import Move, State  # noqa: E402

# Palette
BG = (24, 26, 33)
PANEL = (32, 35, 45)
EMPTY = (44, 47, 60)
GRID_LINE = (58, 62, 78)
TEXT = (232, 234, 242)
MUTED = (150, 155, 175)
ACCENT = (98, 140, 255)
PILE_COLORS = [(82, 128, 255), (122, 98, 255), (172, 84, 240), (214, 78, 200)]
HINGER = (255, 176, 32)
DANGER = (255, 92, 92)
WIN_GREEN = (80, 210, 140)

PANEL_W = 300
MARGIN = 28


@dataclass
class HingerView:
    """Everything needed to draw one frame."""

    state: State
    names: tuple[str, str]
    turn: int = 0  # 0 = player A to move
    last: Move | None = None
    winning: Move | None = None  # the hinger that ended the game
    hints: bool = False
    hover: Move | None = None
    status: str = ""
    info: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    over: bool = False


class Renderer:
    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.cell = max(40, min(92, 560 // max(rows, cols)))
        board_w = cols * self.cell + 2 * MARGIN
        board_h = rows * self.cell + 2 * MARGIN + 40
        self.size = (board_w + PANEL_W, max(board_h, 560))
        self.board_origin = (MARGIN, MARGIN + 40)
        pygame.font.init()
        self.f_title = pygame.font.Font(None, 44)
        self.f_big = pygame.font.Font(None, max(28, int(self.cell * 0.55)))
        self.f_med = pygame.font.Font(None, 28)
        self.f_small = pygame.font.Font(None, 22)

    def cell_at(self, pos: tuple[int, int]) -> Move | None:
        x0, y0 = self.board_origin
        c, r = (pos[0] - x0) // self.cell, (pos[1] - y0) // self.cell
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return int(r), int(c)
        return None

    def _text(self, surf, font, text, color, pos, anchor="topleft"):
        img = font.render(text, True, color)
        rect = img.get_rect(**{anchor: pos})
        surf.blit(img, rect)
        return rect

    def draw(self, surf: pygame.Surface, v: HingerView) -> None:
        surf.fill(BG)
        s = v.state
        x0, y0 = self.board_origin
        cs = self.cell
        for c in range(s.cols):
            self._text(surf, self.f_small, str(c), MUTED, (x0 + c * cs + cs // 2, y0 - 8), "midbottom")
        for r in range(s.rows):
            self._text(surf, self.f_small, str(r), MUTED, (x0 - 8, y0 + r * cs + cs // 2), "midright")

        hingers = s.hingers() if (v.hints and not v.over) else frozenset()
        unsafe: set[Move] = set()
        if v.hints and not v.over and not s.has_hinger():
            unsafe = {mv for mv, child in s.successors() if child.has_hinger()}

        for r in range(s.rows):
            for c in range(s.cols):
                rect = pygame.Rect(x0 + c * cs + 3, y0 + r * cs + 3, cs - 6, cs - 6)
                n = s[(r, c)]
                if (r, c) == v.winning:
                    color = HINGER
                elif n:
                    color = PILE_COLORS[min(n, len(PILE_COLORS)) - 1]
                    if (r, c) == v.hover and not v.over:
                        color = tuple(min(255, x + 40) for x in color)
                else:
                    color = EMPTY
                pygame.draw.rect(surf, color, rect, border_radius=12)
                if n:
                    self._text(surf, self.f_big, str(n), TEXT, rect.center, "center")
                if (r, c) in hingers and (r, c) != v.winning:
                    pygame.draw.rect(surf, HINGER, rect.inflate(2, 2), width=4, border_radius=13)
                if (r, c) in unsafe:
                    pygame.draw.circle(surf, DANGER, (rect.right - 10, rect.top + 10), 6)
                if (r, c) == v.last:
                    pygame.draw.rect(surf, TEXT, rect.inflate(-10, -10), width=2, border_radius=9)

        # Side panel
        px = self.size[0] - PANEL_W
        pygame.draw.rect(surf, PANEL, (px, 0, PANEL_W, self.size[1]))
        y = 22
        self._text(surf, self.f_title, "HINGER", TEXT, (px + 22, y))
        y += 50
        for i, name in enumerate(v.names):
            mark = ">" if (i == v.turn and not v.over) else " "
            col = TEXT if i == v.turn and not v.over else MUTED
            self._text(surf, self.f_med, f"{mark} {'AB'[i]}: {name}", col, (px + 22, y))
            y += 28
        y += 10
        for line in _wrap(self.f_med, v.status, PANEL_W - 40):
            self._text(surf, self.f_med, line, WIN_GREEN if v.over else ACCENT, (px + 22, y))
            y += 26
        y += 10
        stats = [f"counters  {s.total_counters()}", f"regions   {s.num_regions()}"]
        if v.hints:
            stats.append(f"hingers   {s.num_hingers()}")
        for line in stats + v.info:
            self._text(surf, self.f_small, line, MUTED, (px + 22, y))
            y += 22
        y += 12
        self._text(surf, self.f_small, "moves", TEXT, (px + 22, y))
        y += 22
        bottom = self.size[1] - (110 if v.hints else 60)
        room = max(1, (bottom - y) // 20)
        for line in v.history[-room:]:
            self._text(surf, self.f_small, line, MUTED, (px + 22, y))
            y += 20
        if v.hints:
            ly = self.size[1] - 98
            pygame.draw.rect(surf, HINGER, (px + 22, ly + 2, 14, 14), width=3, border_radius=3)
            self._text(surf, self.f_small, "hinger: take it to win", MUTED, (px + 44, ly))
            pygame.draw.circle(surf, DANGER, (px + 29, ly + 31), 6)
            self._text(surf, self.f_small, "gives the opponent a hinger", MUTED, (px + 44, ly + 23))
        hint_txt = "[H] hints on" if v.hints else "[H] hints off"
        self._text(surf, self.f_small, f"{hint_txt}   [U] undo", MUTED, (px + 22, self.size[1] - 48))
        self._text(surf, self.f_small, "[N] new game   [Esc] quit", MUTED, (px + 22, self.size[1] - 26))


class HingerApp:
    """Interactive game: each side is a human (``None``) or an agent."""

    def __init__(self, board: State, players: tuple[Agent | None, Agent | None]):
        self.initial = board
        self.players = players
        names = tuple(p.name if p is not None else "You" for p in players)
        self.view = HingerView(board, names)  # type: ignore[arg-type]
        self.renderer = Renderer(board.rows, board.cols)
        self.moves: list[Move] = []
        self._thinking: threading.Thread | None = None
        self._pending: Move | None = None
        self._generation = 0  # invalidates AI results after undo/new game
        self.new_game()

    # -- game logic ---------------------------------------------------------
    def new_game(self) -> None:
        self._generation += 1
        self.moves = []
        self.view.info = []
        for p in self.players:
            if p is not None:
                p.reset()
        self._rebuild()

    def _rebuild(self) -> None:
        """Recompute the view by replaying ``self.moves`` from the start."""
        v = self.view
        s = self.initial
        v.history = []
        v.winning, v.over, v.last = None, False, None
        for i, mv in enumerate(self.moves):
            hinger = s.is_hinger(mv)
            cost = s.move_cost(mv)
            s = s.apply(mv)
            v.history.append(f"{i + 1:>2}. {'AB'[i % 2]} {mv}  cost {cost}" + ("  HINGER" if hinger else ""))
            v.last = mv
            if hinger:
                v.winning, v.over = mv, True
        v.state = s
        v.turn = len(self.moves) % 2
        if not v.over and s.is_empty():
            v.over = True
        self._update_status()

    def _update_status(self) -> None:
        v = self.view
        if v.over:
            if v.winning is None:
                v.status = "Draw - board cleared"
            else:
                winner = 1 - v.turn  # the player who just moved
                v.status = "You win!" if self.players[winner] is None else f"{v.names[winner]} wins"
        elif self.players[v.turn] is None:
            v.status = "Your move"
        else:
            v.status = f"{v.names[v.turn]} is thinking..."

    def play_move(self, mv: Move) -> None:
        if self.view.over or not self.view.state.is_legal(mv):
            return
        self.moves.append(mv)
        self._rebuild()

    def undo(self) -> None:
        if self._thinking is not None:
            return
        # Undo back to the human's previous turn.
        while self.moves:
            self.moves.pop()
            if self.players[len(self.moves) % 2] is None:
                break
        self._generation += 1
        self._rebuild()

    def _maybe_start_ai(self) -> None:
        v = self.view
        agent = self.players[v.turn]
        if v.over or agent is None or self._thinking is not None:
            return
        state, gen = v.state, self._generation

        def work():
            mv = agent.choose(state)
            if gen == self._generation:
                self._pending = mv
            self._thinking = None

        self._thinking = threading.Thread(target=work, daemon=True)
        self._thinking.start()

    def _collect_ai(self) -> None:
        if self._pending is not None:
            mv, self._pending = self._pending, None
            agent = self.players[self.view.turn]
            self.play_move(mv)
            if agent is not None:
                self.view.info = _describe(agent)

    # -- main loop ------------------------------------------------------------
    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode(self.renderer.size)
        pygame.display.set_caption("Hinger")
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_h:
                        self.view.hints = not self.view.hints
                    elif event.key == pygame.K_n:
                        self.new_game()
                    elif event.key == pygame.K_u:
                        self.undo()
                elif event.type == pygame.MOUSEMOTION:
                    self.view.hover = self.renderer.cell_at(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mv = self.renderer.cell_at(event.pos)
                    if mv is not None and self.players[self.view.turn] is None and self._thinking is None:
                        self.play_move(mv)
            self._collect_ai()
            self._maybe_start_ai()
            self.renderer.draw(screen, self.view)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


def _wrap(font: pygame.font.Font, text: str, width: int) -> list[str]:
    """Greedy word wrap to a pixel width."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and font.size(trial)[0] > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


def _describe(agent: Agent) -> list[str]:
    st = agent.stats
    out = []
    if "iterations" in st:
        out.append(f"AI: {st['iterations']:,} playouts")
    if "nodes" in st:
        out.append(f"AI: depth {st.get('depth')}, {st['nodes']:,} nodes")
    if "time" in st:
        out.append(f"AI: {st['time']:.2f}s")
    if "by" in st:
        out.append(f"AI: decided by {st['by']}")
    if "win_rate" in st:
        out.append(f"AI: est. win {st['win_rate']:.0%}")
    return out


def run_gui(
    board: State,
    *,
    opponent: str = "mcts",
    human_first: bool = True,
    think: float = 1.0,
    depth: int = 3,
    seed: int | None = None,
) -> None:
    kwargs: dict = {"depth": depth} if opponent == "minimax" else {}
    if opponent in ("alphabeta", "mcts"):
        kwargs["time_limit"] = think
    if opponent in ("random", "greedy", "mcts"):
        kwargs["seed"] = seed
    ai = make_agent(opponent, **kwargs)
    players = (None, ai) if human_first else (ai, None)
    HingerApp(board, players).run()
