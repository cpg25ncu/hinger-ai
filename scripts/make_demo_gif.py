"""Render the README demo: two agents play in the GUI, captured frame by frame.

    python scripts/make_demo_gif.py            # -> reports/figures/demo.gif, gui.png

Runs headless (no window). Needs the gui and bench extras (pygame, Pillow via matplotlib).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from hinger.agents import AlphaBetaAgent, MCTSAgent  # noqa: E402
from hinger.gui import HingerApp, _describe  # noqa: E402
from hinger.scenarios import preset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def snapshot(app: HingerApp) -> Image.Image:
    surf = pygame.Surface(app.renderer.size)
    app.renderer.draw(surf, app.view)
    return Image.frombytes("RGB", surf.get_size(), pygame.image.tobytes(surf, "RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="cross")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "reports" / "figures" / "demo.gif"))
    args = ap.parse_args()

    pygame.init()
    a = MCTSAgent("MCTS", time_limit=1.0, seed=args.seed)
    b = AlphaBetaAgent("Alpha-beta", time_limit=0.5)
    app = HingerApp(preset(args.board), (a, b))
    app.view.hints = True

    frames: list[Image.Image] = []
    durations: list[int] = []

    def grab(ms: int) -> None:
        frames.append(snapshot(app))
        durations.append(ms)

    app._update_status()
    grab(1200)
    screenshot_at = None
    while not app.view.over:
        agent = app.players[app.view.turn]
        mv = agent.choose(app.view.state)
        app.play_move(mv)
        app.view.info = _describe(agent)
        grab(700)
        if screenshot_at is None and len(app.moves) >= 6:
            screenshot_at = len(frames) - 1
    durations[-1] = 3500

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    frames[screenshot_at or -1].save(out.with_name("gui.png"))
    print(f"{len(frames)} frames -> {out} ({out.stat().st_size / 1024:.0f} KB); {app.view.status}")


if __name__ == "__main__":
    main()
