"""Turn the benchmark CSVs in reports/tables into the README figures.

python benchmarks/make_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _util import FIGURES, TABLES, read_csv  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
# Categorical slots, fixed order (validated: adjacent CVD dE >= 9.1, normal-vision >= 22.9).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "axes.axisbelow": True,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def compact(x, _pos=None):
    if x >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    if x >= 10_000:
        return f"{x / 1000:g}K"
    if x >= 1:
        return f"{x:,.0f}"
    return f"{x:g}"


def line(ax, x, y, color, label, *, dashed=False, end_label=None, offset=0):
    ax.plot(
        x,
        y,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "-",
        marker="o",
        markersize=6,
        markerfacecolor=color,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=label,
        zorder=3 if not dashed else 4,
    )
    if end_label:
        ax.annotate(
            end_label,
            (x[-1], y[-1]),
            xytext=(8, offset),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK2,
        )


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name}")


# ---------------------------------------------------------------------- figures
def fig_hinger_detection():
    rows = read_csv("hinger_detection.csv")
    cells = [int(r["cells"]) for r in rows]
    fast = [float(r["tarjan_ms"]) for r in rows]
    slow = [float(r["bruteforce_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    line(ax, cells, slow, SLOTS[1], "Remove-and-recount, O(N²)", end_label=f"{slow[-1]:.0f} ms")
    line(ax, cells, fast, SLOTS[0], "Articulation points (Tarjan), O(N)", end_label=f"{fast[-1]:.2f} ms")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(cells[::2] + [cells[-1]])
    ax.set_xticklabels([r["board"] for r in rows][::2] + [rows[-1]["board"]])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("board size (dense random boards)")
    ax.set_ylabel("ms per call (log)")
    ax.set_title(f"Hinger detection: {float(rows[-1]['speedup']):.0f}x faster on a 40x40 board")
    ax.legend(loc="upper left")
    ax.set_xlim(right=cells[-1] * 2.2)
    save(fig, "hinger_detection.png")


def fig_search_unit():
    rows = read_csv("search_unit_cost.csv")
    methods = ["BFS", "DFS", "IDDFS", "A*"]
    colors = dict(zip(methods, SLOTS))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, problem, title in zip(
        axes, ["reachable", "unreachable"], ["Safe path exists", "No safe path (search must prove it)"]
    ):
        series = {}
        for m in methods:
            pts = [
                (int(r["k"]), float(r["expanded_mean"])) for r in rows if r["problem"] == problem and r["method"] == m
            ]
            series[m] = ([p[0] for p in pts], [p[1] for p in pts])
        drawn: dict[tuple, list[str]] = {}
        for m in methods:
            x, y = series[m]
            twin = next((k for k in drawn if list(k) == y), None)
            line(ax, x, y, colors[m], m, dashed=twin is not None)
            drawn.setdefault(tuple(y) if twin is None else twin, []).append(m)
        offsets = iter([6, -6, 0, 0])
        for ys, ms in drawn.items():
            x = series[ms[0]][0]
            ax.annotate(
                " = ".join(ms),
                (x[-1], ys[-1]),
                xytext=(8, next(offsets)),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK2,
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(compact))
        ax.set_xticks(series["BFS"][0])
        ax.set_xlabel("counters to remove (k), 5x5 board")
        ax.set_title(title)
        ax.set_xlim(right=max(series["BFS"][0]) + 3.2)
    axes[0].set_ylabel("states expanded (mean of 30, log)")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Unit-cost safe-path search: every path has length k, the effort differs",
        x=0.06,
        ha="left",
        fontsize=12,
        color=INK,
        fontweight="bold",
        y=1.02,
    )
    save(fig, "search_unit_cost.png")


def fig_search_cost():
    rows = read_csv("search_min_cost.csv")
    methods = ["Dijkstra", "A* (h_count)", "A* (h_pair)"]
    colors = dict(zip(methods, SLOTS))
    fig, ax = plt.subplots(figsize=(7, 4))
    pts = {
        m: [(int(r["k"]), float(r["expanded_mean"])) for r in rows if r["max_counters"] == "2" and r["method"] == m]
        for m in methods
    }
    for m in methods:
        x, y = [p[0] for p in pts[m]], [p[1] for p in pts[m]]
        line(ax, x, y, colors[m], m, dashed=(m == "A* (h_count)"))
    x = [p[0] for p in pts["Dijkstra"]]
    ax.annotate(
        "Dijkstra ≈ A* (h_count)",
        (x[-1], pts["Dijkstra"][-1][1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK2,
    )
    ratio = pts["Dijkstra"][-1][1] / pts["A* (h_pair)"][-1][1]
    ax.annotate(
        f"A* (h_pair): {ratio:.0f}x fewer",
        (x[-1], pts["A* (h_pair)"][-1][1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK2,
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    ax.set_xticks(x)
    ax.set_xlim(right=x[-1] + 6.5)
    ax.set_xlabel("counters to remove (k), 5x5 board with piles of 1-2")
    ax.set_ylabel("states expanded (mean of 30, log)")
    ax.set_title("Minimum-cost safe path: same optimal cost, far less search")
    ax.legend(loc="upper left")
    save(fig, "search_min_cost.png")


def fig_agents_efficiency():
    rows = read_csv("agents_efficiency.csv")
    variants = ["minimax", "alpha-beta", "+ move ordering", "+ transposition table"]
    colors = dict(zip(variants, SLOTS))
    fig, ax = plt.subplots(figsize=(7, 4))
    for v in variants:
        pts = [(int(r["depth"]), float(r["nodes_mean"])) for r in rows if r["variant"] == v]
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        line(ax, x, y, colors[v], v, end_label=compact(y[-1]))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    ax.set_xticks(range(1, 7))
    ax.set_xlim(right=6.8)
    ax.set_xlabel("search depth (plies)")
    ax.set_ylabel("nodes searched (mean of 40 positions, log)")
    ax.set_title("Same minimax value, fewer nodes (5x5 mid-game positions)")
    ax.legend(loc="upper left")
    save(fig, "agents_efficiency.png")


def _hbars(ax, labels, values, color, fmt):
    ys = range(len(labels))
    ax.barh(list(ys), values, height=0.55, color=color)
    for y, v in zip(ys, values):
        ax.annotate(fmt(v), (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def fig_agents_quality():
    rows = read_csv("agents_move_quality.csv")
    agents = list(dict.fromkeys(r["agent"] for r in rows))
    boards = list(dict.fromkeys(r["board"] for r in rows))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    h = 0.36
    for j, b in enumerate(boards):
        vals = [float(next(r for r in rows if r["agent"] == a and r["board"] == b)["optimal_move_pct"]) for a in agents]
        ys = [i + (j - 0.5) * (h + 0.04) for i in range(len(agents))]
        ax.barh(ys, vals, height=h, color=SLOTS[j], label=f"{b} board")
        for y, v in zip(ys, vals):
            ax.annotate(
                f"{v:.0f}%", (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK2
            )
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 108)
    ax.set_xlabel("positions where the agent's move is game-theoretically optimal (%)")
    ax.set_title("Move quality graded by the exact solver (0.2 s per move)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "agents_move_quality.png")


def fig_tournament():
    rows = read_csv("agents_tournament.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    _hbars(ax, [r["agent"] for r in rows], [float(r["score_pct"]) for r in rows], SLOTS[0], lambda v: f"{v:.1f}%")
    ax.set_xlim(0, 100)
    ax.set_xlabel("tournament score (win = 1, draw = 0.5), %")
    games = rows[0]["games"]
    ax.set_title(f"Round robin: {games} games per agent, 0.2 s per move, 12 openings x both colours")
    save(fig, "agents_tournament.png")


if __name__ == "__main__":
    fig_hinger_detection()
    fig_search_unit()
    fig_search_cost()
    if (TABLES / "agents_efficiency.csv").exists():
        fig_agents_efficiency()
    if (TABLES / "agents_move_quality.csv").exists():
        fig_agents_quality()
    if (TABLES / "agents_tournament.csv").exists():
        fig_tournament()
