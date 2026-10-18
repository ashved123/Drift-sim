"""Training dashboard tabs: Live, Progress, Learning curves, Episodes.

Each tab is a ttk.Frame with `update(run, changed, force)`: `run` is a
RunData (or None), `changed` says which of its files got new data since the
last poll, `force` redraws regardless (tab just shown / run switched).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from src.ui import theme as T
from src.ui.run_data import RunData

PARTS = [("progress", "Progress", T.S1), ("offtrack", "Off-track", T.S2), ("time", "Time", T.S3),
         ("damage", "Damage", T.S4), ("drift", "Drift bonus", T.S5)]
PPO_STATS = [("entropy_loss", "Entropy loss"), ("approx_kl", "Approx KL"), ("clip_fraction", "Clip fraction"),
             ("explained_variance", "Explained variance"), ("value_loss", "Value loss"),
             ("policy_gradient_loss", "Policy loss")]
STEP_SECONDS = 0.05  # 20 Hz control


class FigureTab(ttk.Frame):
    def __init__(self, parent, figsize=(14, 7)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, dpi=90, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)


class LiveTab(FigureTab):
    """The episode being driven right now, ~twice a second."""

    def __init__(self, parent):
        super().__init__(parent)
        self.info = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.info, padding=(10, 6), font=("Segoe UI", 11)).pack(side="top", anchor="w")
        gs = self.fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15])
        f = self.fig
        self.ax = {"speed": f.add_subplot(gs[0, 0]), "slip": f.add_subplot(gs[0, 1]),
                   "controls": f.add_subplot(gs[1, 0]), "lateral": f.add_subplot(gs[1, 1]),
                   "map": f.add_subplot(gs[:, 2])}
        ax = self.ax
        self.lines = {
            "speed": ax["speed"].plot([], [], color=T.S1)[0],
            "slip": ax["slip"].plot([], [], color=T.S1)[0],
            "steer": ax["controls"].plot([], [], color=T.S1, label="Steering (−1 left … +1 right)")[0],
            "throttle": ax["controls"].plot([], [], color=T.S2, label="Throttle (0–1)")[0],
            "lateral": ax["lateral"].plot([], [], color=T.S1)[0],
        }
        for key, title, ylabel in [("speed", "Speed", "m/s"), ("slip", "Slip angle", "degrees"),
                                   ("controls", "Steering & throttle", ""),
                                   ("lateral", "Distance from centerline", "m (+ = left)")]:
            ax[key].set_title(title)
            ax[key].set_ylabel(ylabel)
            ax[key].set_xlabel("seconds ago")
        ax["controls"].set_ylim(-1.1, 1.1)
        ax["controls"].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
        ax["lateral"].axhline(0, color=T.AXIS, lw=1)
        ax["map"].set_title("Current track (car trail, last ~15 s)")
        T.clear_map_axes(ax["map"])
        self.map_seed = None
        self.trail = ax["map"].plot([], [], color=T.S1, lw=2, zorder=2)[0]
        self.car = ax["map"].plot([], [], "o", color=T.S2, ms=8, mec=T.SURFACE, mew=2, zorder=4)[0]
        self.marks = []

    def update(self, run: RunData | None, changed: dict, force: bool = False) -> None:
        st = (run.status if run else None) or {}
        cur, live = st.get("current"), st.get("live") or []
        if run and cur and run.is_live():
            self.info.set(f"Episode in progress · track {cur['track_seed']} · {cur['laps']:.2f} laps · "
                          f"{cur['speed']:.1f} m/s · reward so far {cur.get('reward', 0):.0f} · "
                          f"off road {cur.get('offtrack_pct', 0):.0f}% of the time")
        else:
            self.info.set("No episode running right now — showing the last data received." if live
                          else "No live data yet — start a training run.")
        if not live:
            self.canvas.draw_idle()
            return
        a = np.array([row[:8] for row in live], dtype=float)
        seeds = [row[8] for row in live]
        secs = (a[:, 0] - a[-1, 0]) * STEP_SECONDS
        for key, col in [("speed", 1), ("slip", 2), ("steer", 3), ("throttle", 4), ("lateral", 5)]:
            self.lines[key].set_data(secs, a[:, col])
        for key, col in [("speed", 1), ("slip", 2), ("lateral", 5)]:
            lo, hi = float(a[:, col].min()), float(a[:, col].max())
            pad = max(1.0, 0.1 * (hi - lo))
            self.ax[key].set_ylim(lo - pad, hi + pad)
        for m in self.marks:
            m.remove()
        self.marks = []
        starts = [i for i in range(1, len(seeds)) if seeds[i] != seeds[i - 1]]
        for key in ("speed", "slip", "controls", "lateral"):
            self.ax[key].set_xlim(min(secs[0], -1.0), 0)
            for i in starts:  # episode reset: car back on a new start line
                self.marks.append(self.ax[key].axvline(secs[i], color=T.MUTED, lw=1))
        if starts:
            self.marks.append(self.ax["speed"].annotate(
                "new episode", (secs[starts[-1]], 1), xycoords=("data", "axes fraction"),
                xytext=(3, -10), textcoords="offset points", color=T.MUTED, fontsize=8))

        m = self.ax["map"]
        if run.track and run.track.get("track_seed") != self.map_seed:
            for line in [ln for ln in m.lines if ln not in (self.trail, self.car)]:
                line.remove()
            for text in list(m.texts):
                text.remove()
            T.draw_road(m, run.track)
            self.map_seed = run.track.get("track_seed")
        same = [i for i, s in enumerate(seeds) if s == self.map_seed]
        if same:
            self.trail.set_data(a[same, 6], a[same, 7])
            self.car.set_data([a[same[-1], 6]], [a[same[-1], 7]])
        self.canvas.draw_idle()


class ProgressTab(FigureTab):
    """Per-episode charts."""

    def __init__(self, parent, smoothing: tk.IntVar):
        super().__init__(parent)
        self.smoothing = smoothing
        axs = self.fig.subplots(2, 4)
        self.ax = dict(zip(["laps", "reward", "parts", "speed", "off", "slip", "ends", "best"], axs.flat))
        self.cbar = None

    def update(self, run: RunData | None, changed: dict, force: bool = False) -> None:
        if not (force or changed.get("episodes")):
            return
        ax, w = self.ax, self.smoothing.get()
        for a in ax.values():
            a.clear()
        eps = run.episodes.rows if run else []
        if not eps:
            ax["laps"].text(0.02, 0.9, "No finished episodes yet", transform=ax["laps"].transAxes, color=T.MUTED)
            self.canvas.draw_idle()
            return
        x = np.array([e["timesteps"] for e in eps])

        def raw_and_smooth(a, y, title, ylabel):
            y = np.asarray(y, float)
            a.scatter(x, y, s=8, color=T.S1, alpha=0.3, linewidths=0)
            a.plot(x, T.rolling(y, w), color=T.S1)
            a.set_title(title)
            a.set_ylabel(ylabel)
            a.set_xlabel("training steps")

        raw_and_smooth(ax["laps"], [e["laps"] for e in eps], "Laps per episode", "laps")
        raw_and_smooth(ax["reward"], [e["reward"] for e in eps], "Episode reward", "reward")
        for key, label, color in PARTS:
            ax["parts"].plot(x, T.rolling([e["parts"].get(key, 0) for e in eps], w), color=color, label=label)
        ax["parts"].set_title("Reward breakdown (per episode)")
        ax["parts"].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=3)
        ax["parts"].set_xlabel("training steps")
        ax["speed"].plot(x, T.rolling([e["mean_speed"] for e in eps], w), color=T.S1, label="Average")
        ax["speed"].plot(x, T.rolling([e["max_speed"] for e in eps], w), color=T.S2, label="Top")
        ax["speed"].set_title("Speed")
        ax["speed"].set_ylabel("m/s")
        ax["speed"].set_xlabel("training steps")
        ax["speed"].legend(loc="lower right")
        raw_and_smooth(ax["off"], [e["offtrack_pct"] for e in eps], "Time off the road", "% of episode")
        raw_and_smooth(ax["slip"], [e["mean_abs_slip_deg"] for e in eps], "Slip angle (avg |slip|)", "degrees")

        recent = eps[-100:]
        reasons = {}
        for e in recent:
            reasons[e["end_reason"]] = reasons.get(e["end_reason"], 0) + 1
        names = sorted(reasons, key=reasons.get)
        ax["ends"].barh(names, [reasons[n] for n in names], color=T.S1, height=0.5)
        ax["ends"].set_ylim(-0.6, max(len(names), 4) - 0.4)
        ax["ends"].set_title(f"Why episodes ended (last {len(recent)})")
        ax["ends"].set_xlabel("episodes")
        ax["ends"].grid(axis="y", visible=False)

        b = ax["best"]
        b.set_title("Best episode's driving line")
        T.clear_map_axes(b)
        best = run.best_episode()
        if best and best.get("track") and best.get("traj"):
            T.draw_road(b, best["track"])
            tr = np.array(best["traj"], float)
            lc = LineCollection(np.stack([tr[:-1, :2], tr[1:, :2]], axis=1), cmap="Blues",
                                norm=matplotlib.colors.Normalize(0, max(tr[:, 2].max(), 1)), linewidths=2, zorder=2)
            lc.set_array(tr[1:, 2])
            b.add_collection(lc)
            # Clearing the axes orphans an old colour bar: create once, then repoint it.
            if self.cbar is None:
                self.cbar = self.fig.colorbar(lc, ax=b, label="speed m/s", shrink=0.7)
            else:
                self.cbar.update_normal(lc)
            b.set_xlabel(f"episode {best['episode']} · {best['laps']:.2f} laps · ended: {best['end_reason']}")
        self.canvas.draw_idle()


class LearningTab(FigureTab):
    """PPO's own statistics per update."""

    def __init__(self, parent):
        super().__init__(parent)
        axs = self.fig.subplots(2, 3)
        self.ax = dict(zip([k for k, _ in PPO_STATS], axs.flat))
        self.fig.suptitle("Entropy should fall slowly · approx KL ≈ 0.005–0.05 · explained variance → 1 "
                          "means the critic predicts returns well", color=T.MUTED, fontsize=9)

    def update(self, run: RunData | None, changed: dict, force: bool = False) -> None:
        if not (force or changed.get("updates")):
            return
        ups = run.updates.rows if run else []
        for key, label in PPO_STATS:
            a = self.ax[key]
            a.clear()
            pts = [(u["timesteps"], u[key]) for u in ups if u.get(key) is not None]
            if pts:
                xs, ys = zip(*pts)
                a.plot(xs, ys, color=T.S1, marker="o" if len(pts) < 40 else None, ms=3)
            a.set_title(label)
            a.set_xlabel("training steps")
        self.canvas.draw_idle()


class EpisodesTab(ttk.Frame):
    COLS = [("episode", "#", 50), ("timesteps", "Step", 80), ("track_seed", "Track", 60), ("laps", "Laps", 60),
            ("progress", "Distance m", 85), ("reward", "Reward", 80), ("mean_speed", "Avg m/s", 70),
            ("max_speed", "Top m/s", 70), ("offtrack_pct", "Off road %", 80), ("mean_abs_slip_deg", "Slip °", 60),
            ("end_reason", "Ended", 110)]

    def __init__(self, parent):
        super().__init__(parent)
        self.table = ttk.Treeview(self, columns=[c for c, _, _ in self.COLS], show="headings")
        for c, label, w in self.COLS:
            self.table.heading(c, text=label)
            self.table.column(c, width=w, anchor="w" if c == "end_reason" else "e")
        sb = ttk.Scrollbar(self, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=sb.set)
        self.table.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.run_name = None

    def update(self, run: RunData | None, changed: dict, force: bool = False) -> None:
        if run is None or run.name != self.run_name:
            self.table.delete(*self.table.get_children())
            self.run_name = run.name if run else None
        if run is None:
            return
        have = len(self.table.get_children())
        for e in run.episodes.rows[have:]:
            vals = []
            for c, _, _ in self.COLS:
                v = e.get(c)
                vals.append(f"{v:,.2f}" if c == "laps" else f"{v:,.0f}" if c in ("timesteps", "progress", "reward")
                            else f"{v:.1f}" if isinstance(v, float) else v)
            self.table.insert("", 0, values=vals)  # newest on top
