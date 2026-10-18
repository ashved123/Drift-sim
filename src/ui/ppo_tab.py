"""PPO tab: how the algorithm works, drawn from the live run.

Top: PPO's loop as a diagram with the phase training is in right now
(driving to collect experience → scoring it → updating the policy).
Below, measured on a 512-step sample of the last update's rollout
(src/training/dashboard_log.py writes ppo_viz.json / ppo_hist.jsonl):
  - the clipped objective, and how far the update actually moved each
    action's probability (the ratio) against the clip range
  - advantages: which actions turned out better / worse than the critic expected
  - the critic: predicted value vs actual return, before and after the update
  - exploration: the policy's action noise (std) per control, over updates
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from src.ui import theme as T
from src.ui.dashboard_tabs import FigureTab
from src.ui.run_data import RunData

POS, NEG = "#2a78d6", "#e34948"  # diverging pair: better than expected / worse than expected
STEPS = [
    ("1  Drive & collect", "The current policy drives; every step's\nobservation, action and reward is stored"),
    ("2  Score the actions", "Advantages (GAE): was each action better or\nworse than the critic expected?"),
    ("3  Update the policy", "Push good actions up, bad ones down —\nclipped so it can't change too much at once"),
]


class PPOTab(FigureTab):
    def __init__(self, parent):
        super().__init__(parent, figsize=(14, 6))
        self.diagram = tk.Canvas(self, height=150, background=T.SURFACE, highlightthickness=0)
        self.diagram.pack(side="top", fill="x")
        self.diagram.bind("<Configure>", lambda _e: self._draw_diagram())
        self.caption = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.caption, padding=(10, 2), foreground=T.INK2).pack(side="top", anchor="w")
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1])
        f = self.fig
        self.ax = {"objective": f.add_subplot(gs[0, 0]), "ratio": f.add_subplot(gs[1, 0]),
                   "adv": f.add_subplot(gs[0, 1]), "critic": f.add_subplot(gs[1, 1]),
                   "explore": f.add_subplot(gs[:, 2])}
        self.status: dict = {}
        self.live = False
        self.viz: dict | None = None

    # ------------------------------------------------------------ diagram

    def _draw_diagram(self) -> None:
        c = self.diagram
        c.delete("all")
        W, H = max(c.winfo_width(), 600), 150
        st, live = self.status, self.live
        phase = st.get("phase") if live else None
        active = {"collecting": 0, "updating": 2}.get(phase)
        bw, bh, top = (W - 120) / 3, 78, 18
        for i, (title, body) in enumerate(STEPS):
            x0 = 30 + i * (bw + 30)
            on = active == i or (phase == "updating" and i == 1)
            c.create_rectangle(x0, top, x0 + bw, top + bh, outline=T.S1 if on else T.AXIS, width=2 if on else 1,
                               fill="#e8f1fc" if on else T.SURFACE)
            c.create_text(x0 + 10, top + 14, text=title, anchor="w", font=("Segoe UI", 11, "bold"),
                          fill=T.INK if on or not live else T.INK2)
            c.create_text(x0 + 10, top + 46, text=body, anchor="w", font=("Segoe UI", 9), fill=T.INK2)
            if i < 2:
                c.create_line(x0 + bw + 4, top + bh / 2, x0 + bw + 26, top + bh / 2, arrow="last", fill=T.MUTED, width=2)
        # loop-back arrow from step 3 to step 1
        x_end, x_start, y = 30 + 2 * (bw + 30) + bw / 2, 30 + bw / 2, top + bh
        c.create_line(x_end, y + 2, x_end, y + 22, x_start, y + 22, x_start, y + 4, arrow="last", fill=T.MUTED, width=2)
        it = st.get("iteration", 0)
        c.create_text(W / 2, y + 30, text=f"repeat — iteration {it + (1 if phase == 'collecting' else 0)}"
                      if live else "repeat (not training right now)", fill=T.MUTED, font=("Segoe UI", 9))
        if phase == "collecting":  # rollout progress bar inside step 1
            frac = min(1.0, st.get("rollout_pos", 0) / max(st.get("n_steps", 1), 1))
            x0 = 30
            c.create_rectangle(x0 + 10, top + bh - 12, x0 + bw - 10, top + bh - 6, outline="", fill=T.GRID)
            c.create_rectangle(x0 + 10, top + bh - 12, x0 + 10 + (bw - 20) * frac, top + bh - 6, outline="", fill=T.S1)
            c.create_text(x0 + bw - 10, top + 14, anchor="e", fill=T.INK2, font=("Segoe UI", 9),
                          text=f"{st.get('rollout_pos', 0):,} / {st.get('n_steps', 0):,} steps")
        elif phase == "updating":
            x0 = 30 + 2 * (bw + 30)
            c.create_text(x0 + bw - 10, top + 14, anchor="e", fill=T.INK2, font=("Segoe UI", 9),
                          text=f"{st.get('n_epochs', 10)} passes over the data…")

    # ------------------------------------------------------------ charts

    def update(self, run: RunData | None, changed: dict, force: bool = False) -> None:
        self.status = (run.status if run else None) or {}
        self.live = bool(run and run.is_live())
        self._draw_diagram()
        if not (force or changed.get("ppo")):
            return
        self.viz = run.ppo_viz if run else None
        hist = run.ppo_hist.rows if run else []
        for a in self.ax.values():
            a.clear()
        v = self.viz
        if not v:
            self.ax["objective"].text(0.02, 0.85, "Appears after the first PPO update\n(every 2,048 training steps).",
                                      transform=self.ax["objective"].transAxes, color=T.MUTED)
            self.caption.set("")
            self.canvas.draw_idle()
            return
        eps = v["clip_range"]
        self.caption.set(
            f"Last update: iteration {v['iteration']} at step {v['timesteps']:,} · "
            f"{v['n_epochs']} passes over {v['n_steps']:,} steps in batches of {v['batch_size']} · "
            f"clip range ε = {eps:.2f} · measured on {v['sample']} sampled steps")
        self._objective(eps)
        self._ratio(v, eps)
        self._advantages(v)
        self._critic(v)
        self._exploration(hist)
        self.canvas.draw_idle()

    def _objective(self, eps: float) -> None:
        a = self.ax["objective"]
        r = np.linspace(1 - 3 * eps, 1 + 3 * eps, 300)
        clip = np.clip(r, 1 - eps, 1 + eps)
        a.axvspan(1 - eps, 1 + eps, color=T.GRID, alpha=0.6, lw=0)
        a.plot(r, np.minimum(r * 1, clip * 1), color=POS, label="good action (advantage > 0)")
        a.plot(r, np.minimum(r * -1, clip * -1), color=NEG, label="bad action (advantage < 0)")
        a.axvline(1, color=T.AXIS, lw=1)
        a.set_title("PPO's clipped objective")
        a.set_ylabel("objective (per unit advantage)")
        a.legend(loc="upper left")
        a.annotate("flat = no reward for\nmoving further", (1 + 2 * eps, 1 + eps), xytext=(0, -28),
                   textcoords="offset points", ha="center", color=T.MUTED, fontsize=8)

    def _ratio(self, v: dict, eps: float) -> None:
        a = self.ax["ratio"]
        edges, counts = np.array(v["ratio"]["edges"]), np.array(v["ratio"]["counts"])
        centers, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
        inside = np.abs(centers - 1) <= eps
        a.bar(centers[inside], counts[inside], width=width[inside] * 0.9, color=T.S1, label="within clip range")
        a.bar(centers[~inside], counts[~inside], width=width[~inside] * 0.9, color=T.S2, label="clipped")
        a.axvspan(1 - eps, 1 + eps, color=T.GRID, alpha=0.6, lw=0, zorder=0)
        a.axvline(1, color=T.AXIS, lw=1)
        a.set_xlim(edges[0], edges[-1])
        a.set_title(f"How much the update moved each action ({100 * v['clipped_frac']:.0f}% hit the clip)")
        a.set_xlabel("probability ratio  new policy ÷ old policy  (1 = unchanged)")
        a.set_ylabel("sampled steps")
        a.legend(loc="upper right")

    def _advantages(self, v: dict) -> None:
        a = self.ax["adv"]
        edges, counts = np.array(v["adv"]["edges"]), np.array(v["adv"]["counts"])
        centers, width = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
        a.bar(centers[centers >= 0], counts[centers >= 0], width=width[centers >= 0] * 0.9, color=POS,
              label="better than expected")
        a.bar(centers[centers < 0], counts[centers < 0], width=width[centers < 0] * 0.9, color=NEG,
              label="worse than expected")
        a.axvline(0, color=T.AXIS, lw=1)
        a.set_title(f"Advantages ({100 * v['adv_pos_frac']:.0f}% of actions better than expected)")
        a.set_xlabel("advantage (normalised reward units)")
        a.set_ylabel("sampled steps")
        a.legend(loc="upper right")

    def _critic(self, v: dict) -> None:
        a = self.ax["critic"]
        ret, old, new = (np.array(v["values"][k]) for k in ("ret", "old", "new"))
        a.scatter(ret, old, s=10, color=T.S2, alpha=0.6, linewidths=0, label="before update")
        a.scatter(ret, new, s=10, color=T.S1, alpha=0.6, linewidths=0, label="after update")
        lo, hi = float(min(ret.min(), old.min(), new.min())), float(max(ret.max(), old.max(), new.max()))
        a.plot([lo, hi], [lo, hi], color=T.AXIS, lw=1, label="perfect prediction")
        ev_b, ev_a = v.get("explained_variance_before"), v.get("explained_variance_after")
        a.set_title("Critic: predicted value vs actual return")
        a.set_xlabel("actual return (what really happened)" +
                     (f"\nexplained variance {ev_b:.2f} before → {ev_a:.2f} after this update"
                      if ev_b is not None and ev_a is not None else ""))
        a.set_ylabel("critic's prediction")
        a.legend(loc="upper left")

    def _exploration(self, hist: list) -> None:
        a = self.ax["explore"]
        a.set_title("Exploration: policy's action noise (std)")
        a.set_xlabel("training steps")
        a.set_ylabel("action noise (std)")
        if not hist:
            return
        x = [h["timesteps"] for h in hist]
        names = list(hist[-1].get("action_std", {}).keys())
        for name, color in zip(names, T.SERIES):
            a.plot(x, [h["action_std"].get(name, np.nan) for h in hist], color=color, label=name,
                   marker="o" if len(hist) < 40 else None, ms=3)
        a.legend(loc="upper right", ncols=2)
        a.annotate("falls as the policy gets more confident;\nmasked controls (not in this stage) barely move",
                   (0.02, 0.02), xycoords="axes fraction", color=T.MUTED, fontsize=8)
