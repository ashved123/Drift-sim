"""Drift Studio: one window for the whole project.

    python scripts/drift_studio.py        (Windows: double-click start_studio.bat)

Tabs:
  Tracks           generate / preview tracks, load them into BeamNG, car & camera
  Training         start / stop training runs (separate process; Stop saves first)
  Models           saved checkpoints: continue training, next curriculum stage,
                   watch one drive, delete
  Live             the episode being driven right now (~2x a second)
  Progress         per-episode charts for the selected run
  PPO              how PPO works, drawn from the live run: its loop and phase,
                   the clipped objective, advantages, critic, exploration
  Learning curves  PPO's statistics per update
  Episodes         table of the selected run's episodes

The header shows whether BeamNG is running with its API port open, with a
"Launch BeamNG" button (Windows: starts the game directly; Linux: through
Proton) that can also restart a BeamNG opened from Steam without the port.
Only one thing drives BeamNG at a time (Tracks, a training run, or a model
replay). Linux needs Tk: `sudo apt install python3-tk`.
"""

from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui import theme as T  # noqa: E402  (sets the matplotlib backend first)
from src.envs import bng_launch  # noqa: E402
from src.ui.dashboard_tabs import EpisodesTab, LearningTab, LiveTab, ProgressTab  # noqa: E402
from src.ui.game_lock import GameLock  # noqa: E402
from src.ui.models_tab import ModelsTab  # noqa: E402
from src.ui.ppo_tab import PPOTab  # noqa: E402
from src.ui.run_data import DASH_DIR, RunData, list_runs  # noqa: E402
from src.ui.tracks_tab import TracksTab  # noqa: E402
from src.ui.training_tab import TrainingTab  # noqa: E402

TICK_MS = 500
RUN_LIST_EVERY = 10  # ticks


class Studio:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.run: RunData | None = None
        self.follow_latest = True  # switch to new runs automatically until the user picks one
        self.tick_count = 0
        self.game_lock = GameLock(on_change=self.update_header)

        root.title("Drift Studio")
        root.geometry("1440x920")
        style = ttk.Style()
        style.configure("Value.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("TNotebook.Tab", padding=(12, 5))

        top = ttk.Frame(root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Drift Studio", font=("Segoe UI", 15, "bold")).pack(side="left")
        self.live_dot = tk.Canvas(top, width=12, height=12, highlightthickness=0)
        self.live_dot.pack(side="left", padx=(16, 4))
        self.live_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.live_var, foreground=T.INK2).pack(side="left")
        self.game_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.game_var, foreground=T.INK2).pack(side="left", padx=(20, 0))
        self.bng_dot = tk.Canvas(top, width=12, height=12, highlightthickness=0)
        self.bng_dot.pack(side="left", padx=(20, 4))
        self.bng_var = tk.StringVar(value="BeamNG: checking…")
        ttk.Label(top, textvariable=self.bng_var, foreground=T.INK2).pack(side="left")
        self.launch_btn = ttk.Button(top, text="Launch BeamNG", command=self.launch_beamng)
        self.launch_btn.pack(side="left", padx=8)
        self.bng_state = "unknown"     # ready / no_port / stopped / starting / unknown
        self.bng_checking = False
        ttk.Label(top, text="Smoothing").pack(side="right")
        self.smoothing = tk.IntVar(value=20)
        ttk.Combobox(top, width=4, state="readonly", values=[10, 20, 50], textvariable=self.smoothing).pack(
            side="right", padx=(4, 12))
        self.smoothing.trace_add("write", lambda *_: self.refresh_visible(force=True))
        self.run_var = tk.StringVar()
        self.run_box = ttk.Combobox(top, width=30, state="readonly", textvariable=self.run_var)
        self.run_box.pack(side="right", padx=4)
        self.run_box.bind("<<ComboboxSelected>>", lambda _e: self.select_run(self.run_var.get(), user=True))
        ttk.Label(top, text="Run").pack(side="right")

        tiles = ttk.Frame(root, padding=(10, 0))
        tiles.pack(fill="x")
        self.tiles = {}
        for key, label in [("steps", "Training steps"), ("episodes", "Episodes"), ("sps", "Speed"),
                           ("elapsed", "Running for"), ("best", "Best episode"), ("recent", "Recent laps (avg)")]:
            f = ttk.Frame(tiles, padding=(10, 5), relief="groove")
            f.pack(side="left", fill="x", expand=True, padx=3)
            ttk.Label(f, text=label, foreground=T.INK2).pack(anchor="w")
            v, s = tk.StringVar(value="–"), tk.StringVar(value="")
            ttk.Label(f, textvariable=v, style="Value.TLabel").pack(anchor="w")
            ttk.Label(f, textvariable=s, foreground=T.MUTED).pack(anchor="w")
            self.tiles[key] = (v, s)

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
        # Dashboard tabs first: the Training tab may select an already-running run while it's being built.
        self.dash_tabs = {
            "live": LiveTab(self.tabs), "progress": ProgressTab(self.tabs, self.smoothing),
            "ppo": PPOTab(self.tabs), "learning": LearningTab(self.tabs), "episodes": EpisodesTab(self.tabs),
        }
        self.tracks_tab = TracksTab(self.tabs, self)
        self.models_tab = ModelsTab(self.tabs, self)
        self.training_tab = TrainingTab(self.tabs, self)
        self.tab_order = [("tracks", self.tracks_tab, "Tracks"), ("training", self.training_tab, "Training"),
                          ("models", self.models_tab, "Models"), ("live", self.dash_tabs["live"], "Live"),
                          ("progress", self.dash_tabs["progress"], "Progress"), ("ppo", self.dash_tabs["ppo"], "PPO"),
                          ("learning", self.dash_tabs["learning"], "Learning curves"),
                          ("episodes", self.dash_tabs["episodes"], "Episodes")]
        for _, widget, label in self.tab_order:
            self.tabs.add(widget, text=label)
        self.tabs.bind("<<NotebookTabChanged>>", lambda _e: self.refresh_visible(force=True))

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_header()
        self.check_beamng()
        self.tick()

    # ------------------------------------------------------------ navigation

    def show_tab(self, key: str) -> None:
        for k, widget, _ in self.tab_order:
            if k == key:
                self.tabs.select(widget)

    def visible_key(self) -> str:
        current = self.tabs.select()
        for k, widget, _ in self.tab_order:
            if str(widget) == current:
                return k
        return ""

    def select_run(self, name: str, user: bool = False) -> None:
        if user:
            self.follow_latest = name == (list_runs() or [None])[0]
        if self.run is None or self.run.name != name:
            self.run = RunData(name) if name else None
            self.run_var.set(name or "")
            if self.run:
                self.run.poll()
            self.refresh_visible(force=True)
            self.update_header()

    # ------------------------------------------------------------ refresh loop

    def tick(self) -> None:
        try:
            if self.tick_count % RUN_LIST_EVERY == 0:
                runs = list_runs()
                if list(self.run_box["values"]) != runs:
                    self.run_box["values"] = runs
                if runs and (self.run is None or (self.follow_latest and self.run.name != runs[0])):
                    self.select_run(runs[0])
            changed = self.run.poll() if self.run else {}
            self.training_tab.tick()
            self.models_tab.tick()
            self.refresh_visible(changed=changed)
            self.update_header()
        except Exception as e:  # noqa: BLE001 - never let one bad read kill the window
            self.live_var.set(f"Error: {e}")
        self.tick_count += 1
        self.root.after(TICK_MS, self.tick)

    def refresh_visible(self, changed: dict | None = None, force: bool = False) -> None:
        key = self.visible_key()
        if key in self.dash_tabs:
            self.dash_tabs[key].update(self.run, changed or {}, force=force)

    # ------------------------------------------------------------ BeamNG process

    def check_beamng(self) -> None:
        """Probe the API port in the background every few seconds (only while nothing is using the game)."""
        if not self.bng_checking and self.game_lock.owner is None and self.bng_state != "starting":
            self.bng_checking = True

            def probe():
                state = bng_launch.game_state()
                self.root.after(0, lambda: self._set_bng_state(state))

            threading.Thread(target=probe, daemon=True).start()
        self.root.after(4000, self.check_beamng)

    def _set_bng_state(self, state: str) -> None:
        self.bng_checking = False
        if self.bng_state != "starting":
            self.bng_state = state
        self.update_bng_status()

    def update_bng_status(self) -> None:
        owner = self.game_lock.owner
        state = "ready" if owner else self.bng_state
        port = bng_launch.DEFAULT_PORT
        text, color = {
            "ready": (f"BeamNG: ready on port {port}" + (f" · in use by {owner}" if owner else ""), T.GOOD),
            "no_port": ("BeamNG: open without the API port (started from Steam?)", T.WARNING),
            "stopped": ("BeamNG: not running", T.MUTED),
            "starting": ("BeamNG: starting… waiting for the API port", T.WARNING),
        }.get(state, ("BeamNG: checking…", T.MUTED))
        self.bng_var.set(text)
        self.bng_dot.delete("all")
        self.bng_dot.create_oval(2, 2, 11, 11, fill=color, outline="")
        self.launch_btn.configure(text="Restart BeamNG with API port" if state == "no_port" else "Launch BeamNG")
        self.launch_btn.state(["!disabled" if state in ("stopped", "no_port") else "disabled"])

    def launch_beamng(self) -> None:
        restart = self.bng_state == "no_port"
        if restart and not messagebox.askyesno(
                "Drift Studio", "BeamNG is open but its API port is closed, so Python can't control it "
                                "(this happens when it's started from Steam).\n\nClose it and start it again "
                                "with the port open? Unsaved in-game progress is lost."):
            return
        self.bng_state = "starting"
        self.update_bng_status()

        def work():
            try:
                bng_launch.ensure_running(restart_without_port=restart)
                result = "ready"
            except Exception as e:  # noqa: BLE001 - shown to the user
                result = e
            self.root.after(0, lambda: self._launch_done(result))

        threading.Thread(target=work, daemon=True).start()

    def _launch_done(self, result) -> None:
        if isinstance(result, Exception):
            self.bng_state = "unknown"
            messagebox.showerror("Drift Studio", f"Couldn't start BeamNG:\n\n{result}")
        else:
            self.bng_state = "ready"
        self.update_bng_status()

    def update_header(self) -> None:
        owner = self.game_lock.owner
        self.game_var.set(f"Game: {'in use by ' + owner if owner else 'free'}")
        if hasattr(self, "bng_var"):
            self.update_bng_status()
        run = self.run
        live = bool(run and run.is_live())
        self.live_dot.delete("all")
        self.live_dot.create_oval(2, 2, 11, 11, fill=T.GOOD if live else T.MUTED, outline="")
        st = (run.status if run else None) or {}
        if run is None:
            self.live_var.set(f"No training runs yet (reading {DASH_DIR})")
        elif live:
            self.live_var.set(f"Training · {st.get('phase', '')} · updated {time.time() - st['updated']:.0f}s ago")
        else:
            self.live_var.set("Not training" if st else "Waiting for the first data…")

        meta, eps = (run.meta or {}) if run else {}, run.episodes.rows if run else []
        t = self.tiles
        steps = st.get("timesteps") or (eps[-1]["timesteps"] if eps else 0)
        target = (meta.get("start_timesteps") or 0) + meta["total_timesteps"] if meta.get("total_timesteps") else None
        t["steps"][0].set(f"{steps:,}" if run else "–")
        t["steps"][1].set((f"Stage {meta.get('stage', '?')}" + (f" · {min(100, 100 * steps / target):.0f}% of "
                                                                 f"{target:,}" if target else "")) if run else "")
        t["episodes"][0].set(f"{len(eps):,}" if run else "–")
        t["episodes"][1].set(f"{len({e['track_seed'] for e in eps})} different tracks" if eps else "")
        sps = st.get("steps_per_s")
        t["sps"][0].set(f"{sps:.1f} /s" if sps else "–")
        t["sps"][1].set(f"≈ {T.fmt_duration((target - steps) / sps)} left" if live and sps and target
                        else "steps per second")
        t["elapsed"][0].set(T.fmt_duration(st.get("elapsed_s")))
        t["elapsed"][1].set(f"started {meta['started']}" if meta.get("started") else "")
        best = max(eps, key=lambda e: e["progress"], default=None)
        t["best"][0].set(f"{best['laps']:.2f} laps" if best else "–")
        t["best"][1].set(f"episode {best['episode']}, {best['progress']:.0f} m" if best else "")
        recent = eps[-self.smoothing.get():]
        t["recent"][0].set(f"{sum(e['laps'] for e in recent) / len(recent):.2f}" if recent else "–")
        t["recent"][1].set(f"last {len(recent)} episodes" if recent else "")
        if hasattr(self, "tracks_tab"):
            self.tracks_tab.refresh_game_state()

    # ------------------------------------------------------------ closing

    def on_close(self) -> None:
        if self.training_tab.running():
            answer = messagebox.askyesnocancel(
                "Drift Studio", "Training is still running.\n\nYes — stop it (it saves first), then close.\n"
                                "No — leave it training in the background and close this window.\n"
                                "Cancel — keep the Studio open.")
            if answer is None:
                return
            if answer:
                self.training_tab.stop()
        self.models_tab.shutdown()
        self.tracks_tab.shutdown()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    Studio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
