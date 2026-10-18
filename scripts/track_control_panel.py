"""Control panel for procedural tracks in BeamNG.

A small window next to the game: pick a seed (prev / next / random / type one),
tweak generation settings, see a preview, and load the track into BeamNG. The
next few seeds are prepared in the background so stepping through tracks is
quick. Also has buttons to reset the car to the start line and to switch
between an overhead view and the car camera.

    python scripts/track_control_panel.py

Connects to a running BeamNG (started with the research API, e.g. by
scripts/launch_beamng_proton.sh on Linux); if none is running it launches the
game from BNG_HOME, or from the default Steam location on Windows.
Linux needs Tk for Python: `sudo apt install python3-tk`.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from beamngpy import BeamNGpy  # noqa: E402

from src.envs.track_loader import TrackLoader, TrackParams  # noqa: E402

WINDOWS_STEAM_HOME = r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive"

SLIDERS = [  # (param, label, from, to, resolution)
    ("num_control_points", "Corners", 8, 20, 1),
    ("radius", "Size (m)", 80, 160, 5),
    ("width", "Road width (m)", 10, 20, 0.5),
    ("irregularity", "Wildness", 0.15, 0.5, 0.01),
    ("hilliness", "Hills", 0.0, 2.0, 0.1),
]


def connect(port: int) -> BeamNGpy:
    """Attach to a running BeamNG, or launch one if none is running."""
    bng = BeamNGpy("localhost", port)
    try:
        bng.open(launch=False)
        return bng
    except Exception:
        home = os.environ.get("BNG_HOME") or (WINDOWS_STEAM_HOME if Path(WINDOWS_STEAM_HOME).exists() else None)
        if not home:
            raise RuntimeError("BeamNG isn't running and BNG_HOME isn't set; start the game first.")
        bng = BeamNGpy("localhost", port, home=home)
        bng.open(launch=True)
        return bng


class ControlPanel:
    def __init__(self, root: tk.Tk, port: int, seed: int):
        self.root = root
        self.loader: TrackLoader | None = None
        self.game = ThreadPoolExecutor(max_workers=1, thread_name_prefix="beamng")  # all game calls go here
        self.busy = False
        self.loaded_seed: int | None = None

        root.title("Drift track generator")
        root.protocol("WM_DELETE_WINDOW", self.quit)
        main = ttk.Frame(root, padding=10)
        main.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        # Seed selection
        seed_box = ttk.LabelFrame(controls, text="Track", padding=8)
        seed_box.grid(sticky="ew")
        self.seed_var = tk.StringVar(value=str(seed))
        ttk.Button(seed_box, text="◀ Prev", command=lambda: self.step(-1)).grid(row=0, column=0)
        entry = ttk.Entry(seed_box, textvariable=self.seed_var, width=8, justify="center")
        entry.grid(row=0, column=1, padx=4)
        entry.bind("<Return>", lambda _e: self.load_seed())
        ttk.Button(seed_box, text="Next ▶", command=lambda: self.step(1)).grid(row=0, column=2)
        ttk.Button(seed_box, text="Random", command=self.random_seed).grid(row=1, column=0, pady=(6, 0))
        self.load_btn = ttk.Button(seed_box, text="Load in game", command=self.load_seed)
        self.load_btn.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(6, 0))

        # Generation settings
        gen = ttk.LabelFrame(controls, text="Generation", padding=8)
        gen.grid(sticky="ew", pady=8)
        defaults = TrackParams()
        self.mode_var = tk.StringVar(value=defaults.mode)
        ttk.Radiobutton(gen, text="Island (hills)", variable=self.mode_var, value="island",
                        command=self.settings_changed).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(gen, text="Flat grid", variable=self.mode_var, value="flat",
                        command=self.settings_changed).grid(row=0, column=1, sticky="w")
        self.slider_vars = {}
        for i, (name, label, lo, hi, res) in enumerate(SLIDERS, start=1):
            ttk.Label(gen, text=label).grid(row=i, column=0, sticky="w")
            var = tk.DoubleVar(value=getattr(defaults, name))
            tk.Scale(gen, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal", length=170,
                     command=lambda _v: self.settings_changed()).grid(row=i, column=1)
            self.slider_vars[name] = var
        ttk.Button(gen, text="Reset defaults", command=self.reset_defaults).grid(row=len(SLIDERS) + 1, column=0,
                                                                                 columnspan=2, pady=(6, 0))

        # Car / camera
        car = ttk.LabelFrame(controls, text="Car & camera", padding=8)
        car.grid(sticky="ew")
        ttk.Button(car, text="Reset car to start", command=lambda: self.run_game(lambda: self.loader.reset_car(),
                                                                                  "Car reset")).grid(sticky="ew")
        ttk.Button(car, text="Overhead view", command=lambda: self.run_game(lambda: self.loader.overview_camera(),
                                                                             "Overhead view")).grid(sticky="ew", pady=4)
        ttk.Button(car, text="Car camera", command=lambda: self.run_game(lambda: self.loader.car_camera(),
                                                                          "Car camera")).grid(sticky="ew")

        # Preview + info
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        self.fig = Figure(figsize=(5, 5), dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.info_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.info_var, justify="left").pack(anchor="w")
        self.prefetch_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.prefetch_var, foreground="gray").pack(anchor="w")
        self.status_var = tk.StringVar(value="Connecting to BeamNG…")
        ttk.Label(main, textvariable=self.status_var, relief="sunken", anchor="w").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self._preview_job = None
        self.watch(self.game.submit(lambda: TrackLoader(connect(port), self.current_params())), self.connected)
        self.poll_prefetch()

    # ------------------------------------------------------------ helpers

    def current_params(self) -> TrackParams:
        v = {k: var.get() for k, var in self.slider_vars.items()}
        return TrackParams(mode=self.mode_var.get(), num_control_points=int(v["num_control_points"]),
                           radius=float(v["radius"]), irregularity=round(float(v["irregularity"]), 2),
                           width=float(v["width"]), hilliness=round(float(v["hilliness"]), 1))

    def seed(self) -> int:
        try:
            return int(self.seed_var.get())
        except ValueError:
            return 0

    def watch(self, fut: Future, on_done) -> None:
        """Call on_done(result) on the UI thread when fut finishes; show errors in a dialog."""
        def check():
            if not fut.done():
                self.root.after(100, check)
                return
            try:
                on_done(fut.result())
            except Exception as e:  # noqa: BLE001 - surface any failure to the user
                self.status_var.set(f"Error: {e}")
                messagebox.showerror("Drift track generator", str(e))
                self.set_busy(False)
        check()

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.load_btn.state(["disabled"] if busy else ["!disabled"])

    def run_game(self, fn, done_msg: str) -> None:
        if self.loader is None or self.busy:
            return
        self.set_busy(True)
        self.watch(self.game.submit(fn), lambda _r: (self.status_var.set(done_msg), self.set_busy(False)))

    # ------------------------------------------------------------ actions

    def connected(self, loader: TrackLoader) -> None:
        self.loader = loader
        self.status_var.set("Connected. Loading first track…")
        self.load_seed()

    def step(self, delta: int) -> None:
        self.seed_var.set(str(self.seed() + delta))
        self.load_seed()

    def random_seed(self) -> None:
        self.seed_var.set(str(random.randint(0, 99_999)))
        self.load_seed()

    def reset_defaults(self) -> None:
        d = TrackParams()
        self.mode_var.set(d.mode)
        for k, var in self.slider_vars.items():
            var.set(getattr(d, k))
        self.settings_changed()

    def settings_changed(self) -> None:
        """Debounce slider drags, then preview the current seed with the new settings."""
        if self._preview_job:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(400, self.preview_only)

    def preview_only(self) -> None:
        self._preview_job = None
        if self.loader is None:
            return
        self.loader.set_params(self.current_params())
        seed = self.seed()
        self.status_var.set(f"Previewing seed {seed} — press 'Load in game' to use it")
        self.watch(self.loader.prepare(seed), self.draw_preview)

    def load_seed(self) -> None:
        if self.loader is None or self.busy:
            return
        self.loader.set_params(self.current_params())
        seed = self.seed()
        self.set_busy(True)
        ready = self.loader.status().get(seed) == "ready"
        self.status_var.set(f"Loading seed {seed}{' (prefetched)' if ready else ' (generating…)'}")
        self.watch(self.loader.prepare(seed), self.draw_preview)
        self.watch(self.game.submit(self.loader.load, seed), self.loaded)

    def loaded(self, prepared) -> None:
        self.loaded_seed = prepared.seed
        self.status_var.set(f"Seed {prepared.seed} loaded in game")
        self.set_busy(False)

    # ------------------------------------------------------------ display

    def draw_preview(self, prepared) -> None:
        if prepared.seed != self.seed():
            return  # an older request finished after the user moved on
        c = prepared.track["centerline"]
        self.ax.clear()
        if prepared.terrain is not None:
            from src.track import terrain as T
            h = prepared.terrain["height"]
            ext = [T.ORIGIN_X, T.ORIGIN_X + T.GRID_SIZE * T.SQUARE_SIZE] * 2
            self.ax.imshow(h.T, origin="lower", extent=ext, cmap="terrain", vmin=T.SEA_Z - 10, vmax=T.SEA_Z + 45)
            z = prepared.terrain["road_z"]
            self.ax.scatter(c[:, 0], c[:, 1], c=z, cmap="inferno", s=4)
            pad = 60
            self.ax.set_xlim(c[:, 0].min() - pad, c[:, 0].max() + pad)
            self.ax.set_ylim(c[:, 1].min() - pad, c[:, 1].max() + pad)
        else:
            loop = np.vstack([c, c[:1]])
            self.ax.plot(loop[:, 0], loop[:, 1], color="0.3", linewidth=prepared.params.width / 3)
        self.ax.plot(*c[0], "o", color="lime", markersize=8, markeredgecolor="k")
        self.ax.set_aspect("equal")
        self.ax.set_title(f"seed {prepared.seed}")
        self.canvas.draw_idle()

        s = prepared.stats
        info = (f"Length {s['length']:.0f} m · {s['corners']} tight corners · tightest {s['tightest']:.0f} m"
                f" · {s['straight']:.0%} straight")
        if "climb" in s:
            info += f" · {s['climb']:.1f} m elevation change"
        self.info_var.set(info)

    def poll_prefetch(self) -> None:
        if self.loader is not None:
            st = self.loader.status()
            marks = {"ready": "✓", "preparing": "…", "failed": "✗"}
            ahead = [f"{s}{marks[v]}" for s, v in st.items() if self.loaded_seed is None or s > self.loaded_seed]
            self.prefetch_var.set("Prepared ahead: " + "  ".join(ahead) if ahead else "")
        self.root.after(500, self.poll_prefetch)

    def quit(self) -> None:
        if self.loader is not None:
            loader = self.loader
            self.game.submit(lambda: (loader.close(), loader.bng.disconnect()))
        self.game.shutdown(wait=True, cancel_futures=False)
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=25252)
    args = parser.parse_args()
    root = tk.Tk()
    ControlPanel(root, args.port, args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()
