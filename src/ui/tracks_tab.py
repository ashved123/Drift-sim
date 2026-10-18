"""Tracks tab: pick seeds, tweak generation, preview, and load tracks into BeamNG.

Previews are computed here (no game needed), so they also work while training
is using the game. Loading a track claims the game from the Studio's game lock
(src/ui/game_lock.py); the connection is dropped again whenever training or a
model replay needs the game.
"""

from __future__ import annotations

import random
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.envs.track_loader import TrackParams
from src.track import terrain as TR
from src.track.generator import generate_track, track_feature_vector
from src.track.terrain import generate_terrain
from src.ui import theme as T

SLIDERS = [  # (param, label, from, to, resolution)
    ("num_control_points", "Corners", 8, 20, 1),
    ("radius", "Size (m)", 80, 160, 5),
    ("width", "Road width (m)", 10, 20, 0.5),
    ("irregularity", "Wildness", 0.15, 0.5, 0.01),
    ("hilliness", "Hills", 0.0, 2.0, 0.1),
    ("min_width", "Min road width (m)", 4, 12, 0.5),
]
OWNER = "tracks"


class TracksTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=8)
        self.app = app
        self.loader = None                     # TrackLoader while this tab holds the game
        self.game = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracks-game")
        self.preview_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracks-preview")
        self.busy = False
        self._preview_job = None
        self._preview_cache: dict = {}

        controls = ttk.Frame(self)
        controls.pack(side="left", fill="y", padx=(0, 10))

        box = ttk.LabelFrame(controls, text="Track", padding=8)
        box.pack(fill="x")
        self.seed_var = tk.StringVar(value="0")
        ttk.Button(box, text="◀ Prev", command=lambda: self.step(-1)).grid(row=0, column=0)
        entry = ttk.Entry(box, textvariable=self.seed_var, width=8, justify="center")
        entry.grid(row=0, column=1, padx=4)
        entry.bind("<Return>", lambda _e: self.preview())
        ttk.Button(box, text="Next ▶", command=lambda: self.step(1)).grid(row=0, column=2)
        ttk.Button(box, text="Random", command=self.random_seed).grid(row=1, column=0, pady=(6, 0))
        self.load_btn = ttk.Button(box, text="Load in game", command=self.load_in_game)
        self.load_btn.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(6, 0))

        gen = ttk.LabelFrame(controls, text="Generation", padding=8)
        gen.pack(fill="x", pady=8)
        d = TrackParams()
        self.mode_var = tk.StringVar(value=d.mode)
        ttk.Radiobutton(gen, text="Island (hills)", variable=self.mode_var, value="island",
                        command=self.settings_changed).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(gen, text="Flat grid", variable=self.mode_var, value="flat",
                        command=self.settings_changed).grid(row=0, column=1, sticky="w")
        self.slider_vars = {}
        for i, (name, label, lo, hi, res) in enumerate(SLIDERS, start=1):
            ttk.Label(gen, text=label).grid(row=i, column=0, sticky="w")
            var = tk.DoubleVar(value=getattr(d, name))
            tk.Scale(gen, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal", length=170,
                     command=lambda _v: self.settings_changed()).grid(row=i, column=1)
            self.slider_vars[name] = var
        self.adapt_var = tk.BooleanVar(value=d.adapt_width)
        ttk.Checkbutton(gen, text="Narrow road where needed (never fail)", variable=self.adapt_var,
                        command=self.settings_changed).grid(row=len(SLIDERS) + 1, column=0, columnspan=2, sticky="w")
        ttk.Button(gen, text="Reset defaults", command=self.reset_defaults).grid(
            row=len(SLIDERS) + 2, column=0, columnspan=2, pady=(6, 0))

        car = ttk.LabelFrame(controls, text="Car & camera (in game)", padding=8)
        car.pack(fill="x")
        self.car_btns = [
            ttk.Button(car, text="Reset car to start", command=lambda: self.run_game(lambda: self.loader.reset_car())),
            ttk.Button(car, text="Overhead view", command=lambda: self.run_game(lambda: self.loader.overview_camera())),
            ttk.Button(car, text="Car camera", command=lambda: self.run_game(lambda: self.loader.car_camera())),
            ttk.Button(car, text="Disconnect from game", command=self.release),
        ]
        for b in self.car_btns:
            b.pack(fill="x", pady=1)
        self.game_note = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.game_note, foreground=T.MUTED, wraplength=260, justify="left").pack(
            fill="x", pady=8)

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(6, 6), dpi=90, layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.info = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.info, justify="left").pack(anchor="w")
        self.status = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status, foreground=T.INK2).pack(anchor="w")

        self.refresh_game_state()
        self.preview()

    # ------------------------------------------------------------ settings

    def params(self) -> TrackParams:
        v = {k: var.get() for k, var in self.slider_vars.items()}
        return TrackParams(mode=self.mode_var.get(), num_control_points=int(v["num_control_points"]),
                           radius=float(v["radius"]), irregularity=round(float(v["irregularity"]), 2),
                           width=float(v["width"]), hilliness=round(float(v["hilliness"]), 1),
                           adapt_width=bool(self.adapt_var.get()), min_width=float(v["min_width"]))

    def seed(self) -> int:
        try:
            return int(self.seed_var.get())
        except ValueError:
            return 0

    def step(self, delta: int) -> None:
        self.seed_var.set(str(self.seed() + delta))
        self.preview()

    def random_seed(self) -> None:
        self.seed_var.set(str(random.randint(0, 99_999)))
        self.preview()

    def reset_defaults(self) -> None:
        d = TrackParams()
        self.mode_var.set(d.mode)
        self.adapt_var.set(d.adapt_width)
        for k, var in self.slider_vars.items():
            var.set(getattr(d, k))
        self.settings_changed()

    def settings_changed(self) -> None:
        if self._preview_job:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(400, self.preview)

    # ------------------------------------------------------------ preview (no game needed)

    def preview(self) -> None:
        self._preview_job = None
        seed, params = self.seed(), self.params()
        key = (seed, params)
        if key not in self._preview_cache:
            self._preview_cache = {k: f for k, f in self._preview_cache.items() if k[1] == params}
            self._preview_cache[key] = self.preview_pool.submit(_build_preview, seed, params)
        self.status.set(f"Previewing seed {seed}…")
        self._when_done(self._preview_cache[key], lambda r: self.draw_preview(seed, params, r))

    def _when_done(self, fut: Future, on_done) -> None:
        def check():
            if not fut.done():
                self.after(100, check)
                return
            try:
                on_done(fut.result())
            except Exception as e:  # noqa: BLE001 - show any failure
                self.status.set(f"Error: {e}")
                self.busy = False
                self.refresh_game_state()
        check()

    def draw_preview(self, seed: int, params: TrackParams, result: dict) -> None:
        if seed != self.seed():
            return  # an older request finished after the user moved on
        track, terrain, stats = result["track"], result["terrain"], result["stats"]
        c = track["centerline"]
        a = self.ax
        a.clear()
        if terrain is not None:
            ext = [TR.ORIGIN_X, TR.ORIGIN_X + TR.GRID_SIZE * TR.SQUARE_SIZE] * 2
            a.imshow(terrain["height"].T, origin="lower", extent=ext, cmap="terrain", vmin=TR.SEA_Z - 10,
                     vmax=TR.SEA_Z + 45)
            a.scatter(c[:, 0], c[:, 1], c=terrain["road_z"], cmap="inferno", s=4)
            a.set_xlim(c[:, 0].min() - 60, c[:, 0].max() + 60)
            a.set_ylim(c[:, 1].min() - 60, c[:, 1].max() + 60)
        else:
            loop = np.vstack([c, c[:1]])
            a.plot(loop[:, 0], loop[:, 1], color="0.3", linewidth=params.width / 3)
        a.plot(*c[0], "o", color="lime", markersize=8, markeredgecolor="k")
        a.set_aspect("equal")
        a.grid(False)
        a.set_title(f"seed {seed}")
        self.canvas.draw_idle()
        info = (f"Length {stats['length']:.0f} m · {stats['corners']} tight corners · tightest {stats['tightest']:.0f} m"
                f" · {stats['straight']:.0%} straight")
        if "climb" in stats:
            info += f" · {stats['climb']:.1f} m elevation change"
        if stats["min_width"] < params.width - 0.05:
            info += f"\nRoad narrowed to {stats['min_width']:.1f} m in places (tight corners / close sections)"
        self.info.set(info)
        if not self.busy:
            self.status.set(f"Seed {seed} — press 'Load in game' to drive it")

    # ------------------------------------------------------------ game

    def refresh_game_state(self) -> None:
        owner = self.app.game_lock.owner
        free = owner in (None, OWNER)
        state = "!disabled" if free and not self.busy else "disabled"
        self.load_btn.state([state])
        for b in self.car_btns:
            b.state(["!disabled" if owner == OWNER and not self.busy else "disabled"])
        self.game_note.set("" if free else f"The game is in use by {owner} — track previews still work; "
                                           "loading is available again when it finishes.")

    def load_in_game(self) -> None:
        if self.busy or not self.app.game_lock.claim(OWNER, release=self.release):
            self.refresh_game_state()
            return
        seed, params = self.seed(), self.params()
        self.busy = True
        self.refresh_game_state()
        self.status.set(f"Loading seed {seed} in BeamNG… (first load also starts the level)")

        def work():
            if self.loader is None:
                from src.envs.bng_launch import connect
                from src.envs.track_loader import TrackLoader
                self.loader = TrackLoader(connect(), params)
            self.loader.set_params(params)
            return self.loader.load(seed)

        def done(_prepared):
            self.busy = False
            self.status.set(f"Seed {seed} loaded in game")
            self.refresh_game_state()

        self._when_done(self.game.submit(work), done)

    def run_game(self, fn) -> None:
        if self.loader is None or self.busy:
            return
        self.busy = True
        self.refresh_game_state()

        def done(_):
            self.busy = False
            self.refresh_game_state()

        self._when_done(self.game.submit(fn), done)

    def release(self) -> None:
        """Drop the game connection (called by the game lock when someone else needs it)."""
        loader, self.loader = self.loader, None
        if loader is not None:
            fut = self.game.submit(lambda: (loader.close(), loader.bng.disconnect()))
            fut.result(timeout=30)
        self.app.game_lock.release(OWNER)
        self.status.set("Disconnected from the game")
        self.refresh_game_state()

    def shutdown(self) -> None:
        if self.loader is not None:
            try:
                self.release()
            except Exception:  # noqa: BLE001 - closing anyway
                pass
        self.game.shutdown(wait=False)
        self.preview_pool.shutdown(wait=False, cancel_futures=True)


def _build_preview(seed: int, params: TrackParams) -> dict:
    track = generate_track(num_control_points=params.num_control_points, radius=params.radius,
                           irregularity=params.irregularity, width=params.width,
                           adapt_width=params.adapt_width, min_width=params.min_width, seed=seed)
    length, _, max_k, corners, straight = track_feature_vector(track)
    stats = dict(length=length, corners=int(corners), tightest=1 / max_k, straight=straight,
                 min_width=float(track["widths"].min()))
    terrain = None
    if params.mode == "island":
        terrain = generate_terrain(track, seed=seed, hilliness=params.hilliness)
        stats["climb"] = float(np.ptp(terrain["road_z"]))
    return dict(track=track, terrain=terrain, stats=stats)
