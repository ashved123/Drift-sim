"""Load generated tracks into a running BeamNG, with background prefetching.

Generating a track + island terrain and writing its bundle takes ~1 s of
Python work; applying it in-game takes another ~1 s. `TrackLoader` keeps the
next few seeds prepared on background threads so switching to them only pays
the in-game part. All calls into BeamNG stay on the caller's thread — the
prefetch workers only compute and write files.

Used by Drift Studio's Tracks tab (`src/ui/tracks_tab.py`) and the training env.
"""

from __future__ import annotations

import shutil
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle

from src.envs import track_placement as P
from src.track.generator import generate_track, track_feature_vector
from src.track.terrain import generate_terrain


@dataclass(frozen=True)
class TrackParams:
    mode: str = "island"            # "island" (hills, grass) or "flat" (smallgrid)
    num_control_points: int = 14    # more points = more corners
    radius: float = 120.0           # overall track size, metres
    irregularity: float = 0.35      # how far corners push in/out from a circle
    width: float = 15.0             # road width, metres
    hilliness: float = 1.0          # island only: 0 = flat island, 2 = very hilly
    adapt_width: bool = True        # narrow the road where a corner/section is too tight, instead of failing
    min_width: float = 6.0          # narrowest the road may get when adapting, metres

    def key(self) -> str:
        return "_".join(f"{v}" for v in asdict(self).values())


@dataclass
class PreparedTrack:
    seed: int
    params: TrackParams
    track: dict
    terrain: dict | None
    bundle: str | None              # game-relative bundle folder (island mode)
    stats: dict = field(default_factory=dict)


class TrackLoader:
    def __init__(self, bng: BeamNGpy, params: TrackParams = TrackParams(), prefetch: int = 5,
                 model: str = P.DRIFT_CAR_MODEL, config: str | None = P.DRIFT_CAR_CONFIG,
                 keep_paused: bool = False):
        self.bng = bng
        self.params = params
        self.prefetch = prefetch
        self.keep_paused = keep_paused  # training steps physics itself, so never unpause
        self.model, self.config = model, config
        self.vehicle: Vehicle | None = None
        self.current: PreparedTrack | None = None
        self._level: str | None = None
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="track-prefetch")
        self._cache: dict[tuple[str, int], Future] = {}
        self._user_dir = Path(bng.user_with_version)

    # ------------------------------------------------------------ preparation

    def set_params(self, params: TrackParams) -> None:
        """Change generation settings; prepared tracks for old settings are dropped."""
        if params != self.params:
            self.params = params
            self._trim_cache(keep=range(0))

    def prepare(self, seed: int) -> Future:
        """Start (or reuse) background preparation of `seed` with the current params."""
        k = (self.params.key(), seed)
        if k not in self._cache:
            self._cache[k] = self._pool.submit(self._build, seed, self.params)
        return self._cache[k]

    def status(self) -> dict[int, str]:
        """{seed: 'ready' | 'preparing' | 'failed'} for the current params."""
        out = {}
        for (key, seed), fut in sorted(self._cache.items(), key=lambda kv: kv[0][1]):
            if key == self.params.key():
                out[seed] = ("failed" if fut.exception() else "ready") if fut.done() else "preparing"
        return out

    def _build(self, seed: int, params: TrackParams) -> PreparedTrack:
        track = generate_track(num_control_points=params.num_control_points, radius=params.radius,
                               irregularity=params.irregularity, width=params.width,
                               adapt_width=params.adapt_width, min_width=params.min_width, seed=seed)
        length, _, max_k, corners, straight = track_feature_vector(track)
        stats = dict(length=length, corners=int(corners), tightest=1 / max_k, straight=straight,
                     min_width=float(track["widths"].min()))
        terrain = bundle = None
        if params.mode == "island":
            terrain = generate_terrain(track, seed=seed, hilliness=params.hilliness)
            bundle = P.write_island_bundle(self._user_dir, f"{seed}_{zlib.crc32(params.key().encode()):08x}", track, terrain)
            stats["climb"] = float(np.ptp(terrain["road_z"]))
        return PreparedTrack(seed, params, track, terrain, bundle, stats)

    # ------------------------------------------------------------ loading

    def load(self, seed: int) -> PreparedTrack:
        """Put `seed` in the game (waits for preparation if needed), then prefetch the following seeds."""
        prepared = self.prepare(seed).result()
        if prepared.params.mode == "island":
            self._load_island(prepared)
        else:
            self._load_flat(prepared)
        self.current = prepared
        for s in range(seed + 1, seed + 1 + self.prefetch):
            self.prepare(s)
        self._trim_cache(keep=range(seed - 2, seed + 1 + self.prefetch))
        return prepared

    def _load_island(self, prepared: PreparedTrack) -> None:
        if self._level != P.ISLAND_LEVEL:
            scenario = Scenario(P.ISLAND_LEVEL, "drift_island")
            self.vehicle = Vehicle("ego", model=self.model, part_config=self.config)
            # Spawn somewhere harmless; it's teleported onto the track once the terrain is in.
            scenario.add_vehicle(self.vehicle, pos=(-143.9, 70.2, 140.0), rot_quat=(0, 0, 0, 1))
            scenario.make(self.bng)
            self.bng.scenario.load(scenario)
            self.bng.scenario.start()
            self._level = P.ISLAND_LEVEL
        self.bng.control.pause()
        try:
            P.apply_island_bundle(self.bng, prepared.bundle)
            self.reset_car(prepared)
        finally:
            if not self.keep_paused:
                self.bng.control.resume()

    def _load_flat(self, prepared: PreparedTrack) -> None:
        # MeshRoads are baked into the scenario, so every flat track reloads the level.
        scenario = Scenario(P.FLAT_LEVEL, f"drift_flat_{prepared.seed}")
        P.add_track_to_scenario(scenario, prepared.track)
        self.vehicle = Vehicle("ego", model=self.model, part_config=self.config)
        pos, rot = P.start_pose(prepared.track)
        scenario.add_vehicle(self.vehicle, pos=pos, rot_quat=rot)
        scenario.make(self.bng)
        self.bng.scenario.load(scenario)
        self.bng.scenario.start()
        self._level = P.FLAT_LEVEL

    def reset_car(self, prepared: PreparedTrack | None = None) -> None:
        """Put the car back on the start line, stopped and repaired."""
        prepared = prepared or self.current
        if prepared is None or self.vehicle is None:
            return
        if not self.vehicle.is_connected():
            self.vehicle.connect(self.bng)
        ground = prepared.terrain["road_z"] if prepared.terrain is not None else P.GROUND_Z
        pos, rot = P.start_pose(prepared.track, ground_z=ground)
        # Leaves the camera alone: if the user is in a free/overhead view it stays put.
        self.vehicle.teleport(pos, rot, reset=True)

    def car_camera(self) -> None:
        """Back to the normal chase camera behind the car."""
        if self.vehicle is not None:
            self.bng.camera.set_player_mode(self.vehicle, "orbit", {})

    def overview_camera(self) -> None:
        """Free camera high above the track, looking down at it."""
        if self.current is None:
            return
        c = P.world_centerline(self.current.track)
        top = float(self.current.terrain["road_z"].max()) if self.current.terrain is not None else P.GROUND_Z
        extent = float(np.ptp(c, axis=0).max())
        cx, cy = c.mean(axis=0)
        self.bng.camera.set_free((float(cx), float(cy) - extent * 0.6, top + extent * 0.9), (0, 0.6, -1))

    def _trim_cache(self, keep: range) -> None:
        for k in [k for k in self._cache if k[1] not in keep or k[0] != self.params.key()]:
            fut = self._cache.pop(k)
            if not fut.cancel() and fut.done() and not fut.exception() and fut.result().bundle:
                if self.current is None or fut.result().bundle != self.current.bundle:
                    shutil.rmtree(self._user_dir / fut.result().bundle, ignore_errors=True)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self._user_dir / P.BUNDLE_DIR, ignore_errors=True)
