"""Gymnasium env wrapping BeamNGpy for the drift agent.

Action vector (always 6-dim, each in [-1, 1]; curriculum stage decides which
dims are "live" — inactive dims are forced to a safe neutral control before
being sent to BeamNG, per CLAUDE.md's progressive-unlock curriculum):

    [steer, throttle, brake, handbrake, clutch, shift_delta]

steer maps to BeamNG steering directly; throttle/brake/handbrake/clutch map
from [-1, 1] to pedal [0, 1]. shift_delta is rounded/clipped to {-1, 0, +1}
and applied to persistent gear state (stage 4+ only; earlier stages use the
automatic "arcade" gearbox). Reverse is simply the gear rung below 1st.

Observation (15 floats, raw units — training wraps the env in VecNormalize):
    0  lateral offset / road half-width (signed, +left; |x| > 1 = off the road)
    1  heading error vs track direction (rad, +left)
    2-4 track curvature 10 / 30 / 60 m ahead (1/m, +left)
    5  forward speed (m/s)      6  sideways speed (m/s, +left)
    7  yaw rate (rad/s, +left)  8  slip angle (rad)
    9  wheelspin: average wheel speed minus forward speed (m/s)
    10 engine rpm / max rpm     11 gear index
    12 damage (scaled, ~1 = wrecked)
    13-14 previous steer and throttle actions
BeamNG only exposes an average wheel speed, not one per wheel, hence the
single wheelspin value.

Simulation runs in BeamNG's deterministic mode: the game stays paused and
each step() advances it a fixed number of frames, so training is
reproducible and not tied to real time. Tracks come from TrackLoader (island
level, ~1 s to switch); a new track is generated every `new_track_every`
episodes.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from beamngpy.sensors import Damage, Electrics
from gymnasium import spaces

from src.envs import track_placement as P
from src.envs.bng_launch import connect
from src.envs.track_loader import TrackLoader, TrackParams

STAGE_ACTIVE_DIMS = {
    1: {"steer", "throttle"},
    2: {"steer", "throttle", "brake"},
    3: {"steer", "throttle", "brake", "handbrake"},
    4: {"steer", "throttle", "brake", "handbrake", "shift_delta"},
    5: {"steer", "throttle", "brake", "handbrake", "shift_delta", "clutch"},
}

ACTION_DIMS = ["steer", "throttle", "brake", "handbrake", "clutch", "shift_delta"]
MIN_GEAR, MAX_GEAR = -1, 6  # -1 = reverse, 0 = neutral, 1..6 = forward gears

# Reward/termination coefficients — placeholders, to be tuned empirically
# once training runs are possible. See CLAUDE.md "Reward function".
K_OFFTRACK = 0.5
K_DAMAGE = 5.0
TIME_PENALTY = 0.01
THETA_MAX = np.radians(55)       # top of the "more slip angle = more reward" range
THETA_FALLOFF = np.radians(20)   # bonus decays back to 0 over this range past THETA_MAX
RECOVERY_TIMEOUT = 150           # steps allowed off-track before terminating
DAMAGE_SCALE = 10_000.0          # raw BeamNG damage per unit of scaled damage (placeholder)
DAMAGE_DESTROYED_THRESHOLD = 0.9 # scaled damage at which the episode ends
FLIPPED_STEPS = 40               # steps upside down/on its side before terminating
STUCK_STEPS = 200                # steps below STUCK_SPEED (after the first 100) before terminating
STUCK_SPEED = 0.5
CURVATURE_LOOKAHEAD_M = (10.0, 30.0, 60.0)

PHYSICS_FPS = 60                 # deterministic-mode frames per simulated second
FRAMES_PER_ACTION = 3            # 3 frames at 60 fps = 20 Hz control
SETTLE_FRAMES = 30               # let the car settle on its suspension after a reset
# The user's own VSync / FPS-limit values while an env has them switched off.
FRAME_SETTINGS_BACKUP = Path(__file__).resolve().parents[2] / "runs" / ".beamng_frame_settings.json"


class DriftEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, stage: int = 1, max_episode_steps: int = 2000, seed: int | None = None,
                 track_params: TrackParams | None = None, new_track_every: int = 1,
                 frames_per_action: int = FRAMES_PER_ACTION, port: int = 25252):
        super().__init__()
        if stage not in STAGE_ACTIVE_DIMS:
            raise ValueError(f"stage must be one of {sorted(STAGE_ACTIVE_DIMS)}")
        self.stage = stage
        self.max_episode_steps = max_episode_steps
        self.track_params = track_params or TrackParams(hilliness=0.0)
        self.new_track_every = new_track_every
        self.frames_per_action = frames_per_action
        self.dt = frames_per_action / PHYSICS_FPS
        self.port = port

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(ACTION_DIMS),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self._track_seed = seed if seed is not None else 0
        self._episodes_on_track = 0
        self.current_gear = 1
        self._step_count = 0
        self._track = None
        self._cum_arc = None
        self._track_total_length = 0.0
        self._lookahead_idx = None
        self._prev_arc_length = 0.0
        self._prev_damage = 0.0
        self._prev_heading = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._off_track_steps = 0
        self._flipped_steps = 0
        self._stuck_steps = 0
        self._nearest_idx = 0
        self._lap_progress = 0.0

        self.bng = None
        self.loader: TrackLoader | None = None
        self.vehicle = None

    # ------------------------------------------------------------ BeamNG plumbing

    def connect(self) -> None:
        """Attach to (or launch) BeamNG and switch it to deterministic stepping."""
        if self.bng is not None:
            return
        self.bng = connect(self.port)
        self.bng.settings.set_deterministic(PHYSICS_FPS)
        # VSync / the FPS limiter pin every simulated frame (and every request, which
        # the game handles once per frame) to the monitor's refresh rate. Turn them
        # off while training; close() restores the user's own values. The originals
        # are also kept in a file, so a crashed/killed run doesn't make the next run
        # mistake "off" for the user's setting.
        current = self._lua(
            "local s = core_settings_settings "
            "local old = {vsync = s.getValue('vsync'), fpsLimitEnabled = s.getValue('fpsLimitEnabled')} "
            "s.setValue('vsync', false) s.setValue('fpsLimitEnabled', false) "
            "return jsonEncode(old)")
        if FRAME_SETTINGS_BACKUP.exists():
            self._saved_frame_settings = FRAME_SETTINGS_BACKUP.read_text(encoding="utf-8")
        else:
            self._saved_frame_settings = current
            FRAME_SETTINGS_BACKUP.parent.mkdir(parents=True, exist_ok=True)
            FRAME_SETTINGS_BACKUP.write_text(current, encoding="utf-8")
        self.loader = TrackLoader(self.bng, self.track_params, prefetch=2, keep_paused=True)

    def close(self):
        if self.loader is not None:
            self.loader.close()
        if self.bng is not None:
            try:
                if getattr(self, "_saved_frame_settings", None):
                    self._lua("local old = jsonDecode('" + self._saved_frame_settings + "') "
                              "for k, v in pairs(old) do core_settings_settings.setValue(k, v) end return 'ok'")
                    FRAME_SETTINGS_BACKUP.unlink(missing_ok=True)
                self.bng.settings.set_nondeterministic()
                self.bng.control.resume()
            finally:
                self.bng.disconnect()  # leave the game running
        self.bng = self.loader = self.vehicle = None

    def _lua(self, chunk: str) -> str:
        """Run a Lua chunk in the game engine and return its result as a string."""
        wrapped = f"local ok, r = pcall(function() {chunk} end) return ok and r or ('ERR ' .. tostring(r))"
        result = self.bng.control.queue_lua_command(wrapped, response=True)
        if isinstance(result, str) and result.startswith("ERR "):
            raise RuntimeError(f"BeamNG Lua error: {result}")
        return result

    def _attach_sensors(self) -> None:
        v = self.loader.vehicle
        if v is not self.vehicle:
            self.vehicle = v
            self.vehicle.sensors.attach("electrics", Electrics())
            self.vehicle.sensors.attach("damage", Damage())
        gearbox = "realistic" if "shift_delta" in STAGE_ACTIVE_DIMS[self.stage] else "arcade"
        self.vehicle.queue_lua_command(f'controller.mainController.setGearboxMode("{gearbox}")')

    # ------------------------------------------------------------ gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.connect()
        if seed is not None:
            self._track_seed, self._episodes_on_track = seed, 0
        new_track = self._track is None or self._episodes_on_track >= self.new_track_every
        if new_track:
            if self._track is not None:
                self._track_seed += 1
            prepared = self.loader.load(self._track_seed)
            self._set_track(prepared.track)
            self._episodes_on_track = 0
        else:
            self.loader.reset_car()
        self._episodes_on_track += 1
        self.bng.control.pause()
        self._attach_sensors()

        self._step_count = 0
        self.current_gear = 1
        self._off_track_steps = self._flipped_steps = self._stuck_steps = 0
        self._prev_action[:] = 0.0
        self._lap_progress = 0.0
        self.vehicle.control(steering=0.0, throttle=0.0, brake=0.0, parkingbrake=0.0, clutch=0.0)
        self.bng.control.step(SETTLE_FRAMES)

        raw = self._get_raw_vehicle_state()
        self._prev_heading = raw["heading"]
        raw["yaw_rate"] = 0.0
        obs, info = self._build_obs(raw)
        self._prev_arc_length = info["arc_length"]
        self._prev_damage = raw["damage"]
        info["track_seed"] = self._track_seed
        return obs, info

    def step(self, action: np.ndarray):
        action = self._apply_stage_mask(action)
        self.vehicle.control(**self._to_controls(action))
        self.bng.control.step(self.frames_per_action)
        self._prev_action = action[:2].copy()

        raw = self._get_raw_vehicle_state()
        obs, info = self._build_obs(raw)
        reward, parts = self._compute_reward(raw, info)
        self._lap_progress += parts["progress"]
        self._step_count += 1
        end_reason = self._termination_reason(raw)
        terminated = end_reason is not None
        truncated = self._step_count >= self.max_episode_steps
        # Extra per-step detail for logging/dashboards (src/training/dashboard_log.py).
        info.update(progress=self._lap_progress, laps=self._lap_progress / self._track_total_length,
                    speed=raw["v_long"], slip_angle=raw["slip_angle"], track_seed=self._track_seed,
                    reward_parts=parts, off_track=self._off_track_steps > 0,
                    pos_xy=raw["pos_xy"].tolist(),
                    end_reason=end_reason or ("time limit" if truncated else None))
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ actions

    def _apply_stage_mask(self, action: np.ndarray) -> np.ndarray:
        active = STAGE_ACTIVE_DIMS[self.stage]
        masked = np.clip(np.array(action, dtype=np.float32, copy=True), -1.0, 1.0)
        for i, name in enumerate(ACTION_DIMS):
            if name not in active:
                # Pedals map [-1, 1] -> [0, 1], so -1 is "released"; steer/shift neutral is 0.
                masked[i] = 0.0 if name in ("steer", "shift_delta") else -1.0
        return masked

    def _to_controls(self, action: np.ndarray) -> dict:
        steer, throttle, brake, handbrake, clutch, shift_delta_raw = (float(a) for a in action)
        pedal = lambda a: (a + 1.0) / 2.0  # noqa: E731
        controls = dict(steering=steer, throttle=pedal(throttle), brake=pedal(brake),
                        parkingbrake=pedal(handbrake), clutch=pedal(clutch))
        if "shift_delta" in STAGE_ACTIVE_DIMS[self.stage]:
            shift = int(np.round(np.clip(shift_delta_raw, -1, 1)))
            self.current_gear = int(np.clip(self.current_gear + shift, MIN_GEAR, MAX_GEAR))
            controls["gear"] = self.current_gear
        return controls

    # ------------------------------------------------------------ observation

    def _get_raw_vehicle_state(self) -> dict:
        self.vehicle.poll_sensors()
        state, elec = self.vehicle.state, self.vehicle.sensors["electrics"]
        damage = float(self.vehicle.sensors["damage"]["damage"] or 0.0)

        pos = np.asarray(state["pos"], dtype=np.float64)
        fwd = np.asarray(state["dir"], dtype=np.float64)
        up = np.asarray(state["up"], dtype=np.float64)
        vel = np.asarray(state["vel"], dtype=np.float64)
        heading = math.atan2(fwd[1], fwd[0])
        f2 = fwd[:2] / (np.linalg.norm(fwd[:2]) + 1e-9)
        left = np.array([-f2[1], f2[0]])
        v_long, v_lat = float(vel[:2] @ f2), float(vel[:2] @ left)
        slip = math.atan2(v_lat, abs(v_long)) if math.hypot(v_long, v_lat) > 1.0 else 0.0

        yaw_rate = _wrap_angle(heading - self._prev_heading) / self.dt
        self._prev_heading = heading
        maxrpm = float(elec.get("maxrpm") or 8000.0)
        return {
            "pos_xy": pos[:2],
            "heading": heading,
            "up_z": float(up[2]),
            "v_long": v_long,
            "v_lat": v_lat,
            "yaw_rate": yaw_rate,
            "slip_angle": slip,
            "wheelspin": float(elec.get("wheelspeed") or 0.0) - v_long,
            "rpm_frac": float(elec.get("rpm") or 0.0) / maxrpm,
            "gear": float(elec.get("gear_index") or 0.0),
            "damage": damage / DAMAGE_SCALE,
        }

    def _build_obs(self, raw: dict) -> tuple[np.ndarray, dict]:
        lateral_offset, arc_length, heading_error, curv_ahead = self._project_to_centerline(
            raw["pos_xy"], raw["heading"]
        )
        half_width = self._track["widths"][self._nearest_idx] / 2.0
        obs = np.array(
            [
                lateral_offset / half_width,
                heading_error,
                *curv_ahead,
                raw["v_long"],
                raw["v_lat"],
                raw["yaw_rate"],
                raw["slip_angle"],
                raw["wheelspin"],
                raw["rpm_frac"],
                raw["gear"],
                raw["damage"],
                *self._prev_action,
            ],
            dtype=np.float32,
        )
        info = {"lateral_offset": lateral_offset, "arc_length": arc_length}
        return obs, info

    def _set_track(self, track: dict) -> None:
        self._track = track
        self._cum_arc, self._track_total_length = _cumulative_arc_length(track["centerline"])
        spacing = self._track_total_length / len(track["centerline"])
        self._lookahead_idx = [int(round(d / spacing)) for d in CURVATURE_LOOKAHEAD_M]

    def _project_to_centerline(self, pos_xy: np.ndarray, heading: float):
        """Nearest-point projection of a world position onto the track centerline.

        Returns (signed_lateral_offset, arc_length, heading_error, curvature_ahead[3]).
        """
        centerline = P.world_centerline(self._track)
        diffs = centerline - pos_xy
        idx = int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
        self._nearest_idx = idx
        n = len(centerline)
        tangent = centerline[(idx + 1) % n] - centerline[(idx - 1) % n]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        normal = np.array([-tangent[1], tangent[0]])

        lateral_offset = float(np.dot(pos_xy - centerline[idx], normal))
        heading_error = _wrap_angle(heading - math.atan2(tangent[1], tangent[0]))
        arc_length = float(self._cum_arc[idx])
        curv = self._track["curvature"]
        curvature_ahead = [float(curv[(idx + k) % n]) for k in self._lookahead_idx]
        return lateral_offset, arc_length, heading_error, curvature_ahead

    # ------------------------------------------------------------ reward / termination

    def _unwrap_progress(self, arc_length: float) -> float:
        """Delta arc-length since last step, handling lap-boundary wraparound."""
        total = self._track_total_length
        delta = arc_length - self._prev_arc_length
        if delta < -total / 2:
            delta += total
        elif delta > total / 2:
            delta -= total
        return delta

    def _drift_bump(self, slip_angle_abs: float) -> float:
        if slip_angle_abs <= THETA_MAX:
            return slip_angle_abs / THETA_MAX
        return max(0.0, 1.0 - (slip_angle_abs - THETA_MAX) / THETA_FALLOFF)

    def _compute_reward(self, raw: dict, info: dict) -> tuple[float, dict]:
        """Total reward and its parts (progress, time, offtrack, damage, drift); parts sum to the total."""
        progress_reward = self._unwrap_progress(info["arc_length"])
        self._prev_arc_length = info["arc_length"]
        parts = dict(progress=progress_reward, time=-TIME_PENALTY, offtrack=0.0, damage=0.0, drift=0.0)

        # Road width varies where the generator narrowed it (adapt_width).
        half_width = self._track["widths"][self._nearest_idx] / 2.0
        lateral = abs(info["lateral_offset"])
        if lateral > half_width:
            # Grows with distance past the road edge (no cliff at the edge itself).
            parts["offtrack"] = -K_OFFTRACK * (lateral - half_width)
            self._off_track_steps += 1
        else:
            self._off_track_steps = 0

        delta_damage = max(0.0, raw["damage"] - self._prev_damage)
        parts["damage"] = -K_DAMAGE * delta_damage
        self._prev_damage = raw["damage"]

        if self.stage >= 3:
            parts["drift"] = progress_reward * self._drift_bump(abs(raw["slip_angle"]))

        return float(sum(parts.values())), parts

    def _termination_reason(self, raw: dict) -> str | None:
        """Why the episode should end now, or None to keep going."""
        self._flipped_steps = self._flipped_steps + 1 if raw["up_z"] < 0.3 else 0
        slow = abs(raw["v_long"]) < STUCK_SPEED and self._step_count > 100
        self._stuck_steps = self._stuck_steps + 1 if slow else 0
        if raw["damage"] >= DAMAGE_DESTROYED_THRESHOLD:
            return "wrecked"
        if self._off_track_steps > RECOVERY_TIMEOUT:
            return "off track"
        if self._flipped_steps > FLIPPED_STEPS:
            return "flipped"
        if self._stuck_steps > STUCK_STEPS:
            return "stuck"
        return None


def _cumulative_arc_length(centerline: np.ndarray) -> tuple[np.ndarray, float]:
    seg_lengths = np.linalg.norm(np.diff(centerline, axis=0, append=centerline[:1]), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)[:-1]])
    return cum, float(seg_lengths.sum())


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
