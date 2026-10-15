"""Gymnasium env wrapping BeamNGpy for the drift agent.

Action vector (always 6-dim; curriculum stage decides which dims are "live" —
inactive dims are forced to a safe neutral value before being sent to BeamNG,
per CLAUDE.md's progressive-unlock curriculum):

    [steer, throttle, brake, handbrake, clutch, shift_delta]

shift_delta is continuous, rounded/clipped to {-1, 0, +1} at the env boundary
and applied to persistent gear state. Reverse is simply the gear rung below
1st — no separate reverse flag.

Observation vector: vehicle state relative to the track centerline (lateral
offset, heading error, upcoming curvature, speed, slip angle, 4 wheel speeds,
current gear, RPM, cumulative damage) — see CLAUDE.md "Reward function".

Only `_get_raw_vehicle_state` depends on BeamNG and is stubbed with a TODO
until BeamNG.drive is installed and a connection is verified
(scripts/test_beamng_connection.py). Everything downstream of it — centerline
projection, reward shaping, termination, curriculum masking, gear state — is
real math that runs today against the stub's placeholder values.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.track.generator import generate_track

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
DAMAGE_DESTROYED_THRESHOLD = 0.9
CURVATURE_LOOKAHEAD = 15         # samples ahead for "curvature ahead" observation


class DriftEnv(gym.Env):
    def __init__(self, stage: int = 1, max_episode_steps: int = 2000, seed: int | None = None):
        super().__init__()
        if stage not in STAGE_ACTIVE_DIMS:
            raise ValueError(f"stage must be one of {sorted(STAGE_ACTIVE_DIMS)}")
        self.stage = stage
        self.max_episode_steps = max_episode_steps
        self._rng_seed = seed

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(ACTION_DIMS),), dtype=np.float32)

        # [lateral_offset, heading_error, curvature_ahead, speed, slip_angle,
        #  wheel_speed_fl, wheel_speed_fr, wheel_speed_rl, wheel_speed_rr,
        #  current_gear, rpm, damage]
        obs_dim = 12
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_gear = 1
        self._step_count = 0
        self._track = None
        self._cum_arc = None
        self._track_total_length = 0.0
        self._prev_arc_length = 0.0
        self._prev_damage = 0.0
        self._off_track_steps = 0
        self._bng = None  # TODO: beamngpy.BeamNGpy connection, set up in connect()

    def connect(self, host: str = "localhost", port: int = 25252):
        """Open the BeamNGpy connection. Call once before using the env.

        TODO: implement once BeamNG.drive is installed and
        scripts/test_beamng_connection.py confirms a working connection.
        """
        raise NotImplementedError("Wire up beamngpy.BeamNGpy(...) here once BeamNG is verified.")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self.current_gear = 1
        self._track = generate_track(seed=seed if seed is not None else self._rng_seed)
        self._cum_arc, self._track_total_length = _cumulative_arc_length(self._track["centerline"])

        # TODO: place self._track's centerline/width into BeamNG's GridMap via
        # the road/decal road API, respawn the vehicle at the track start.

        raw = self._get_raw_vehicle_state()
        obs, info = self._build_obs(raw)
        self._prev_arc_length = info["arc_length"]
        self._prev_damage = raw["damage"]
        self._off_track_steps = 0
        return obs, {}

    def step(self, action: np.ndarray):
        action = self._apply_stage_mask(action)
        steer, throttle, brake, handbrake, clutch, shift_delta_raw = action

        shift_delta = int(np.round(np.clip(shift_delta_raw, -1, 1)))
        self.current_gear = int(np.clip(self.current_gear + shift_delta, MIN_GEAR, MAX_GEAR))

        # TODO: send to BeamNGpy, e.g.
        # self._bng.vehicle.control(throttle=throttle, steering=steer, brake=brake,
        #                            parkingbrake=handbrake, clutch=clutch, gear=self.current_gear)
        # self._bng.step(...)

        raw = self._get_raw_vehicle_state()
        obs, info = self._build_obs(raw)
        reward = self._compute_reward(raw, info)
        self._step_count += 1
        terminated = self._is_terminated(raw)
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def _apply_stage_mask(self, action: np.ndarray) -> np.ndarray:
        active = STAGE_ACTIVE_DIMS[self.stage]
        masked = np.array(action, dtype=np.float32, copy=True)
        for i, name in enumerate(ACTION_DIMS):
            if name not in active:
                masked[i] = 0.0  # neutral: no brake/handbrake/clutch input, hold gear
        return masked

    def _get_raw_vehicle_state(self) -> dict:
        """Pull raw state from BeamNGpy sensors (Electrics, State, damage).

        TODO: replace this stub with real BeamNGpy reads once the connection
        is verified. Returns neutral defaults so the rest of the pipeline
        (projection, reward, termination) is exercisable without BeamNG.
        """
        return {
            "pos_xy": np.zeros(2, dtype=np.float32),
            "heading": 0.0,
            "speed": 0.0,
            "slip_angle": 0.0,
            "wheel_speeds": np.zeros(4, dtype=np.float32),
            "rpm": 0.0,
            "damage": 0.0,
        }

    def _build_obs(self, raw: dict) -> tuple[np.ndarray, dict]:
        lateral_offset, arc_length, heading_error, curvature_ahead = self._project_to_centerline(
            raw["pos_xy"], raw["heading"]
        )
        obs = np.array(
            [
                lateral_offset,
                heading_error,
                curvature_ahead,
                raw["speed"],
                raw["slip_angle"],
                *raw["wheel_speeds"],
                float(self.current_gear),
                raw["rpm"],
                raw["damage"],
            ],
            dtype=np.float32,
        )
        info = {"lateral_offset": lateral_offset, "arc_length": arc_length}
        return obs, info

    def _project_to_centerline(self, pos_xy: np.ndarray, heading: float):
        """Nearest-point projection of a world position onto the track centerline.

        Returns (signed_lateral_offset, arc_length, heading_error, curvature_ahead).
        """
        centerline = self._track["centerline"]
        diffs = centerline - pos_xy
        idx = int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
        nxt = (idx + 1) % len(centerline)
        prv = (idx - 1) % len(centerline)

        tangent = centerline[nxt] - centerline[prv]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        normal = np.array([-tangent[1], tangent[0]])

        lateral_offset = float(np.dot(pos_xy - centerline[idx], normal))
        track_angle = float(np.arctan2(tangent[1], tangent[0]))
        heading_error = _wrap_angle(heading - track_angle)
        arc_length = float(self._cum_arc[idx])
        curvature_ahead = float(self._track["curvature"][(idx + CURVATURE_LOOKAHEAD) % len(centerline)])

        return lateral_offset, arc_length, heading_error, curvature_ahead

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

    def _compute_reward(self, raw: dict, info: dict) -> float:
        progress_reward = self._unwrap_progress(info["arc_length"])
        self._prev_arc_length = info["arc_length"]

        reward = progress_reward - TIME_PENALTY

        half_width = self._track["width"] / 2.0
        lateral = abs(info["lateral_offset"])
        if lateral > half_width:
            reward -= K_OFFTRACK * lateral
            self._off_track_steps += 1
        else:
            self._off_track_steps = 0

        delta_damage = max(0.0, raw["damage"] - self._prev_damage)
        reward -= K_DAMAGE * delta_damage
        self._prev_damage = raw["damage"]

        if self.stage >= 3:
            reward += progress_reward * self._drift_bump(abs(raw["slip_angle"]))

        return float(reward)

    def _is_terminated(self, raw: dict) -> bool:
        if raw["damage"] >= DAMAGE_DESTROYED_THRESHOLD:
            return True
        if self._off_track_steps > RECOVERY_TIMEOUT:
            return True
        return False


def _cumulative_arc_length(centerline: np.ndarray) -> tuple[np.ndarray, float]:
    seg_lengths = np.linalg.norm(np.diff(centerline, axis=0, append=centerline[:1]), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)[:-1]])
    return cum, float(seg_lengths.sum())


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
