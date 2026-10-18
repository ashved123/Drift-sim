"""Sanity-check DriftEnv against the real game before training on it.

Drives the env with a simple hand-written controller (steer back toward the
centerline, hold a target speed) and reports what a training run would see:
simulation speed (env steps per second), reward, progress along the track,
off-track steps and why the episode ended. If the controller laps the track
with a positive reward, the env's signs and units are sane; if it can't, the
env (not the RL) needs fixing.

    python scripts/test_drift_env.py                 # one episode, controller
    python scripts/test_drift_env.py --random 300    # also 300 random-action steps

Close the track control panel first — both would try to control the game.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.drift_env import DriftEnv  # noqa: E402


def controller(obs: np.ndarray, target_speed: float, steer_sign: float) -> np.ndarray:
    lat_norm, heading_err, curv10 = obs[0], obs[1], obs[2]
    speed = obs[5]
    # Positive lateral offset / heading error = car is left of / pointing left of the
    # track; steer right to correct, and lead into corners using curvature ahead.
    steer = steer_sign * (0.6 * lat_norm + 1.5 * heading_err - 25.0 * curv10)
    throttle = np.clip(0.4 * (target_speed - speed), -1.0, 1.0)
    return np.array([np.clip(steer, -1, 1), throttle, -1, -1, -1, 0], dtype=np.float32)


def run_episode(env: DriftEnv, policy, max_steps: int, label: str) -> None:
    obs, info = env.reset()
    t0 = time.time()
    total, off, steps, reason = 0.0, 0, 0, "step limit"
    for steps in range(1, max_steps + 1):
        obs, r, terminated, truncated, info = env.step(policy(obs))
        total += r
        off += abs(obs[0]) > 1.0
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            if terminated:
                reason += (" (off track too long)" if env._off_track_steps > 0 else
                           " (flipped)" if env._flipped_steps > 0 else
                           " (stuck)" if env._stuck_steps > 0 else " (damage)")
            break
    dt = time.time() - t0
    print(f"[{label}] track seed {info['track_seed']}: {steps} steps in {dt:.1f}s "
          f"= {steps / dt:.1f} steps/s ({steps * env.dt / dt:.1f}x real time)")
    print(f"[{label}] reward {total:.1f} | progress {info['progress']:.0f} m = {info['laps']:.2f} laps | "
          f"off-track {off} steps | top speed {info['speed']:.1f} m/s at end | ended: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--target-speed", type=float, default=14.0)
    parser.add_argument("--steer-sign", type=float, default=1.0, help="flip if the car steers away from the track")
    parser.add_argument("--random", type=int, default=0, help="also run this many random-action steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = DriftEnv(stage=1, seed=args.seed, max_episode_steps=args.steps)
    try:
        run_episode(env, lambda o: controller(o, args.target_speed, args.steer_sign), args.steps, "controller")
        if args.random:
            run_episode(env, lambda o: env.action_space.sample(), args.random, "random")
    finally:
        env.close()


if __name__ == "__main__":
    main()
