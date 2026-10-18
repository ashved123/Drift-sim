"""Watch a trained checkpoint drive in BeamNG.

    python scripts/watch_policy.py models/stage1/latest.zip
    python scripts/watch_policy.py models/stage1/final.zip --episodes 3 --seed 500

Uses the checkpoint's saved VecNormalize stats and deterministic actions, on
track seeds the policy didn't train on by default (--seed 100000). Prints laps,
progress and reward per episode. Can be run while training is stopped (both
need the game to themselves).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from src.envs.drift_env import DriftEnv  # noqa: E402
from src.envs.track_loader import TrackParams  # noqa: E402
from src.training.train import stats_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", type=int, default=None, help="defaults to the stageN folder the checkpoint is in")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100_000, help="first track seed")
    parser.add_argument("--hills", type=float, default=0.0)
    args = parser.parse_args()

    stage = args.stage or int(args.checkpoint.parent.name.removeprefix("stage") or 1)
    venv = DummyVecEnv([lambda: DriftEnv(stage=stage, seed=args.seed, track_params=TrackParams(hilliness=args.hills))])
    venv = VecNormalize.load(str(stats_path(args.checkpoint)), venv)
    venv.training, venv.norm_reward = False, False
    model = PPO.load(args.checkpoint, env=venv)
    venv.seed(args.seed)  # PPO.load re-seeds the env with the training seed; use ours

    obs = venv.reset()
    episode, total = 1, 0.0
    try:
        while episode <= args.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = venv.step(action)
            total += float(reward[0])
            if done[0]:
                info = infos[0]
                print(f"episode {episode} (track seed {info['track_seed']}): {info['laps']:.2f} laps, "
                      f"{info['progress']:.0f} m, reward {total:.0f}")
                episode, total = episode + 1, 0.0
    finally:
        venv.close()


if __name__ == "__main__":
    main()
