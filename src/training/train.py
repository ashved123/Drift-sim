"""Train (or curriculum-resume) the drift policy.

This is where the neural network actually gets built: Stable-Baselines3's
PPO constructs the actor-critic MLP for us from `policy_kwargs["net_arch"]`
matched against our env's action/observation space (see CLAUDE.md "Learning
approach") — we are not hand-writing PyTorch layers, SB3 does that from this
spec. Each curriculum stage fine-tunes from the previous stage's checkpoint
by loading it with a new env (new active action dims), rather than starting
over from random weights.

Usage:
    python -m src.training.train --stage 1
    python -m src.training.train --stage 2 --resume-from models/stage1_ppo.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.envs.drift_env import DriftEnv

NET_ARCH = [256, 256]  # shared trunk; SB3 attaches separate actor/critic heads on top
MODELS_DIR = Path("models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint to fine-tune from")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    env = Monitor(DriftEnv(stage=args.stage, seed=args.seed))

    if args.resume_from:
        model = PPO.load(args.resume_from, env=env)
        print(f"Resumed from {args.resume_from}, continuing as stage {args.stage}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=NET_ARCH),
            verbose=1,
            seed=args.seed,
        )
        print(f"Fresh model for stage {args.stage}")

    model.learn(total_timesteps=args.total_timesteps)

    out_path = MODELS_DIR / f"stage{args.stage}_ppo"
    model.save(out_path)
    print(f"Saved checkpoint to {out_path}.zip")


if __name__ == "__main__":
    main()
