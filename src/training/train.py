"""Train (or curriculum-resume) the drift policy.

This is where the neural network actually gets built: Stable-Baselines3's
PPO constructs the actor-critic MLP for us from `policy_kwargs["net_arch"]`
matched against our env's action/observation space (see CLAUDE.md "Learning
approach") — we are not hand-writing PyTorch layers, SB3 does that from this
spec. Each curriculum stage fine-tunes from the previous stage's checkpoint
by loading it with a new env (new active action dims), rather than starting
over from random weights.

Observations and rewards are normalised with VecNormalize (raw values range
from radians to thousands of RPM); its running statistics are saved next to
every checkpoint and must be loaded with it.

Usage (BeamNG is launched with its API port open, or attached to, automatically
on Windows and Linux; normally you start runs from Drift Studio's Training tab):
    python -m src.training.train --stage 1
    python -m src.training.train --stage 1 --resume-from models/stage1/latest.zip --seed 5000
    python -m src.training.train --stage 2 --resume-from models/stage1/final.zip
    python scripts/drift_studio.py   # start/stop runs + live dashboards (or start_studio.bat)
    tensorboard --logdir runs        # raw SB3 curves

Ctrl+C stops training and saves a checkpoint first.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.drift_env import DriftEnv
from src.envs.track_loader import TrackParams
from src.training.dashboard_log import DashboardLogger

NET_ARCH = [256, 256]  # shared trunk; SB3 attaches separate actor/critic heads on top
MODELS_DIR = Path("models")
RUNS_DIR = Path("runs")


class StopOnFile(BaseCallback):
    """Ends training cleanly when `path` appears (how Drift Studio's Stop button works)."""

    def __init__(self, path: Path | None):
        super().__init__()
        self.path = path
        self.requested = False

    def _on_step(self) -> bool:
        if self.path is not None and self.n_calls % 20 == 0 and self.path.exists():
            self.requested = True
            print("Stop requested - finishing up and saving", flush=True)
            return False
        return True


class EpisodeStats(BaseCallback):
    """Logs laps completed and mean speed per episode to TensorBoard."""

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.logger.record_mean("drift/laps", info.get("laps", 0.0))
                self.logger.record_mean("drift/progress_m", info.get("progress", 0.0))
        return True


def stats_path(checkpoint: Path) -> Path:
    """VecNormalize stats saved with a checkpoint: ours (x_vecnormalize.pkl) or
    CheckpointCallback's (ppo_N_steps.zip -> ppo_vecnormalize_N_steps.pkl)."""
    ours = checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl")
    sb3 = checkpoint.with_name(checkpoint.stem.replace("ppo_", "ppo_vecnormalize_", 1) + ".pkl")
    return sb3 if not ours.exists() and sb3.exists() else ours


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint .zip to fine-tune from")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0, help="first track seed (later episodes use the next seeds)")
    parser.add_argument("--hills", type=float, default=0.0, help="terrain hilliness (0 = flat island)")
    parser.add_argument("--new-track-every", type=int, default=1, help="episodes per generated track")
    parser.add_argument("--save-every", type=int, default=20_000, help="checkpoint every N steps")
    parser.add_argument("--run-name", type=str, default=None, help="dashboard run folder (default: stageN_<time>)")
    parser.add_argument("--stop-file", type=str, default=None, help="stop cleanly (and save) when this file appears")
    args = parser.parse_args()

    out_dir = MODELS_DIR / f"stage{args.stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        env = DriftEnv(stage=args.stage, seed=args.seed, track_params=TrackParams(hilliness=args.hills),
                       new_track_every=args.new_track_every)
        return Monitor(env, info_keywords=("laps", "progress"))

    venv = DummyVecEnv([make_env])
    if args.resume_from:
        ckpt = Path(args.resume_from)
        venv = VecNormalize.load(str(stats_path(ckpt)), venv)
        venv.training = True
        model = PPO.load(ckpt, env=venv, tensorboard_log=str(RUNS_DIR))
        venv.seed(args.seed)  # PPO.load re-seeds the env with the old run's seed; use --seed
        print(f"Resumed from {ckpt}, continuing as stage {args.stage} from track seed {args.seed}")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy",
            venv,
            policy_kwargs=dict(net_arch=NET_ARCH),
            n_steps=2048,
            batch_size=128,
            gamma=0.99,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(RUNS_DIR),
        )
        print(f"Fresh model for stage {args.stage}")

    checkpoints = CheckpointCallback(save_freq=args.save_every, save_path=str(out_dir), name_prefix="ppo",
                                     save_vecnormalize=True)
    run_name = args.run_name or f"stage{args.stage}_{time.strftime('%Y%m%d-%H%M%S')}"
    stop = StopOnFile(Path(args.stop_file) if args.stop_file else None)
    dashboard = DashboardLogger(run_name, meta=dict(vars(args), run=run_name, models_dir=str(out_dir)))

    def save(name: str) -> None:
        path = out_dir / f"{name}.zip"
        model.save(path)
        venv.save(str(stats_path(path)))
        print(f"Saved {path} (+ {stats_path(path).name})")

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=[checkpoints, EpisodeStats(), dashboard, stop],
                    tb_log_name=f"stage{args.stage}", reset_num_timesteps=not args.resume_from)
        if not stop.requested:
            save("final")
    except KeyboardInterrupt:
        print("Interrupted — saving before exit")
    finally:
        save("latest")
        venv.close()
        if stop.path is not None:
            stop.path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
