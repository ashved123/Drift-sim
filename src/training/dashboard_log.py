"""Training logger feeding the live dashboard (scripts/drift_studio.py).

Writes one folder per training run, runs/dash/<run>/:
  meta.json            stage, settings, start time, checkpoint folder
  episodes.jsonl       one line per finished episode: laps, distance, speed,
                       slip, % off track, end reason, reward split into parts
  updates.jsonl        one line per PPO update: losses, entropy, KL, clip
                       fraction, explained variance, learning rate
  status.json          rewritten every 10 steps (~0.5 s): steps, speed, the
                       episode being driven, and the last ~15 s of telemetry
                       (speed, slip, steer, throttle, lateral offset, position)
  current_track.json   shape of the track currently being driven
  latest_episode.json  the last finished episode's driving line + its track
  best_episode.json    same, for the episode that got furthest
  ppo_viz.json         what the last PPO update did, measured on a 512-step
                       sample of its rollout: probability-ratio histogram vs
                       the clip range, advantage histogram, critic predictions
                       vs actual returns, policy exploration noise per control
  ppo_hist.jsonl       one line per update: exploration noise, % clipped, ...

All plain JSON so the dashboard (or anything else) can read it while training
is running. Assumes a single env (DummyVecEnv of one DriftEnv), as train.py builds.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from src.envs import track_placement as P

DASH_DIR = Path("runs") / "dash"
TRACK_STEP = 4      # keep every 4th centerline sample (~4 m) for the map
TRAJ_EVERY = 4      # record the car's position every 4 env steps (0.2 s)
STATUS_EVERY = 10   # rewrite status.json every N env steps (~0.5 s at 20 Hz)
LIVE_SAMPLES = 300  # telemetry kept for the live view (~15 s at 20 Hz)
PPO_SAMPLE = 512    # rollout steps re-evaluated after each update for ppo_viz.json
ACTION_NAMES = ["steer", "throttle", "brake", "handbrake", "clutch", "shift"]


class DashboardLogger(BaseCallback):
    def __init__(self, run_name: str, meta: dict):
        super().__init__()
        self.dir = DASH_DIR / run_name
        self.meta = meta
        self._tracks: dict[int, dict] = {}
        self._episode = 0
        self._best_progress = -np.inf
        self._last_update_ts = -1
        self._t_start = time.time()
        self._sps_window: list[tuple[float, int]] = []
        self._live: deque = deque(maxlen=LIVE_SAMPLES)
        self._phase = "starting"
        self._iteration = 0
        self._snap: dict | None = None
        self._last_info: dict = {}
        self._last_done = False
        self._reset_episode()

    # ------------------------------------------------------------ helpers

    def _env(self):
        env = self.training_env
        while hasattr(env, "venv"):  # VecNormalize -> DummyVecEnv
            env = env.venv
        return env.envs[0].unwrapped

    def _reset_episode(self) -> None:
        self._steps = 0
        self._parts: dict[str, float] = {}
        self._speeds: list[float] = []
        self._abs_slip = 0.0
        self._offtrack = 0
        self._traj: list[list[float]] = []
        self._t_episode = time.time()
        self._reward_so_far = 0.0

    # Logging must never take training down. On Windows a file can't be replaced
    # while another process (the dashboard, OneDrive sync, antivirus) has it open,
    # so writes retry briefly and otherwise skip that update.

    def _append(self, name: str, record: dict) -> None:
        line = json.dumps(record) + "\n"
        for attempt in range(5):
            try:
                with open(self.dir / name, "a", encoding="utf-8") as f:
                    f.write(line)
                return
            except OSError:
                time.sleep(0.05 * (attempt + 1))
        print(f"[dashboard] couldn't append to {name} (file locked); record skipped")

    def _write(self, name: str, data: dict) -> None:
        tmp = self.dir / (name + ".tmp")
        for attempt in range(5):
            try:
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.replace(self.dir / name)  # atomic, so the dashboard never reads half a file
                return
            except OSError:
                time.sleep(0.05 * (attempt + 1))
        # status.json is rewritten every 10 steps anyway; a skipped write is harmless.

    def _cache_track(self, seed: int) -> None:
        track = self._env()._track
        if seed in self._tracks or track is None:
            return
        c = P.world_centerline(track)[::TRACK_STEP]
        self._tracks = {seed: dict(x=np.round(c[:, 0], 1).tolist(), y=np.round(c[:, 1], 1).tolist(),
                                   w=np.round(track["widths"][::TRACK_STEP], 1).tolist(),
                                   length=float(self._env()._track_total_length))}
        self._write("current_track.json", dict(track_seed=int(seed), **self._tracks[seed]))

    # ------------------------------------------------------------ SB3 hooks

    def _on_training_start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write("meta.json", dict(self.meta, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                                      start_timesteps=int(self.num_timesteps)))

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = bool(self.locals["dones"][0])
        self._last_info, self._last_done = info, done
        if "reward_parts" not in info:
            return True
        seed = info["track_seed"]
        if not done:
            self._cache_track(seed)  # at a done step the env has already moved to the next track

        self._steps += 1
        self._reward_so_far += sum(float(v) for v in info["reward_parts"].values())
        steer, throttle = (float(a) for a in np.clip(self.locals["actions"][0][:2], -1, 1))
        x, y = info["pos_xy"]
        self._live.append([int(self.num_timesteps), round(float(info["speed"]), 2),
                           round(float(np.degrees(info["slip_angle"])), 1), round(steer, 2),
                           round((throttle + 1) / 2, 2), round(float(info["lateral_offset"]), 2),
                           round(x, 1), round(y, 1), int(seed)])
        for k, v in info["reward_parts"].items():
            self._parts[k] = self._parts.get(k, 0.0) + float(v)
        self._speeds.append(float(info["speed"]))
        self._abs_slip += abs(float(info["slip_angle"]))
        self._offtrack += bool(info["off_track"])
        if self._steps % TRAJ_EVERY == 1 or done:
            x, y = info["pos_xy"]
            self._traj.append([round(x, 1), round(y, 1), round(float(info["speed"]), 1),
                               round(float(np.degrees(info["slip_angle"])), 1)])

        if done:
            self._finish_episode(info)
        if self.num_timesteps % STATUS_EVERY == 0:
            self._write_status(info, done)
        return True

    def _on_rollout_start(self) -> None:
        self._log_update()
        self._ppo_after_update()
        self._phase = "collecting"

    def _on_rollout_end(self) -> None:
        # Advantages/returns are computed by now; PPO's train() runs next.
        self._snapshot_rollout()
        self._phase = "updating"
        self._iteration += 1
        if self._last_info.get("reward_parts"):
            self._write_status(self._last_info, self._last_done)

    def _on_training_end(self) -> None:
        self._log_update()  # the last PPO update has no rollout after it
        self._ppo_after_update()
        self._phase = "finished"
        if self._last_info.get("reward_parts"):
            self._write_status(self._last_info, self._last_done)

    # ------------------------------------------------------------ PPO visualisation data

    def _snapshot_rollout(self) -> None:
        buf = self.model.rollout_buffer
        n = buf.buffer_size * buf.n_envs
        idx = np.random.default_rng(self._iteration).choice(n, size=min(PPO_SAMPLE, n), replace=False)

        def flat(a):
            return a.reshape(n, *a.shape[2:])[idx]

        self._snap = dict(obs=flat(buf.observations).astype(np.float32), act=flat(buf.actions).astype(np.float32),
                          old_logp=flat(buf.log_probs), adv=flat(buf.advantages), ret=flat(buf.returns),
                          old_val=flat(buf.values), timesteps=int(self.num_timesteps))

    def _ppo_after_update(self) -> None:
        """Re-evaluate the snapshot with the just-updated policy and write ppo_viz.json."""
        snap, self._snap = self._snap, None
        if snap is None:
            return
        policy = self.model.policy
        with torch.no_grad():
            values, logp, _ = policy.evaluate_actions(torch.as_tensor(snap["obs"], device=policy.device),
                                                      torch.as_tensor(snap["act"], device=policy.device))
        ratio = np.exp(logp.cpu().numpy().ravel() - snap["old_logp"].ravel())
        eps = float(self.model.clip_range(self.model._current_progress_remaining))
        adv, ret, old_val = snap["adv"].ravel(), snap["ret"].ravel(), snap["old_val"].ravel()
        new_val = values.cpu().numpy().ravel()
        std = policy.log_std.exp().detach().cpu().numpy().ravel() if hasattr(policy, "log_std") else np.array([])

        def hist(x, edges):
            counts, edges = np.histogram(np.clip(x, edges[0], edges[-1]), bins=edges)
            return dict(edges=np.round(edges, 4).tolist(), counts=counts.tolist())

        lo, hi = np.percentile(adv, [1, 99])
        span = max(abs(lo), abs(hi), 1e-6)
        var_ret = float(np.var(ret))
        pick = np.random.default_rng(0).choice(len(ret), size=min(300, len(ret)), replace=False)
        clipped = np.abs(ratio - 1) > eps
        viz = dict(
            timesteps=snap["timesteps"], iteration=self._iteration, clip_range=eps, n_epochs=self.model.n_epochs,
            n_steps=self.model.n_steps, batch_size=self.model.batch_size, sample=len(ratio),
            ratio=hist(ratio, np.linspace(1 - 3 * eps, 1 + 3 * eps, 49)),
            clipped_frac=float(clipped.mean()), clipped_up=float((ratio > 1 + eps).mean()),
            clipped_down=float((ratio < 1 - eps).mean()),
            adv=hist(adv, np.linspace(-span, span, 41)), adv_pos_frac=float((adv > 0).mean()),
            values=dict(old=np.round(old_val[pick], 3).tolist(), new=np.round(new_val[pick], 3).tolist(),
                        ret=np.round(ret[pick], 3).tolist()),
            explained_variance_before=float(1 - np.var(ret - old_val) / var_ret) if var_ret > 0 else None,
            explained_variance_after=float(1 - np.var(ret - new_val) / var_ret) if var_ret > 0 else None,
            action_std=dict(zip(ACTION_NAMES, np.round(std, 4).tolist())),
        )
        self._write("ppo_viz.json", viz)
        self._append("ppo_hist.jsonl", dict(
            timesteps=viz["timesteps"], iteration=viz["iteration"], clipped_frac=round(viz["clipped_frac"], 4),
            ratio_mean=round(float(ratio.mean()), 4), adv_std=round(float(adv.std()), 4),
            ev_before=viz["explained_variance_before"], ev_after=viz["explained_variance_after"],
            action_std=viz["action_std"]))

    def _log_update(self) -> None:
        # PPO's train() records train/* values after the previous dump, so they
        # are still in the logger when the next rollout starts.
        values = {k.split("/", 1)[1]: float(v) for k, v in self.model.logger.name_to_value.items()
                  if k.startswith("train/")}
        if values and self.num_timesteps != self._last_update_ts:
            self._last_update_ts = self.num_timesteps
            self._append("updates.jsonl", dict(timesteps=int(self.num_timesteps), time=round(time.time(), 1),
                                               **values))

    # ------------------------------------------------------------ records

    def _finish_episode(self, info: dict) -> None:
        self._episode += 1
        seed = info["track_seed"]
        track = self._tracks.get(seed)
        speeds = np.array(self._speeds)
        record = dict(
            episode=self._episode, timesteps=int(self.num_timesteps), time=round(time.time(), 1),
            wall_seconds=round(time.time() - self._t_episode, 1), track_seed=int(seed),
            track_length=round(track["length"], 1) if track else None, steps=self._steps,
            laps=round(float(info["laps"]), 3), progress=round(float(info["progress"]), 1),
            reward=round(sum(self._parts.values()), 2),
            parts={k: round(v, 2) for k, v in self._parts.items()},
            mean_speed=round(float(speeds.mean()), 2), max_speed=round(float(speeds.max()), 2),
            mean_abs_slip_deg=round(float(np.degrees(self._abs_slip / self._steps)), 2),
            offtrack_pct=round(100.0 * self._offtrack / self._steps, 1),
            end_reason=info.get("end_reason") or "unknown",
        )
        self._append("episodes.jsonl", record)
        replay = dict(episode=self._episode, track_seed=int(seed), laps=record["laps"], reward=record["reward"],
                      end_reason=record["end_reason"], track=track, traj=self._traj)
        self._write("latest_episode.json", replay)
        if record["progress"] > self._best_progress:
            self._best_progress = record["progress"]
            self._write("best_episode.json", replay)
        self._reset_episode()

    def _write_status(self, info: dict, done: bool) -> None:
        now = time.time()
        self._sps_window = [(t, n) for t, n in self._sps_window if now - t < 60] + [(now, self.num_timesteps)]
        (t0, n0), (t1, n1) = self._sps_window[0], self._sps_window[-1]
        self._write("status.json", dict(
            updated=round(now, 1), timesteps=int(self.num_timesteps), episodes=self._episode,
            elapsed_s=round(now - self._t_start, 1),
            phase=self._phase, iteration=self._iteration,
            rollout_pos=int(self.model.rollout_buffer.pos), n_steps=int(self.model.n_steps),
            n_epochs=int(self.model.n_epochs),
            steps_per_s=round((n1 - n0) / (t1 - t0), 1) if t1 > t0 else None,
            current=None if done else dict(steps=self._steps, laps=round(float(info["laps"]), 3),
                                           track_seed=int(info["track_seed"]),
                                           speed=round(float(info["speed"]), 1),
                                           reward=round(self._reward_so_far, 1),
                                           offtrack_pct=round(100.0 * self._offtrack / max(self._steps, 1), 1)),
            live_fields=["timesteps", "speed", "slip_deg", "steer", "throttle", "lateral_m", "x", "y", "track_seed"],
            live=list(self._live),
        ))
