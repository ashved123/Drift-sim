"""Reads a training run's dashboard files (runs/dash/<run>/, written by src/training/dashboard_log.py).

Incremental: .jsonl files are tailed, so polling every half second stays cheap
even for long runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DASH_DIR = ROOT / "runs" / "dash"
LIVE_AFTER_S = 30  # status older than this = the run isn't training right now


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class JsonlTail:
    """Reads only the new complete lines of a growing .jsonl file each poll."""

    def __init__(self, path: Path):
        self.path, self.offset, self.rows = path, 0, []

    def poll(self) -> bool:
        try:
            with open(self.path, encoding="utf-8") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return False
        complete = chunk[: chunk.rfind("\n") + 1]  # leave a half-written last line for next time
        self.offset += len(complete.encode("utf-8"))
        new = []
        for line in complete.splitlines():
            try:
                new.append(json.loads(line))
            except ValueError:
                pass
        self.rows += new
        return bool(new)


def list_runs() -> list[str]:
    """Run folder names, most recently active first."""
    if not DASH_DIR.exists():
        return []
    runs = [d for d in DASH_DIR.iterdir() if d.is_dir()]
    return [d.name for d in sorted(runs, key=lambda d: max((f.stat().st_mtime for f in d.iterdir()),
                                                         default=d.stat().st_mtime), reverse=True)]


class RunData:
    """Everything known about one run; `poll()` refreshes it and reports what changed."""

    def __init__(self, name: str):
        self.name = name
        self.dir = DASH_DIR / name
        self.episodes = JsonlTail(self.dir / "episodes.jsonl")
        self.updates = JsonlTail(self.dir / "updates.jsonl")
        self.ppo_hist = JsonlTail(self.dir / "ppo_hist.jsonl")
        self.meta: dict | None = None
        self.status: dict | None = None
        self.track: dict | None = None
        self.ppo_viz: dict | None = None
        self._viz_mtime = 0.0

    def poll(self) -> dict:
        self.meta = self.meta or read_json(self.dir / "meta.json")
        self.status = read_json(self.dir / "status.json") or self.status
        self.track = read_json(self.dir / "current_track.json") or self.track
        changed = dict(episodes=self.episodes.poll(), updates=self.updates.poll(), ppo=self.ppo_hist.poll())
        try:
            mtime = (self.dir / "ppo_viz.json").stat().st_mtime
        except OSError:
            mtime = 0.0
        if mtime != self._viz_mtime:
            self.ppo_viz = read_json(self.dir / "ppo_viz.json") or self.ppo_viz
            self._viz_mtime = mtime
            changed["ppo"] = True
        return changed

    def best_episode(self):
        return read_json(self.dir / "best_episode.json")

    def is_live(self) -> bool:
        return bool(self.status and time.time() - self.status.get("updated", 0) < LIVE_AFTER_S
                    and self.status.get("phase") != "finished")
