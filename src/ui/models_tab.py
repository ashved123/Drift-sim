"""Models tab: every saved checkpoint, and what to do with it.

- Continue training from it (same stage), or train the next curriculum stage
  from it — fills in the Training tab.
- Watch it drive in BeamNG on tracks it hasn't trained on (scripts/watch_policy.py
  in a separate process; holds the game while it runs).
- Delete it (with its VecNormalize stats).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import messagebox, ttk

from src.training.train import stats_path
from src.ui import theme as T
from src.ui.run_data import ROOT

MODELS_DIR = ROOT / "models"
REPLAY_DIR = ROOT / "runs" / "replay"  # replay log + stop file (keeps models/ checkpoints-only)
OWNER = "a model replay"
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_steps_cache: dict[tuple[str, float], int | None] = {}


def checkpoint_steps(path: Path) -> int | None:
    """Training steps stored inside an SB3 .zip (its 'data' JSON has num_timesteps)."""
    key = (str(path), path.stat().st_mtime)
    if key not in _steps_cache:
        try:
            with zipfile.ZipFile(path) as z:
                _steps_cache[key] = int(json.loads(z.read("data"))["num_timesteps"])
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            _steps_cache[key] = None
    return _steps_cache[key]


def list_checkpoints() -> list[dict]:
    """All checkpoints under models/stageN/, newest first."""
    out = []
    for stage_dir in sorted(MODELS_DIR.glob("stage*")):
        try:
            stage = int(stage_dir.name.removeprefix("stage"))
        except ValueError:
            continue
        for zp in stage_dir.glob("*.zip"):
            st = zp.stat()
            out.append(dict(path=zp, name=zp.stem, stage=stage, steps=checkpoint_steps(zp), mtime=st.st_mtime,
                            size_mb=st.st_size / 1e6, has_stats=stats_path(zp).exists()))
    return sorted(out, key=lambda c: c["mtime"], reverse=True)


class ModelsTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.log_offset = 0
        self.stop_file = REPLAY_DIR / "STOP"

        top = ttk.Frame(self)
        top.pack(fill="both", expand=True)
        cols = [("stage", "Stage", 60), ("name", "Checkpoint", 220), ("steps", "Steps trained", 110),
                ("saved", "Saved", 150), ("size", "Size", 70), ("path", "File", 360)]
        self.tree = ttk.Treeview(top, columns=[c for c, _, _ in cols], show="headings", height=12,
                                 selectmode="browse")
        for c, label, w in cols:
            self.tree.heading(c, text=label)
            self.tree.column(c, width=w, anchor="e" if c in ("steps", "size", "stage") else "w")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.refresh_buttons())

        actions = ttk.LabelFrame(top, text="Selected model", padding=8)
        actions.pack(side="left", fill="y", padx=(10, 0))
        self.btn_continue = ttk.Button(actions, text="Continue training (same stage)", command=self.continue_training)
        self.btn_next = ttk.Button(actions, text="Train next stage from it", command=self.next_stage)
        for b in (self.btn_continue, self.btn_next):
            b.pack(fill="x", pady=2)
        ttk.Separator(actions).pack(fill="x", pady=8)
        ttk.Label(actions, text="Watch it drive in the game").pack(anchor="w")
        wf = ttk.Frame(actions)
        wf.pack(fill="x")
        ttk.Label(wf, text="episodes").grid(row=0, column=0, sticky="w")
        self.episodes = tk.StringVar(value="3")
        ttk.Spinbox(wf, from_=1, to=50, width=6, textvariable=self.episodes).grid(row=0, column=1, padx=4)
        ttk.Label(wf, text="track seed").grid(row=1, column=0, sticky="w")
        self.watch_seed = tk.StringVar(value="100000")
        ttk.Entry(wf, width=8, textvariable=self.watch_seed).grid(row=1, column=1, padx=4)
        ttk.Label(wf, text="hills").grid(row=2, column=0, sticky="w")
        self.watch_hills = tk.StringVar(value="0")
        ttk.Entry(wf, width=8, textvariable=self.watch_hills).grid(row=2, column=1, padx=4)
        self.btn_watch = ttk.Button(actions, text="▶ Watch", command=self.watch)
        self.btn_watch.pack(fill="x", pady=(6, 2))
        self.btn_stop_watch = ttk.Button(actions, text="■ Stop watching", command=self.stop_watch)
        self.btn_stop_watch.pack(fill="x", pady=2)
        ttk.Separator(actions).pack(fill="x", pady=8)
        self.btn_delete = ttk.Button(actions, text="Delete…", command=self.delete)
        self.btn_delete.pack(fill="x", pady=2)
        ttk.Button(actions, text="Open models folder", command=self.open_folder).pack(fill="x", pady=2)
        ttk.Button(actions, text="Refresh list", command=self.refresh).pack(fill="x", pady=2)

        out = ttk.LabelFrame(self, text="Replay output", padding=6)
        out.pack(fill="both", expand=True, pady=(10, 0))
        self.state_var = tk.StringVar(value="Pick a model above. 'final' = a run that finished; 'latest' = where the "
                                            "last run stopped; 'ppo_N_steps' = periodic checkpoints.")
        ttk.Label(out, textvariable=self.state_var, foreground=T.INK2, wraplength=1100, justify="left").pack(anchor="w")
        self.log = tk.Text(out, height=10, wrap="none", font=("Consolas", 9), background=T.SURFACE,
                           foreground=T.INK, relief="flat")
        self.log.pack(fill="both", expand=True, pady=(6, 0))
        self.log.configure(state="disabled")
        self.items: dict[str, dict] = {}
        self.refresh()

    # ------------------------------------------------------------ list

    def refresh(self) -> None:
        keep = self.selected()
        self.tree.delete(*self.tree.get_children())
        self.items = {}
        for c in list_checkpoints():
            iid = str(c["path"])
            self.items[iid] = c
            self.tree.insert("", "end", iid=iid, values=(
                c["stage"], c["name"] + ("" if c["has_stats"] else "  (no normalisation stats!)"),
                f"{c['steps']:,}" if c["steps"] is not None else "?",
                time.strftime("%Y-%m-%d %H:%M", time.localtime(c["mtime"])), f"{c['size_mb']:.1f} MB",
                str(c["path"].relative_to(ROOT))))
        if keep and str(keep["path"]) in self.items:
            self.tree.selection_set(str(keep["path"]))
        self.refresh_buttons()

    def selected(self) -> dict | None:
        sel = self.tree.selection() if hasattr(self, "tree") else ()
        return self.items.get(sel[0]) if sel else None

    def refresh_buttons(self) -> None:
        c = self.selected()
        watching = self.proc is not None and self.proc.poll() is None
        ok = c is not None and c["has_stats"]
        self.btn_continue.state(["!disabled" if ok else "disabled"])
        self.btn_next.state(["!disabled" if ok and c["stage"] < 5 else "disabled"])
        self.btn_watch.state(["!disabled" if ok and not watching else "disabled"])
        self.btn_stop_watch.state(["!disabled" if watching else "disabled"])
        self.btn_delete.state(["!disabled" if c is not None and not watching else "disabled"])

    # ------------------------------------------------------------ actions

    def continue_training(self) -> None:
        c = self.selected()
        self.app.training_tab.prefill(c["path"], c["stage"], seed=(c["steps"] or 0) // 100 + 1000)
        self.app.show_tab("training")

    def next_stage(self) -> None:
        c = self.selected()
        self.app.training_tab.prefill(c["path"], c["stage"] + 1, seed=(c["steps"] or 0) // 100 + 1000)
        self.app.show_tab("training")

    def watch(self) -> None:
        c = self.selected()
        if c is None:
            return
        if not self.app.game_lock.claim(OWNER):
            messagebox.showinfo("Drift Studio", f"The game is in use by {self.app.game_lock.owner}.")
            return
        try:
            episodes, seed, hills = int(self.episodes.get()), int(self.watch_seed.get()), float(self.watch_hills.get())
        except ValueError:
            self.app.game_lock.release(OWNER)
            messagebox.showerror("Drift Studio", "Episodes, track seed and hills must be numbers.")
            return
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = REPLAY_DIR / "watch.log"
        log_file = open(self.log_path, "w", encoding="utf-8")
        self.stop_file.unlink(missing_ok=True)  # a stale one would stop the replay immediately
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "watch_policy.py"), str(c["path"]),
               "--stage", str(c["stage"]), "--episodes", str(episodes), "--seed", str(seed), "--hills", str(hills),
               "--stop-file", str(self.stop_file)]
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                                     creationflags=CREATE_NO_WINDOW)
        log_file.close()
        self.log_offset = 0
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.state_var.set(f"Watching {c['name']} (stage {c['stage']}) drive {episodes} episode(s) from track seed "
                           f"{seed} — switch to the game window.")
        self.refresh_buttons()

    def stop_watch(self) -> None:
        """Ask the replay to stop cleanly (it restores the game's settings); kill it if it doesn't."""
        if self.proc is None or self.proc.poll() is not None:
            return
        self.stop_file.touch()
        self.state_var.set("Stopping the replay…")
        proc = self.proc
        self.after(20_000, lambda: proc.poll() is None and proc.kill())

    def delete(self) -> None:
        c = self.selected()
        if c is None or not messagebox.askyesno("Drift Studio", f"Delete {c['path'].relative_to(ROOT)} and its "
                                                                "normalisation stats? This can't be undone."):
            return
        c["path"].unlink(missing_ok=True)
        stats_path(c["path"]).unlink(missing_ok=True)
        self.refresh()
        self.app.training_tab.refresh_models()

    def open_folder(self) -> None:
        MODELS_DIR.mkdir(exist_ok=True)
        if sys.platform == "win32":
            os.startfile(MODELS_DIR)  # noqa: S606 - opening a folder in Explorer
        else:
            subprocess.Popen(["xdg-open", str(MODELS_DIR)])

    # ------------------------------------------------------------ supervision (called every tick)

    def tick(self) -> None:
        if self.log_path is not None:
            try:
                with open(self.log_path, encoding="utf-8", errors="replace") as f:
                    f.seek(self.log_offset)
                    new = f.read()
                    self.log_offset = f.tell()
                if new:
                    self.log.configure(state="normal")
                    self.log.insert("end", new)
                    self.log.see("end")
                    self.log.configure(state="disabled")
            except OSError:
                pass
        if self.proc is not None and self.proc.poll() is not None:
            code, self.proc = self.proc.returncode, None
            self.state_var.set("Replay finished." if code == 0 else f"Replay stopped (exit code {code}).")
            self.app.game_lock.release(OWNER)
            self.refresh_buttons()

    def shutdown(self) -> None:
        self.stop_watch()
