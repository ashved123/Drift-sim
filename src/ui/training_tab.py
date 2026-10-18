"""Training tab: start, monitor and stop training runs.

Each run is a separate process (`python -m src.training.train ...`), so the
Studio stays responsive and closing the window doesn't have to end a long
run. Stopping drops a stop file that training checks between steps: it saves
`latest.zip` and exits cleanly. Output goes to runs/dash/<run>/train.log,
shown live here.
"""

from __future__ import annotations

import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.ui import theme as T
from src.ui.run_data import DASH_DIR, ROOT, list_runs, read_json

OWNER = "training"
NEW_MODEL = "(start a new model)"
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class TrainingTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.proc: subprocess.Popen | None = None
        self.run_name: str | None = None
        self.stop_file: Path | None = None
        self.log_path: Path | None = None
        self.log_offset = 0
        self.stop_requested_at: float | None = None
        self.adopted = False  # a run started outside this window (e.g. before the Studio was reopened)

        form = ttk.LabelFrame(self, text="New training run", padding=10)
        form.pack(side="left", fill="y")
        self.vars = {}

        def row(r, label, widget, hint=""):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="ew", pady=3)
            if hint:
                ttk.Label(form, text=hint, foreground=T.MUTED).grid(row=r, column=2, sticky="w", padx=6)

        self.vars["stage"] = tk.StringVar(value="1")
        row(0, "Curriculum stage", ttk.Combobox(form, width=22, state="readonly", textvariable=self.vars["stage"],
                                                values=["1", "2", "3", "4", "5"]),
            "1 steer+throttle · 2 +brake · 3 +handbrake · 4 +gears · 5 +clutch")
        self.vars["resume"] = tk.StringVar(value=NEW_MODEL)
        self.resume_box = ttk.Combobox(form, width=40, state="readonly", textvariable=self.vars["resume"])
        row(1, "Start from", self.resume_box, "continue a saved model, or train a fresh one")
        self.vars["steps"] = tk.StringVar(value="1000000")
        row(2, "Training steps", ttk.Entry(form, textvariable=self.vars["steps"], width=24), "1M ≈ 15 h at ~18 steps/s")
        self.vars["hills"] = tk.DoubleVar(value=0.0)
        row(3, "Hills", tk.Scale(form, variable=self.vars["hills"], from_=0, to=2, resolution=0.1, orient="horizontal",
                                 length=170), "0 = flat island (recommended to start)")
        self.vars["new_track_every"] = tk.StringVar(value="1")
        row(4, "New track every", ttk.Spinbox(form, from_=1, to=50, textvariable=self.vars["new_track_every"], width=22),
            "episodes per generated track")
        self.vars["seed"] = tk.StringVar(value="0")
        row(5, "First track seed", ttk.Entry(form, textvariable=self.vars["seed"], width=24),
            "use a new range when continuing, e.g. 5000")
        self.vars["save_every"] = tk.StringVar(value="20000")
        row(6, "Checkpoint every", ttk.Entry(form, textvariable=self.vars["save_every"], width=24), "steps")

        btns = ttk.Frame(form)
        btns.grid(row=7, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self.start_btn = ttk.Button(btns, text="▶ Start training", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="■ Stop (save first)", command=self.stop)
        self.stop_btn.pack(side="left", padx=6)
        self.kill_btn = ttk.Button(btns, text="Force stop", command=self.force_stop)
        self.kill_btn.pack(side="left")
        self.note = tk.StringVar(value="")
        ttk.Label(form, textvariable=self.note, foreground=T.INK2, wraplength=520, justify="left").grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(10, 0))

        right = ttk.LabelFrame(self, text="Training output", padding=6)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.state_var = tk.StringVar(value="No training running.")
        ttk.Label(right, textvariable=self.state_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.log = tk.Text(right, height=30, wrap="none", font=("Consolas", 9), background=T.SURFACE,
                           foreground=T.INK, relief="flat")
        self.log.pack(fill="both", expand=True, pady=(6, 0))
        self.log.configure(state="disabled")

        self.refresh_models()
        self._adopt_running_run()
        self.refresh_buttons()

    # ------------------------------------------------------------ helpers

    def refresh_models(self) -> None:
        from src.ui.models_tab import list_checkpoints
        values = [NEW_MODEL] + [str(c["path"].relative_to(ROOT)) for c in list_checkpoints()]
        self.resume_box["values"] = values
        if self.vars["resume"].get() not in values:
            self.vars["resume"].set(NEW_MODEL)

    def prefill(self, resume: Path | None, stage: int, seed: int | None = None) -> None:
        """Called from the Models tab: set up a run continuing from a checkpoint."""
        self.refresh_models()
        self.vars["resume"].set(str(resume.relative_to(ROOT)) if resume else NEW_MODEL)
        self.vars["stage"].set(str(stage))
        if seed is not None:
            self.vars["seed"].set(str(seed))
        self.note.set("Settings filled in from the Models tab — check them and press Start.")

    def running(self) -> bool:
        return (self.proc is not None and self.proc.poll() is None) or self.adopted

    def refresh_buttons(self) -> None:
        run = self.running()
        self.start_btn.state(["disabled" if run else "!disabled"])
        self.stop_btn.state(["!disabled" if run and self.stop_file else "disabled"])
        self.kill_btn.state(["!disabled" if self.proc is not None and self.proc.poll() is None else "disabled"])

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _adopt_running_run(self) -> None:
        """If a run is still training from an earlier Studio session, show it and allow stopping it."""
        for name in list_runs()[:3]:
            d = DASH_DIR / name
            st, meta = read_json(d / "status.json"), read_json(d / "meta.json")
            if st and meta and time.time() - st.get("updated", 0) < 30 and st.get("phase") != "finished":
                self.adopted, self.run_name = True, name
                self.stop_file = Path(meta["stop_file"]) if meta.get("stop_file") else None
                self.log_path, self.log_offset = d / "train.log", 0
                if not self.app.game_lock.claim(OWNER):
                    self.app.game_lock.owner = OWNER
                self.state_var.set(f"Training run '{name}' is in progress (started earlier).")
                self.app.select_run(name)
                return

    # ------------------------------------------------------------ start / stop

    def start(self) -> None:
        try:
            stage, steps = int(self.vars["stage"].get()), int(float(self.vars["steps"].get()))
            seed, save_every = int(self.vars["seed"].get()), int(float(self.vars["save_every"].get()))
            new_every = int(self.vars["new_track_every"].get())
        except ValueError:
            messagebox.showerror("Drift Studio", "Steps, seed, checkpoint interval and 'new track every' must be numbers.")
            return
        if not self.app.game_lock.claim(OWNER):
            messagebox.showinfo("Drift Studio", f"The game is in use by {self.app.game_lock.owner}. "
                                                "Stop it first, then start training.")
            return
        name = f"stage{stage}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir = DASH_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file = run_dir / "STOP"
        self.stop_file.unlink(missing_ok=True)
        cmd = [sys.executable, "-u", "-m", "src.training.train", "--stage", str(stage),
               "--total-timesteps", str(steps), "--seed", str(seed), "--hills", str(self.vars["hills"].get()),
               "--new-track-every", str(new_every), "--save-every", str(save_every),
               "--run-name", name, "--stop-file", str(self.stop_file)]
        resume = self.vars["resume"].get()
        if resume != NEW_MODEL:
            cmd += ["--resume-from", str(ROOT / resume)]
        self.log_path, self.log_offset = run_dir / "train.log", 0
        log_file = open(self.log_path, "w", encoding="utf-8")
        log_file.write("$ " + " ".join(cmd[1:]) + "\n\n")
        log_file.flush()
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
                                     creationflags=CREATE_NO_WINDOW)
        log_file.close()  # the child keeps its own handle
        self.run_name, self.adopted, self.stop_requested_at = name, False, None
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.state_var.set(f"Training '{name}' — starting (BeamNG may take a minute to load)…")
        self.note.set("Watch it on the Live, Progress and PPO tabs. Stop saves the model first.")
        self.app.select_run(name)
        self.refresh_buttons()

    def stop(self) -> None:
        if self.stop_file is None:
            return
        self.stop_file.touch()
        self.stop_requested_at = time.time()
        self.state_var.set(f"Stopping '{self.run_name}' — finishing the current step and saving…")
        self.refresh_buttons()

    def force_stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        if messagebox.askyesno("Drift Studio", "Force-stop training now? Progress since the last checkpoint "
                                               "is lost (Stop saves first; use this only if it's stuck)."):
            self.proc.kill()

    # ------------------------------------------------------------ supervision (called every tick)

    def tick(self) -> None:
        if self.log_path is not None:
            try:
                with open(self.log_path, encoding="utf-8", errors="replace") as f:
                    f.seek(self.log_offset)
                    new = f.read()
                    self.log_offset = f.tell()
                if new:
                    self._append_log(new)
            except OSError:
                pass
        if self.proc is not None and self.proc.poll() is not None:
            code, self.proc = self.proc.returncode, None
            self._finished(f"Training '{self.run_name}' finished" + ("" if code == 0 else f" (exit code {code})") + ".")
        elif self.adopted and self.run_name:
            st = read_json(DASH_DIR / self.run_name / "status.json") or {}
            if st.get("phase") == "finished" or time.time() - st.get("updated", 0) > 60:
                self.adopted = False
                self._finished(f"Training '{self.run_name}' finished.")
        elif self.running() and self.stop_requested_at is None:
            st = read_json(DASH_DIR / self.run_name / "status.json") if self.run_name else None
            if st:
                self.state_var.set(f"Training '{self.run_name}' — step {st.get('timesteps', 0):,} · "
                                   f"{st.get('episodes', 0)} episodes · {st.get('phase', '')}")
        self.refresh_buttons()

    def _finished(self, message: str) -> None:
        self.state_var.set(message + " Models are on the Models tab.")
        self.app.game_lock.release(OWNER)
        self.app.models_tab.refresh()
        self.refresh_models()
