# drift-rl

ML course project: train a neural network to drift a car around a procedurally
generated race track with high accuracy, using real soft-body vehicle physics.

## Framing

This is a reinforcement learning / control project, not classic supervised ML.
A policy network outputs continuous steering/throttle/brake, trained against a
reward signal that encourages sustained, controlled drift while following a
procedurally generated track.

## Simulator

- **BeamNG.drive** (Steam App ID 284160) is the physics simulator, controlled
  via the official **beamngpy** Python API (TCP link to a running instance).
- Running the **Windows build via Proton** (`scripts/launch_beamng_proton.sh`,
  also a desktop icon). The native Linux build exists but its Vulkan renderer
  segfaults in Mesa's Intel driver (`libvulkan_intel.so`) on this Arrow Lake
  laptop, even on Mesa 26.2.5 (kisak PPA). The launcher passes beamngpy's
  usual flags (`-nosteam -tcom -tport 25252`); scripts then attach with
  `bng.open(launch=False)` (i.e. run without `BNG_HOME`).
- (If using GridMap v2 instead: its ground is at **z ~ 100**, not 0. Spawning at
  z=0.5 drops the car through the map. `spawn_middle` is (-122, -188, 100.75).
  `scripts/test_beamng_connection.py` still uses GridMap v2.)
- Levels (`src/envs/track_placement.py`), two modes:
  - **island** (default): BeamNG's `template` level, a 1 km grass terrain with
    ocean (z~115.7) around it. `src/track/terrain.py` generates an island
    heightmap with rolling hills and the track cut in (smoothed road profile,
    max 8% grade, gravel shoulders). It's written as a bundle (16-bit height
    PNG, 16-bit material PNG, roads JSON) to `<BeamNG user dir>/temp/drift_rl/`
    and applied by one Lua call: terrain heights + materials, the level's
    rocks deleted, track recreated as DecalRoads. ~1 s per switch, no level
    reload. Gotchas: `TerrainBlock:setHeight` takes height *above the block
    base* in [0, maxHeight) and wraps past it; the game reads 8-bit PNG
    texels as 0 (use 16-bit); ground cover grows by terrain material, so the
    ground under the asphalt is BeachSand (the only material with none).
    Large DecalRoads (long + 15 m wide) silently fail to render in whole or
    in part, so roads are sent as ~60 m pieces with a node every ~4 m, and
    DecalRoad's `looped` flag is not used (it leaves a gap at the seam).
  - **flat**: `smallgrid` ("Grid, Small, Pure"), endless flat plane at z=0.
    Its GroundPlane can't take DecalRoads, so the track is baked into the
    scenario as MeshRoads (level reload per track).
- Car: base-game drift car, Ibishu 200BX `bx` with `vehicles/bx/pro_drift_M.pc`.
- Control panel: `python scripts/track_control_panel.py` (tkinter; Linux
  needs `python3-tk`): seed prev/next/random, generation sliders, preview,
  reset car, overhead/car camera. `src/envs/track_loader.py` prefetches the
  next 5 seeds on worker threads; only one thread talks to BeamNG.
  Terminal-only alternative: `scripts/view_track_in_beamng.py`.

## Track generation

Procedural closed-loop track, generated in Python:
1. Scatter control points in an annulus: one per evenly spaced angle slot,
   jittered within its slot (fully random angles let two points bunch up and
   produce hairpin kinks), with a random radius.
2. Fit a smoothed periodic B-spline near them, resample evenly by arc length,
   and take curvature analytically from the spline derivatives.
3. Reject and regenerate (deterministically from the seed) any track whose
   tightest corner radius is below the road width or whose road would overlap
   itself. `scripts/check_track_generator.py` checks this over many seeds.
4. Place it in the game (see Simulator above): surface, white edge lines,
   striped curbs on tight corners, chequered start/finish line.

This keeps "procedurally generated" literal (new random track per run/episode)
while using BeamNG's real tire/surface physics instead of a toy model.

## Learning approach

- **Algorithm:** Deep RL, PPO or SAC. Full action space is **hybrid**
  (continuous steering/throttle/brake/handbrake/clutch + discrete gear), which
  Stable-Baselines3 does not support out of the box (its policies take Box,
  Discrete, MultiDiscrete or MultiBinary, not a mixed Dict action space) — so
  the policy network needs custom action heads: one continuous (Gaussian/Beta)
  head per continuous control, one categorical head for gear, trained jointly.
  Built this way from the start so curriculum stages (below) only change what
  the *environment* lets the policy actually use — the network architecture
  never changes between stages, so each stage fine-tunes from the previous
  checkpoint rather than retraining from scratch.
- **Env wrapper:** Custom Gymnasium env (`src/envs/`) wraps BeamNGpy: `reset()`
  regenerates a track + respawns the car at the start, `step()` sends the full
  control vector and returns the next observation/reward/done. A per-stage
  mask overrides unlocked-but-not-yet-active dims to safe defaults (e.g. brake
  forced 0, clutch forced "fully engaged", gear left on auto) until that stage
  unlocks them.
- **Observation space:** vehicle state relative to the track centerline —
  lateral offset, heading error, upcoming curvature, speed, slip angle
  (velocity-vector vs heading mismatch), wheel speeds, current gear + RPM
  once gear control is unlocked, cumulative damage — rather than raw pixels.
- **Reward shaping:** see "Reward function" below.

### Reward function

All stages:
- **Progress:** `+ (arc_length_t - arc_length_{t-1})` — car position projected
  onto the track centerline's arc-length each step. Main "keep moving around
  the track" signal (same idea as CarRacing's tile rewards).
- **Off-track penalty, with recovery grace:** while
  `lateral_offset > track_half_width`, apply an escalating penalty
  `- k_offtrack * lateral_offset` and increment an `off_track_steps` counter;
  only **terminate** once `off_track_steps > recovery_timeout` (placeholder:
  150 steps). Chosen deliberately over instant termination so the agent gets
  to practice catching a slide and recovering, at the cost of longer episodes
  (relevant since BeamNG runs near real-time, not a fast parallel sim).
- **Damage penalty:** `- k_damage * delta_damage` each step; **terminate** if
  cumulative damage crosses a "destroyed" threshold (placeholder: 0.9). This
  is also how bad-shift (moneyshift) consequences are punished — no separate
  hand-coded penalty needed, see Gear control below.
- **Small time penalty:** `- time_penalty` every step, just to prevent a
  "stand still somewhere safe" local optimum.

Stage 3+ (handbrake unlocked) adds a **drift bonus**, style-focused
(maximize sustained slip angle, chosen over a target-angle/line-following
formulation for simplicity):

```
drift_bonus = progress_reward * f(|slip_angle|)
f(theta) = theta / theta_max                                   for theta <= theta_max
         = max(0, 1 - (theta - theta_max) / theta_falloff)      for theta >  theta_max
```

`theta_max` (placeholder ~55°) is the top of the "more angle = more reward"
range; past it the car is approaching full spin-out (losing directional
control entirely), so the bonus falls back off rather than continuing to
reward an uncontrolled spin. **Multiplying by `progress_reward` is the key
anti-reward-hacking move** — without it the agent's dominant strategy is
spinning donuts in one spot (maximizes slip angle, zero actual progress).
Gating the bonus on progress means slip angle only pays off while the car is
actually advancing around the track.

All coefficients above (`k_offtrack`, `k_damage`, `time_penalty`, `theta_max`,
`theta_falloff`, `recovery_timeout`) are placeholders to be tuned empirically
once training runs are possible — noted in code as such.

### Gear control (Stage 4)

Action output is a single continuous scalar `shift_delta` (same Gaussian head
as every other continuous control — Approach 1, no custom policy code).
At the env boundary it's rounded/clipped to `{-1, 0, +1}` = downshift / hold /
upshift.

Gear itself is **persistent state in the environment**, not re-decided from
scratch each tick: `current_gear = clip(current_gear + shift_delta, -1, max_gear)`.
Reverse is simply the rung below 1st (`current_gear == -1`) — no separate
reverse flag or action needed. `current_gear` and RPM feed back into the
observation so the policy knows what gear it's in.

No hand-coded guardrail against bad shifts (e.g. moneyshifting into too low a
gear at high RPM). BeamNG's own damage model simulates the real consequence
(blown engine/gearbox) when that happens — we read it off BeamNG's damage
sensor and let the existing damage penalty in the reward handle it. The agent
learns to avoid bad shifts the same way a real driver does, by actually
breaking the car.

### Curriculum (progressive action-space unlock)

Same policy/network throughout; each stage fine-tunes from the prior stage's
checkpoint rather than training from scratch.

1. **Steering + throttle** — basic track following, no drifting expected yet.
   Reward: progress + stay on track.
2. **+ Brake** — learn to slow for corners safely. Still not drifting.
3. **+ Handbrake** — primary drift-initiation tool unlocked. This is where
   real drifting behavior first emerges; reward adds the slip-angle bonus.
   **This is the fallback deliverable if later stages run out of time.**
4. **+ Manual gear shift** (discrete head) — agent must manage RPM/torque
   delivery itself instead of relying on BeamNG's automatic transmission.
5. **+ Clutch** (continuous head) — clutch-kick initiation/correction as an
   additional technique layered on top of handbrake initiation.
6. **Car-picker + setup tuner** — a *separate* model, not part of the driving
   policy: given track features (curvature profile, corner count, straight %,
   length), classify the best vehicle from a fixed roster (2-6 BeamNG cars
   with different weight/power/drivetrain) AND regress continuous setup
   params (diff lock strength, brake bias, ...) for that car+track pairing.
   Training data for this comes from rolling out the Stage 5 driving policy
   across many (car, setup, track) combinations and recording performance.

## Project layout

```
drift-rl/
  venv/                  Python virtual environment (not committed)
  src/
    track/               Procedural spline track generator (pure Python, no BeamNG needed)
    envs/                 Gymnasium env wrapping beamngpy (BeamNG required/running)
    training/             PPO/SAC training scripts (Stable-Baselines3)
  scripts/               One-off utility scripts (e.g. BeamNG connection test)
  notebooks/             Exploration / result analysis
```

## Environment notes

- Python deps live in `venv/` (beamngpy, gymnasium, stable-baselines3, numpy,
  scipy, matplotlib, shapely). Activate with `source venv/bin/activate`
  (Linux) or `venv\Scripts\activate` (Windows). The project is developed on
  both; keep paths in code OS-neutral (`pathlib`, no hardcoded separators).
- BeamNG.drive must be running (with its Lua console / API enabled) before any
  script under `src/envs/` or `scripts/test_beamng_connection.py` will connect.
- Track generation code under `src/track/` has no BeamNG dependency and can be
  developed/tested standalone at any time.

## Timeline (4 weeks)

- **Week 1:** BeamNG connection, procedural track generator, env scaffold,
  Stage 1 (steering+throttle) training to validate the pipeline end-to-end.
- **Week 2:** Stage 2 (brake) + Stage 3 (handbrake) — core drifting behavior.
  Fallback submission if time runs short.
- **Week 3:** Stage 4 (manual gear) + Stage 5 (clutch) — full manual control.
- **Week 4:** Stage 6 (car-picker + setup tuner), polish, report/demo.

## Status

- [x] BeamNG.drive purchased, installing via Steam.
- [x] Project scaffold + venv created.
- [x] Python deps installed in `venv/` (beamngpy 1.35.1 — pinned, 1.36 speaks a newer protocol than BeamNG 0.39.4 — gymnasium 1.4.0,
      stable-baselines3 2.9.0, torch 2.14.1, numpy, scipy, matplotlib, shapely).
      Track generator + env smoke-tested successfully end-to-end.
- [x] BeamNG connection smoke test passes (`scripts/test_beamng_connection.py`
      against BeamNG launched via Proton).
- [x] Procedural track generator (validated: no corner tighter than road
      width, no self-overlap, evenly spaced samples), ~650-950 m long, 15 m wide.
      Placed and drivable in BeamNG (smallgrid) via `scripts/view_track_in_beamng.py`.
- [ ] Gymnasium env wrapper around BeamNGpy (full hybrid action space, with
      per-stage masking).
- [ ] Custom hybrid-action policy (continuous heads + discrete gear head).
- [ ] Stage 1 training run.
