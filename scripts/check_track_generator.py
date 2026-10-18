"""Sanity check for the procedural track generator (no BeamNG needed).

Generates many seeded tracks and verifies each one is drivable: closed,
deterministic, no corner tighter than the road width, no overlapping road.
Optionally saves a grid plot of sample tracks coloured by curvature.

    python scripts/check_track_generator.py
    python scripts/check_track_generator.py --seeds 500 --plot runs/tracks.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.track.generator import _min_clearance, generate_track, track_feature_vector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=200, help="number of seeded tracks to check")
    parser.add_argument("--plot", type=Path, help="save a grid of sample tracks to this image path")
    args = parser.parse_args()

    failures = []
    lengths, min_radii, clearances = [], [], []
    for seed in range(args.seeds):
        track = generate_track(seed=seed)
        c, k, w = track["centerline"], track["curvature"], track["width"]

        if not (np.isfinite(c).all() and np.isfinite(k).all()):
            failures.append((seed, "non-finite values"))
            continue
        r_min = 1.0 / np.abs(k).max()
        clearance = _min_clearance(c, ignore_arc=3 * w)
        if r_min < w:
            failures.append((seed, f"corner radius {r_min:.1f} m < width {w} m"))
        if clearance < 1.5 * w:
            failures.append((seed, f"road overlaps itself (clearance {clearance:.1f} m)"))

        lengths.append(track_feature_vector(track)[0])
        min_radii.append(r_min)
        clearances.append(clearance)

    a, b = generate_track(seed=42), generate_track(seed=42)
    if not np.array_equal(a["centerline"], b["centerline"]):
        failures.append((42, "same seed gave different tracks"))

    # Samples should be evenly spaced by arc length, including across the start/finish wrap.
    spacing = np.linalg.norm(np.diff(a["centerline"], axis=0, append=a["centerline"][:1]), axis=1)
    if spacing.max() / spacing.min() > 1.05:
        failures.append((42, f"uneven sample spacing ({spacing.min():.2f}-{spacing.max():.2f} m)"))

    print(f"checked {args.seeds} tracks")
    print(f"  length (m):            min {min(lengths):.0f}  median {np.median(lengths):.0f}  max {max(lengths):.0f}")
    print(f"  tightest corner (m):   min {min(min_radii):.1f}  median {np.median(min_radii):.1f}")
    print(f"  road clearance (m):    min {min(clearances):.1f}  median {np.median(clearances):.1f}")
    print(f"  sample spacing (m):    {spacing.min():.3f}-{spacing.max():.3f}")

    if args.plot:
        _plot(args.plot)

    if failures:
        print(f"FAILED ({len(failures)}):")
        for seed, msg in failures[:20]:
            print(f"  seed {seed}: {msg}")
        return 1
    print("OK")
    return 0


def _plot(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: works without a display on either OS
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for seed, ax in enumerate(axes.flat):
        track = generate_track(seed=seed)
        c = track["centerline"]
        ax.plot(*np.vstack([c, c[:1]]).T, color="0.8", linewidth=track["width"] * 0.9, solid_capstyle="round")
        ax.scatter(c[:, 0], c[:, 1], c=np.abs(track["curvature"]), s=3, cmap="plasma", vmin=0, vmax=1 / track["width"], zorder=3)
        ax.plot(*c[0], "go")
        ax.set_title(f"seed {seed}  tightest corner {1 / np.abs(track['curvature']).max():.0f} m")
        ax.set_aspect("equal")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=70)
    print(f"saved plot to {path}")


if __name__ == "__main__":
    sys.exit(main())
