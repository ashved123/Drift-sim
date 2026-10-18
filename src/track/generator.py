"""Procedural closed-loop track generation.

Pure Python/numpy/scipy — no BeamNG dependency, so this can be developed and
tested standalone. Produces a smooth closed centerline (scatter random points
in an annulus, order by angle, fit a periodic spline) plus a constant or
varying track width, which `src/envs/drift_env.py` later feeds into BeamNG's
road API to actually place the track in GridMap.

Raw random splines often contain hairpin kinks tighter than the road is wide
(the inner road edge folds over itself) or loop back across themselves, so
candidates are validated and regenerated until one is drivable.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep


def generate_track(
    num_control_points: int = 14,
    radius: float = 120.0,
    irregularity: float = 0.35,
    angle_jitter: float = 0.4,
    smoothing: float = 0.05,
    width: float = 15.0,
    num_samples: int = 800,
    min_corner_radius: float | None = None,
    min_clearance: float | None = None,
    max_attempts: int = 200,
    seed: int | None = None,
) -> dict:
    """Generate a random, drivable closed-loop track centerline.

    angle_jitter: how far each control point's angle may stray from an evenly
      spaced slot, as a fraction of the slot size (< 0.5 keeps points ordered
      and prevents two points bunching up into a hairpin).
    smoothing: allowed RMS deviation of the spline from the control points,
      as a fraction of `radius` (0 = pass exactly through every point).
    min_corner_radius: tightest allowed corner radius; defaults to `width`.
    min_clearance: minimum centerline distance between parts of the track that
      are far apart along it, so road surfaces never overlap; defaults to
      1.5 * width.

    Returns a dict with:
      centerline: (num_samples, 2) array of (x, y) points, closed loop,
        evenly spaced along the arc length.
      width: constant track width (float) for now.
      curvature: (num_samples,) signed curvature at each centerline sample,
        used both for the car-picker's track features later and for
        feeding "curvature ahead" into the driving policy's observation.
    """
    if min_corner_radius is None:
        min_corner_radius = width
    if min_clearance is None:
        min_clearance = 1.5 * width

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        centerline, curvature = _sample_centerline(
            rng, num_control_points, radius, irregularity, angle_jitter, smoothing, num_samples
        )
        if np.abs(curvature).max() > 1.0 / min_corner_radius:
            continue
        if _min_clearance(centerline, ignore_arc=3 * width) < min_clearance:
            continue
        return {"centerline": centerline, "width": width, "curvature": curvature}

    raise RuntimeError(
        f"No drivable track found in {max_attempts} attempts; loosen irregularity/angle_jitter "
        "or min_corner_radius/min_clearance, or increase radius."
    )


def _sample_centerline(rng, num_control_points, radius, irregularity, angle_jitter, smoothing, num_samples):
    slot = 2 * np.pi / num_control_points
    angles = np.arange(num_control_points) * slot + rng.uniform(-angle_jitter, angle_jitter, num_control_points) * slot
    radii = radius * (1 + rng.uniform(-irregularity, irregularity, num_control_points))
    control_x = radii * np.cos(angles)
    control_y = radii * np.sin(angles)

    # Periodic B-spline through (near) the control points, closed loop.
    s = num_control_points * (smoothing * radius) ** 2
    tck, _ = splprep([control_x, control_y], per=True, s=s)

    # Resample evenly by arc length so "N samples ahead" is a fixed distance.
    u_dense = np.linspace(0, 1, num_samples * 20, endpoint=False)
    dense = np.stack(splev(u_dense, tck), axis=1)
    seg = np.linalg.norm(np.diff(dense, axis=0, append=dense[:1]), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    u_even = np.interp(np.linspace(0, cum[-1], num_samples, endpoint=False), cum, np.append(u_dense, 1.0))

    x, y = splev(u_even, tck)
    dx, dy = splev(u_even, tck, der=1)
    ddx, ddy = splev(u_even, tck, der=2)
    # Analytic curvature from the spline's derivatives: exact and periodic.
    curvature = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-12)
    return np.stack([x, y], axis=1), curvature


def _min_clearance(centerline: np.ndarray, ignore_arc: float) -> float:
    """Smallest distance between centerline points more than `ignore_arc` apart along the loop.

    Catches both self-intersections (distance ~0) and near-misses where two
    stretches of road would overlap.
    """
    n = len(centerline)
    spacing = np.linalg.norm(np.diff(centerline, axis=0, append=centerline[:1]), axis=1).mean()
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    sep = np.minimum(sep, n - sep) * spacing
    diffs = centerline[:, None, :] - centerline[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs))
    far = sep > ignore_arc
    return float(dist[far].min()) if far.any() else np.inf


def track_feature_vector(track: dict) -> np.ndarray:
    """Summary features for the Stage 6 car-picker/tuner model.

    [length, mean_abs_curvature, max_abs_curvature, corner_count, straight_frac]
    """
    centerline = track["centerline"]
    curvature = track["curvature"]

    segment_lengths = np.linalg.norm(np.diff(centerline, axis=0, append=centerline[:1]), axis=1)
    length = segment_lengths.sum()

    corner_thresh = 0.02  # curvature magnitude above which a point counts as "in a corner"
    is_corner = np.abs(curvature) > corner_thresh
    corner_count = np.sum(is_corner & ~np.roll(is_corner, 1))  # corner entries, counted around the loop
    straight_frac = 1.0 - is_corner.mean()

    return np.array(
        [length, np.abs(curvature).mean(), np.abs(curvature).max(), corner_count, straight_frac]
    )


if __name__ == "__main__":
    track = generate_track(seed=0)
    print("centerline shape:", track["centerline"].shape)
    print("track features:", track_feature_vector(track))
