"""Procedural closed-loop track generation.

Pure Python/numpy/scipy — no BeamNG dependency, so this can be developed and
tested standalone. Produces a smooth closed centerline (scatter random points
in an annulus, order by angle, fit a periodic spline) plus a constant or
varying track width, which `src/envs/drift_env.py` later feeds into BeamNG's
road API to actually place the track in GridMap.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep


def generate_track(
    num_control_points: int = 12,
    radius: float = 60.0,
    irregularity: float = 0.35,
    width: float = 10.0,
    num_samples: int = 400,
    seed: int | None = None,
) -> dict:
    """Generate a random closed-loop track centerline.

    Returns a dict with:
      centerline: (num_samples, 2) array of (x, y) points, closed loop.
      width: constant track width (float) for now.
      curvature: (num_samples,) signed curvature at each centerline sample,
        used both for the car-picker's track features later and for
        feeding "curvature ahead" into the driving policy's observation.
    """
    rng = np.random.default_rng(seed)

    angles = np.sort(rng.uniform(0, 2 * np.pi, num_control_points))
    radii = radius * (1 + rng.uniform(-irregularity, irregularity, num_control_points))
    control_x = radii * np.cos(angles)
    control_y = radii * np.sin(angles)

    # Periodic B-spline through the control points, closed loop.
    tck, _ = splprep([control_x, control_y], per=True, s=0)
    u = np.linspace(0, 1, num_samples, endpoint=False)
    x, y = splev(u, tck)
    centerline = np.stack([x, y], axis=1)

    curvature = _signed_curvature(centerline)

    return {"centerline": centerline, "width": width, "curvature": curvature}


def _signed_curvature(points: np.ndarray) -> np.ndarray:
    """Signed curvature at each point of a closed polyline (finite differences)."""
    dx = np.gradient(points[:, 0], edge_order=1)
    dy = np.gradient(points[:, 1], edge_order=1)
    ddx = np.gradient(dx, edge_order=1)
    ddy = np.gradient(dy, edge_order=1)
    denom = (dx**2 + dy**2) ** 1.5
    denom[denom == 0] = 1e-9
    return (dx * ddy - dy * ddx) / denom


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
    corner_count = np.sum(np.diff(is_corner.astype(int)) == 1)
    straight_frac = 1.0 - is_corner.mean()

    return np.array(
        [length, np.abs(curvature).mean(), np.abs(curvature).max(), corner_count, straight_frac]
    )


if __name__ == "__main__":
    track = generate_track(seed=0)
    print("centerline shape:", track["centerline"].shape)
    print("track features:", track_feature_vector(track))
