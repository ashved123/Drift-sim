"""Procedural island terrain around a generated track.

Pure numpy/scipy — no BeamNG dependency. Produces a heightmap and a terrain
material map for BeamNG's `template` level (a 1024 m x 1024 m terrain block at
1 m per cell with an ocean at z~115.7), shaped as a grassy island with rolling
hills. The track is cut into the hills: the road gets a smoothed elevation
profile with a capped gradient, the ground under it is flattened, and the
surrounding hills blend back in over a short distance.

Arrays are indexed [ix, iy], with world position
  (ORIGIN_X + ix * SQUARE_SIZE, ORIGIN_Y + iy * SQUARE_SIZE).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree

# Geometry of the template level's TerrainBlock.
GRID_SIZE = 1024
SQUARE_SIZE = 1.0
ORIGIN_X, ORIGIN_Y = -512.0, -512.0
BASE_Z = 100.0           # terrain block position z (lowest possible height)
MAX_HEIGHT = 120.0       # heights span BASE_Z .. BASE_Z + MAX_HEIGHT
SEA_Z = 115.73           # the level's ocean surface

# Indices into the template terrain's material list.
MAT_GRASS, MAT_SAND, MAT_DIRT, MAT_ROCK = 0, 1, 2, 3

LAND_Z = SEA_Z + 10.0     # average height of the island interior
ISLAND_RADIUS = 400.0     # coastline distance from the centre (before noise)
SHOULDER = 4.0            # flat gravel run-off each side of the road, metres
BLEND = 30.0              # distance over which hills blend back in beyond the shoulder
MAX_GRADE = 0.08          # steepest allowed road slope (8 %)


def generate_terrain(track: dict, seed: int | None = None, hilliness: float = 1.0) -> dict:
    """Island heightmap + material map with `track` (centred on world (0, 0)) cut into it.

    hilliness: scales hill height; 0 gives a flat island.

    Returns a dict with:
      height: (GRID_SIZE, GRID_SIZE) absolute terrain z.
      material: (GRID_SIZE, GRID_SIZE) uint8 terrain material index.
      road_z: (num_samples,) road surface z at each centerline sample.
    """
    rng = np.random.default_rng(seed)
    xs = ORIGIN_X + np.arange(GRID_SIZE) * SQUARE_SIZE
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    natural = _island(X, Y, rng) + hilliness * _hills(X, Y, rng)

    centerline = track["centerline"]
    width = track["width"]
    road_z = _road_profile(natural, centerline)

    # Distance from every cell to the centerline, and the road height there.
    tree = cKDTree(centerline)
    dist, nearest = tree.query(np.stack([X.ravel(), Y.ravel()], axis=1), workers=-1)
    dist = dist.reshape(X.shape)
    road_here = road_z[nearest].reshape(X.shape)

    flat = width / 2 + SHOULDER
    t = np.clip((dist - flat) / BLEND, 0.0, 1.0)
    t = t * t * (3 - 2 * t)  # smoothstep
    height = road_here * (1 - t) + natural * t
    # Soften creases where the blend zones of neighbouring track sections meet,
    # then re-impose the exact flat road bed.
    height = np.where(dist <= flat, road_here, gaussian_filter(height, 2.0))
    height = np.clip(height, BASE_Z, BASE_Z + MAX_HEIGHT)

    material = _materials(height, dist, width)
    return {"height": height, "material": material, "road_z": road_z}


def sample_height(height: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear terrain height at world (x, y) points."""
    ix = (points[:, 0] - ORIGIN_X) / SQUARE_SIZE
    iy = (points[:, 1] - ORIGIN_Y) / SQUARE_SIZE
    return map_coordinates(height, [ix, iy], order=1, mode="nearest")


def _island(X, Y, rng) -> np.ndarray:
    """Plateau at LAND_Z inside a wobbly coastline, sloping down below the sea outside it."""
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    coast = ISLAND_RADIUS
    for k in range(2, 6):  # a few low-frequency wobbles so the coast isn't a circle
        coast = coast + rng.uniform(-12, 12) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    t = np.clip((r - (coast - 60)) / 120, 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    return LAND_Z * (1 - t) + (SEA_Z - 8.0) * t


def _hills(X, Y, rng) -> np.ndarray:
    """Rolling hills: random Gaussian bumps and dips plus fine ground undulation."""
    hills = np.zeros_like(X)
    for _ in range(22):
        cx, cy = rng.uniform(-380, 380, 2)
        sigma = rng.uniform(35, 110)
        amp = rng.normal(5.0, 7.0)  # mostly hills, some hollows
        hills += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
    bumps = gaussian_filter(rng.standard_normal(X.shape), 10.0)
    hills += bumps / bumps.std() * 0.8
    # Fade hills out towards the coast so the shoreline stays a gentle beach.
    fade = np.clip((ISLAND_RADIUS - 40 - np.hypot(X, Y)) / 120, 0.0, 1.0)
    return hills * fade


def _road_profile(natural: np.ndarray, centerline: np.ndarray) -> np.ndarray:
    """Road height along the track: the terrain under it, smoothed until the gradient is drivable."""
    seg = np.linalg.norm(np.diff(centerline, axis=0, append=centerline[:1]), axis=1)
    spacing = seg.mean()
    z = sample_height(natural, centerline)
    sigma_m = 25.0
    while True:
        smooth = gaussian_filter1d(z, sigma_m / spacing, mode="wrap")
        grade = np.abs(np.diff(smooth, append=smooth[:1])) / seg
        if grade.max() <= MAX_GRADE or sigma_m > 400:
            break
        sigma_m *= 1.4
    return np.maximum(smooth, SEA_Z + 3.0)


def _materials(height: np.ndarray, dist: np.ndarray, width: float) -> np.ndarray:
    material = np.full(height.shape, MAT_GRASS, dtype=np.uint8)
    gy, gx = np.gradient(height, SQUARE_SIZE)
    material[np.hypot(gx, gy) > 0.7] = MAT_ROCK          # steeper than ~35 degrees
    material[height < SEA_Z + 1.5] = MAT_SAND             # beaches
    material[dist <= width / 2 + SHOULDER] = MAT_DIRT     # gravel run-off
    # Under the asphalt itself: sand, the one material with no ground cover, so
    # neither grass nor the gravel's pebbles poke through the road.
    material[dist <= width / 2 + 0.5] = MAT_SAND
    return material
