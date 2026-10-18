"""Place a generated track into BeamNG.

Shared by the viewer / control panel scripts and, later, `DriftEnv.reset()`.
Two settings:

- **flat**: the `smallgrid` level ("Grid, Small, Pure"), an endless flat plane at
  z=0 with nothing on it. Its ground is a GroundPlane, which DecalRoads cannot
  project onto, so the track is built from MeshRoads baked into the scenario.
- **island**: the `template` level, a 1 km grass terrain surrounded by ocean.
  `src/track/terrain.py` generates hills with the track cut into them; this
  module writes that as a "bundle" (heightmap PNG, material PNG, roads JSON)
  into the game's user folder, and one Lua call applies it to the loaded level:
  terrain heights and materials are replaced, the level's scattered rocks are
  removed, and the track is (re)created as DecalRoads draped on the terrain.
  Switching tracks only re-applies a bundle — the level is not reloaded.

Only materials from BeamNG's shared art library are used for the track, since
neither level ships suitable road materials of its own.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from beamngpy import BeamNGpy, MeshRoad, Scenario
from beamngpy.misc.quat import angle_to_quat

from src.track import terrain as T

FLAT_LEVEL = "smallgrid"
ISLAND_LEVEL = "template"
TRACK_CENTER = (0.0, 0.0)
GROUND_Z = 0.0  # smallgrid

# Base-game drift car: Ibishu 200BX, "Pro Drift" config.
DRIFT_CAR_MODEL = "bx"
DRIFT_CAR_CONFIG = "vehicles/bx/pro_drift_M.pc"

MESH_ROAD_MATERIAL = "m_asphalt_new_01"
DECAL_ROAD_MATERIAL = "road_asphalt_2lane"
EDGE_MATERIAL = "m_line_white"
CURB_MATERIAL = "line_yellowblack"  # striped texture, reads as a curb
START_MATERIAL = "checkered_line"
EDGE_LINE_WIDTH = 0.4
CURB_WIDTH = 1.5
CURB_MIN_CURVATURE = 1 / 45  # corners tighter than a 45 m radius get curbs

# MeshRoad (flat mode) heights of each layer's top surface above the ground. The
# road sits just proud of the ground, lines just proud of the road (avoids
# z-fighting), and curbs are a low raised strip the car can feel.
LAYER_TOP = {"surface": 0.02, "edge": 0.03, "start": 0.03, "curb": 0.08}
MESH_DEPTH = 0.3
# DecalRoad (island mode) draw order: higher draws on top.
LAYER_PRIORITY = {"surface": 10, "edge": 9, "curb": 8, "start": 8}

BUNDLE_DIR = "temp/drift_rl"  # relative to the BeamNG user folder
MATERIAL_PNG_STEP = 4096      # material index i is stored as i * 4096 in a 16-bit PNG
DECAL_NODE_STEP = 4           # use every 4th centerline sample (~4 m) as a DecalRoad node
LOOP_OVERLAP = 3              # nodes a looped decal road runs past its start
DECAL_CHUNK = 15              # max nodes (~60 m) per DecalRoad piece; big decals silently fail to render


def world_centerline(track: dict, center=TRACK_CENTER) -> np.ndarray:
    """Track centerline as (N, 2) world x, y."""
    return track["centerline"] + np.asarray(center)


def track_layers(track: dict, center=TRACK_CENTER) -> list[dict]:
    """The track's visual pieces: surface, edge lines, corner curbs and start/finish line.

    Each layer: name, kind, material, points (M, 2), idx (M,) centerline sample
    each point belongs to (for looking up road height), widths (M,), looped.
    """
    pts = world_centerline(track, center)
    road_w = track.get("widths", np.full(len(pts), track["width"]))
    normals = _normals(pts)
    all_idx = np.arange(len(pts))
    layers = [dict(name="drift_track", kind="surface", material=None, points=pts, idx=all_idx,
                   widths=road_w, looped=True)]

    edge_offset = (road_w / 2 - EDGE_LINE_WIDTH)[:, None]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        layers.append(dict(name=f"drift_track_edge_{side}", kind="edge", material=EDGE_MATERIAL,
                           points=pts + sign * edge_offset * normals, idx=all_idx,
                           widths=np.full(len(pts), EDGE_LINE_WIDTH), looped=True))

    # Curbs on the inside edge of each tight corner (positive curvature turns left).
    curb_offset = (road_w / 2 + CURB_WIDTH / 2)[:, None]
    for i, (start, end) in enumerate(_corner_runs(track["curvature"], CURB_MIN_CURVATURE)):
        idx = np.arange(start, end + 1) % len(pts)
        sign = np.sign(track["curvature"][idx].mean())
        layers.append(dict(name=f"drift_track_curb_{i}", kind="curb", material=CURB_MATERIAL,
                           points=pts[idx] + sign * curb_offset[idx] * normals[idx], idx=idx,
                           widths=np.full(len(idx), CURB_WIDTH), looped=False))

    half = road_w[0] / 2
    layers.append(dict(name="drift_track_start", kind="start", material=START_MATERIAL,
                       points=np.array([pts[0] - half * normals[0], pts[0] + half * normals[0]]),
                       idx=np.array([0, 0]), widths=np.array([2.0, 2.0]), looped=False))
    return layers


# ---------------------------------------------------------------- flat (smallgrid)

def add_track_to_scenario(scenario: Scenario, track: dict, center=TRACK_CENTER, ground_z: float = GROUND_Z) -> None:
    """Bake the track into a smallgrid scenario as MeshRoads."""
    for layer in track_layers(track, center):
        points, widths = layer["points"], layer["widths"]
        if layer["looped"]:  # MeshRoads have no loop flag; close it by repeating the first node
            points, widths = np.vstack([points, points[:1]]), np.append(widths, widths[0])
        road = MeshRoad(layer["material"] or MESH_ROAD_MATERIAL, rid=layer["name"],
                        default_width=float(widths[0]), default_depth=MESH_DEPTH)
        top = ground_z + LAYER_TOP[layer["kind"]]
        road.add_nodes(*[(x, y, top, float(w)) for (x, y), w in zip(points, widths)])
        scenario.add_mesh_road(road)


# ---------------------------------------------------------------- island (template)

def write_island_bundle(user_dir: str | Path, name: str, track: dict, terrain: dict) -> str:
    """Write the heightmap, material map and roads for one track; returns its game-relative folder."""
    from PIL import Image  # installed with matplotlib

    bundle_rel = f"{BUNDLE_DIR}/{name}/"
    out = Path(user_dir) / bundle_rel
    out.mkdir(parents=True, exist_ok=True)

    # GBitmap:getTexel(x, y) reads column x, row y, and terrain arrays are [ix, iy]: transpose.
    # Both maps are 16-bit: the game reads 8-bit PNG texels back as 0.
    rel = np.clip(terrain["height"] - T.BASE_Z, 0.0, T.MAX_HEIGHT)
    h16 = np.round(rel / T.MAX_HEIGHT * 65535).astype(np.uint16)
    m16 = terrain["material"].astype(np.uint16) * MATERIAL_PNG_STEP
    Image.fromarray(np.ascontiguousarray(h16.T)).save(out / "height.png")
    Image.fromarray(np.ascontiguousarray(m16.T)).save(out / "material.png")

    roads = []
    for layer in track_layers(track):
        points, idx, widths = layer["points"], layer["idx"], layer["widths"]
        if len(points) > 2 * DECAL_NODE_STEP:
            # DecalRoads interpolate a smooth spline between nodes, so a node
            # every few metres is plenty; denser roads blow BeamNG's decal
            # geometry budget and segments silently stop rendering.
            keep = np.arange(0, len(points), DECAL_NODE_STEP)
            if not layer["looped"] and keep[-1] != len(points) - 1:
                keep = np.append(keep, len(points) - 1)
            points, idx, widths = points[keep], idx[keep], widths[keep]
        if layer["looped"]:
            # DecalRoad's 'looped' flag leaves a gap at the seam, so run the road
            # a few metres past its own start instead.
            points = np.vstack([points, points[:LOOP_OVERLAP]])
            idx = np.concatenate([idx, idx[:LOOP_OVERLAP]])
            widths = np.concatenate([widths, widths[:LOOP_OVERLAP]])
        z = terrain["road_z"][idx]
        nodes = [[round(float(x), 3), round(float(y), 3), round(float(zz), 3), round(float(w), 3)]
                 for (x, y), zz, w in zip(points, z, widths)]
        # Very long DecalRoads stop rendering partway along, so split into overlapping chunks.
        for k, start in enumerate(range(0, max(len(nodes) - 2, 1), DECAL_CHUNK)):
            roads.append(dict(name=f"{layer['name']}_{k}", material=layer["material"] or DECAL_ROAD_MATERIAL,
                              priority=LAYER_PRIORITY[layer["kind"]], looped=False,
                              nodes=nodes[start:start + DECAL_CHUNK + 3]))
    (out / "roads.json").write_text(json.dumps(roads))
    (out / "materials.json").write_text(json.dumps(list(T.TERRAIN_MATERIALS)))
    return bundle_rel


_APPLY_LUA = r"""
local dir = '/__DIR__'
local N, OX, OY, SQ, MAXH, MATSTEP = __N__, __OX__, __OY__, __SQ__, __MAXH__, __MATSTEP__
local tb = scenetree.findObject('theTerrain')
if not tb then return 'ERR no terrain block' end

-- The template level scatters thousands of rocks; remove them (once).
for _, name in ipairs(scenetree.findClassObjects('Forest') or {}) do
  local f = scenetree.findObject(name)
  if f then f:delete() end
end

-- Material code (index into materials.json) -> terrain block material index,
-- adding any terrain material the block doesn't have yet (e.g. Asphalt).
local codeToIdx = {}
for code, name in ipairs(jsonReadFile(dir .. 'materials.json')) do
  local idx = -1
  for i = 0, tb:getMaterialCount() - 1 do
    if tb:getMaterialName(i) == name then idx = i end
  end
  if idx < 0 then
    tb:addMaterial(name, -1)
    idx = tb:getMaterialCount() - 1
    tb:updateMaterial(idx, name)
  end
  codeToIdx[code - 1] = idx
end

local hb, mb = GBitmap(), GBitmap()
if not hb:loadFile(dir .. 'height.png') then return 'ERR loading height.png' end
if not mb:loadFile(dir .. 'material.png') then return 'ERR loading material.png' end
-- setHeight takes height above the block's base, in [0, MAXH); values past MAXH wrap around.
local scale = (MAXH - 0.01) / 65535
local floor = math.floor
local pos = vec3()
for x = 0, N - 1 do
  for y = 0, N - 1 do
    tb:setHeight(x, y, hb:getTexel(x, y) * scale)
    pos:set(OX + x * SQ, OY + y * SQ, 0)
    tb:setMaterialIdxWs(pos, codeToIdx[floor(mb:getTexel(x, y) / MATSTEP + 0.5)])
  end
end
tb:updateGrid(vec3(0, 0), vec3(N - 1, N - 1))
tb:updateGridMaterials(vec3(0, 0), vec3(N - 1, N - 1))
be:reloadCollision()

local old = scenetree.findObject('drift_track_roads')
if old then old:deleteAllObjects(); old:delete() end
local group = createObject('SimGroup')
group:registerObject('drift_track_roads')
scenetree.MissionGroup:addObject(group)
for _, r in ipairs(jsonReadFile(dir .. 'roads.json')) do
  local road = createObject('DecalRoad')
  road:setField('material', 0, r.material)
  road:setField('drivability', 0, '1')
  road:setField('renderPriority', 0, tostring(r.priority))
  road:setField('looped', 0, r.looped and '1' or '0')
  road:setField('improvedSpline', 0, '1')
  road:registerObject(r.name)
  group:addObject(road)
  for i, n in ipairs(r.nodes) do
    road:insertNode(vec3(n[1], n[2], n[3]), n[4], i - 1)
  end
end
return 'ok'
"""


def apply_island_bundle(bng: BeamNGpy, bundle_rel: str) -> None:
    """Apply a written bundle to the loaded `template` level (terrain, materials, track roads)."""
    chunk = (_APPLY_LUA.replace("__DIR__", bundle_rel)
             .replace("__N__", str(T.GRID_SIZE)).replace("__OX__", str(T.ORIGIN_X))
             .replace("__OY__", str(T.ORIGIN_Y)).replace("__SQ__", str(T.SQUARE_SIZE))
             .replace("__MAXH__", str(T.MAX_HEIGHT)).replace("__MATSTEP__", str(MATERIAL_PNG_STEP)))
    wrapped = f"local ok, r = pcall(function() {chunk} end) return ok and r or ('ERR ' .. tostring(r))"
    result = bng.control.queue_lua_command(wrapped, response=True)
    if result != "ok":
        raise RuntimeError(f"applying track bundle failed: {result}")


# ---------------------------------------------------------------- shared

def start_pose(track: dict, center=TRACK_CENTER, ground_z: float | np.ndarray = GROUND_Z):
    """(pos, rot_quat) placing a vehicle just behind the start line, facing along the track.

    ground_z: a constant, or the per-sample road height (island mode `terrain["road_z"]`).
    """
    pts = world_centerline(track, center)
    tx, ty = pts[1] - pts[-1]
    # BeamNG vehicles face -y at zero rotation.
    yaw = math.degrees(math.atan2(tx, -ty))
    z = float(np.asarray(ground_z)[-5]) if np.ndim(ground_z) else float(ground_z)
    pos = (float(pts[-5, 0]), float(pts[-5, 1]), z + 0.5)
    return pos, angle_to_quat((0.0, 0.0, yaw))


def _normals(pts: np.ndarray) -> np.ndarray:
    """Unit left-hand normals of a closed polyline."""
    tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


def _corner_runs(curvature: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """(start, end) index ranges of consecutive samples with |curvature| > threshold, wrapping the loop."""
    tight = np.abs(curvature) > threshold
    if tight.all() or not tight.any():
        return []
    shift = int(np.argmin(tight))  # start scanning from a non-corner sample so no run straddles the wrap
    runs, start = [], None
    for k in range(len(tight) + 1):
        i = (shift + k) % len(tight)
        if tight[i] and start is None:
            start = shift + k
        elif not tight[i] and start is not None:
            runs.append((start, shift + k - 1))
            start = None
    return runs
