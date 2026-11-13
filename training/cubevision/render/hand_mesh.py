"""A real hand (Blender Studio's Human Base Meshes, CC0) that holds the cube: its fingers found on the mesh,
a skeleton of hinges for them, and a pose per frame (each finger curled until it touches the cube).

The asset is built once from the bundle's .blend (``python -m cubevision.render.hand_mesh <bundle.blend>``)
into ``data/assets/hand.npz``: the mesh (a right hand, metres), the joints of each finger, and per vertex
the two bones it follows. Posing is numpy only (linear blend skinning), so a frame costs a millisecond.

Bones: 0 the hand itself (palm and wrist), then per finger (thumb, index, middle, ring, little) three.
"""
from __future__ import annotations

import heapq
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSET = Path(__file__).parents[2] / "data" / "assets" / "hand.npz"
BUNDLE = Path(__file__).parents[2] / "data" / "assets" / "human-base-meshes" / "human-base-meshes-bundle-v1.4.1" / "human_base_meshes_bundle.blend"
# Shares of a finger from its knuckle (MCP) to its tip: phalanges (anatomy, middle finger 4.4 / 2.9 / 2.3 cm)
PHALANGES = (0.46, 0.30, 0.24)
THUMB = (0.43, 0.30, 0.27)  # metacarpal, proximal, distal


def _neighbours(n: int, faces: np.ndarray) -> list[set[int]]:
    nb: list[set[int]] = [set() for _ in range(n)]
    for f in faces:
        for a, b in zip(f, np.roll(f, -1), strict=True):
            nb[a].add(int(b))
            nb[b].add(int(a))
    return nb


def _geodesic(verts: np.ndarray, nb: list[set[int]], seeds) -> np.ndarray:
    g = np.full(len(verts), np.inf)
    heap = [(0.0, int(s)) for s in seeds]
    for s in seeds:
        g[s] = 0.0
    while heap:
        d, u = heapq.heappop(heap)
        if d > g[u]:
            continue
        for w in nb[u]:
            nd = d + float(np.linalg.norm(verts[u] - verts[w]))
            if nd < g[w]:
                g[w] = nd
                heapq.heappush(heap, (nd, w))
    return g


def _component(nb: list[set[int]], start: int, keep: np.ndarray) -> set[int]:
    seen, todo = {start}, [start]
    while todo:
        u = todo.pop()
        for w in nb[u]:
            if keep[w] and w not in seen:
                seen.add(w)
                todo.append(w)
    return seen


def _normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.zeros_like(verts)
    for k in range(faces.shape[1]):
        a, b, c = verts[faces[:, k]], verts[faces[:, (k + 1) % faces.shape[1]]], verts[faces[:, k - 1]]
        np.add.at(n, faces[:, k], np.cross(b - a, c - a))
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def build(verts: np.ndarray, faces: np.ndarray) -> dict:
    """The rig of a right hand whose fingers point down (-z), palm towards +y, thumb towards +x."""
    nb = _neighbours(len(verts), faces)
    g = _geodesic(verts, nb, np.where(verts[:, 2] > verts[:, 2].max() - 0.01)[0])
    # Fingertips: the five points farthest from the wrist along the skin, apart from each other
    tips: list[int] = []
    for u in np.argsort(-g):
        near = {int(u)}
        for _ in range(4):
            near |= {w for x in near for w in nb[x]}
        if g[u] < max(g[w] for w in near):
            continue  # not the farthest of its neighbourhood: up a finger, not its tip
        if all(np.linalg.norm(verts[u] - verts[t]) > 0.015 for t in tips):
            tips.append(int(u))
        if len(tips) == 5:
            break
    tips.sort(key=lambda t: -verts[t, 0])  # thumb (+x), index, middle, ring, little
    # Each finger: the skin farther from the wrist than where it parts from its neighbours
    regions, webs = [], []
    for t in tips:
        lo, hi = 0.0, float(g[t])
        for _ in range(30):
            mid = (lo + hi) / 2
            part = _component(nb, t, g >= mid)
            lo, hi = (mid, hi) if any(o in part for o in tips if o != t) else (lo, mid)
        regions.append(_component(nb, t, g >= hi + 0.002))
        webs.append(hi)
    palmar, radial = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    joints, axes, radii = [], [], []
    label = np.full(len(verts), -1)
    for c, (t, region, web) in enumerate(zip(tips, regions, webs, strict=True)):
        idx = np.array(sorted(region))
        label[idx] = c
        # The centre line: the middle of the finger's skin at each distance from the wrist
        centres = []
        for a in np.arange(web, g[t] - 0.004, 0.003):
            band = idx[(g[idx] >= a) & (g[idx] < a + 0.003)]
            if len(band) >= 4:
                centres.append(verts[band].mean(0))
        centres = np.array(centres)
        base, tip = centres[: max(2, len(centres) // 3)].mean(0), verts[t]
        direction = tip - centres[0]
        direction /= np.linalg.norm(direction)
        length = float(np.dot(tip - centres[0], direction))
        # The knuckle lies in the palm, a quarter of the finger before where the skin parts (the thumb's
        # first joint, deep at the wrist, nearly half)
        knuckle = centres[0] - direction * length * (1.3 if c == 0 else 0.45)
        webs[c] = length * (1.3 if c == 0 else 0.45)  # from the knuckle to where the finger leaves the palm
        total = float(np.linalg.norm(tip - knuckle))
        shares = THUMB if c == 0 else PHALANGES
        points = [knuckle]
        for s in np.cumsum(shares)[:-1]:
            # Along the centre line where it exists, the straight line before
            p = knuckle + direction * total * s
            near = centres[np.argmin(np.linalg.norm(centres - p, axis=1))]
            points.append(near if np.linalg.norm(near - p) < 0.012 else p)
        points.append(tip - direction * 0.004)
        joints.append(np.array(points))
        # Curling: towards the palm (the thumb across it, towards the little finger's side)
        towards = palmar if c else 0.6 * palmar - 0.8 * radial
        axis = np.cross(direction, towards)
        axes.append(axis / np.linalg.norm(axis))
        radii.append(float(np.median(np.linalg.norm(np.cross(verts[idx] - base, direction), axis=1))))
    joints_a = np.array(joints)
    # Weights: each vertex follows the nearest bone of its finger (the palm's near a knuckle, the knuckle's
    # finger), blended over a joint's width with the bone before or after
    n = len(verts)
    bone = np.zeros((n, 2), np.int32)
    weight = np.zeros((n, 2), np.float32)
    weight[:, 0] = 1.0
    width = 0.009
    for v in range(n):
        p = verts[v]
        c = label[v]
        if c < 0:
            # Palm: the skin before where a finger leaves the palm (the finger's base, in line with it)
            # follows the finger more and more, from none to all of it
            best_c, share = -1, 0.0
            for k in range(5):
                u = joints_a[k, 1] - joints_a[k, 0]
                u = u / np.linalg.norm(u)
                a = float(np.dot(p - joints_a[k, 0], u))
                side = float(np.linalg.norm(p - joints_a[k, 0] - a * u))
                blend = webs[k] * (0.5 if k == 0 else 0.6)
                if webs[k] - blend < a < webs[k] + width:
                    # Less and less to the side of the finger's line too, none at two and a half radii
                    f = float(_smooth(np.array((a - webs[k] + blend) / blend)) * _smooth(np.array((2.5 * radii[k] - side) / (1.5 * radii[k]))))
                    if f > share:
                        best_c, share = k, f
            if best_c >= 0:
                bone[v] = (1 + 3 * best_c, 0)
                weight[v] = (share, 1 - share)
            continue
        j = joints_a[c]
        best, best_d, best_a = 0, np.inf, 0.0
        for k in range(3):
            u = j[k + 1] - j[k]
            ln = float(np.linalg.norm(u))
            a = float(np.dot(p - j[k], u / ln))
            d = float(np.linalg.norm(p - (j[k] + u / ln * np.clip(a, 0, ln))))
            if d < best_d:
                best, best_d, best_a = k, d, a
        k, a = best, best_a
        ln = float(np.linalg.norm(j[k + 1] - j[k]))
        this = 1 + 3 * c + k
        before = 0 if k == 0 else this - 1
        if a < width:
            # Near the joint at the bone's start: shared with the bone before (the palm before the first)
            share = 0.5 * _smooth(np.array((width - a) / (2 * width)))
            if label[v] < 0 and a < 0:
                share = 0.5 + 0.5 * float(_smooth(np.array(-a / width)))
            bone[v] = (this, before)
            weight[v] = (1 - share, share)
        elif a > ln - width and k < 2:
            share = 0.5 * float(_smooth(np.array((a - (ln - width)) / (2 * width))))
            bone[v] = (this, this + 1)
            weight[v] = (1 - share, share)
        else:
            bone[v] = (this, this)
            weight[v] = (1.0, 0.0)
    # Nails: the back of each last bone's outer half
    normals = _normals(verts, faces)
    nail = np.zeros(n, np.float32)
    for c in range(5):
        j = joints_a[c]
        u = (j[3] - j[2]) / np.linalg.norm(j[3] - j[2])
        back = -np.cross(axes[c], u)  # away from the curl
        idx = np.where(label == c)[0]
        a = (verts[idx] - j[2]) @ u / np.linalg.norm(j[3] - j[2])
        side = normals[idx] @ back
        nail[idx] = _smooth((a - 0.25) / 0.15) * _smooth((side - 0.45) / 0.2)
    return {"verts": verts.astype(np.float32), "faces": faces.astype(np.int32), "joints": joints_a.astype(np.float32),
            "axes": np.array(axes, np.float32), "radii": np.array(radii, np.float32), "bone": bone, "weight": weight,
            "nail": nail, "label": label.astype(np.int8)}


@dataclass
class Rig:
    verts: np.ndarray
    faces: np.ndarray
    joints: np.ndarray  # (5, 4, 3): knuckle, two joints, tip per finger
    axes: np.ndarray  # (5, 3) curling axis per finger
    radii: np.ndarray
    bone: np.ndarray
    weight: np.ndarray
    nail: np.ndarray

    @classmethod
    def load(cls, path: Path = ASSET) -> Rig:
        d = np.load(path)
        return cls(**{k: d[k] for k in ("verts", "faces", "joints", "axes", "radii", "bone", "weight", "nail")})

    def skin(self, transforms: np.ndarray) -> np.ndarray:
        """The vertices moved by the bones' transforms ((16, 3, 4), rest to posed)."""
        h = np.concatenate([self.verts, np.ones((len(self.verts), 1), np.float32)], axis=1)
        a = np.einsum("nij,nj->ni", transforms[self.bone[:, 0]], h)
        b = np.einsum("nij,nj->ni", transforms[self.bone[:, 1]], h)
        return a * self.weight[:, :1] + b * self.weight[:, 1:]


def rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The shortest rotation taking direction a to direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return np.eye(3)
    return rotation(axis, math.degrees(math.atan2(s, float(np.dot(a, b)))))


def main(argv: list[str]) -> int:
    import bpy

    bundle = Path(argv[0]) if argv else BUNDLE
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(bundle)) as (_src, dst):
        dst.objects = ["Hand  - Realistic"]
    obj = dst.objects[0]
    bpy.context.scene.collection.objects.link(obj)
    # The sculpted level (multires 1: 3300 vertices), in the object's own frame
    mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    verts = np.array([v.co[:] for v in mesh.vertices], np.float64)
    faces = np.array([list(p.vertices) for p in mesh.polygons], np.int64)
    rig = build(verts, faces)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ASSET, **rig)
    print(f"{ASSET}: {len(verts)} vertices, finger radii {np.round(rig['radii'] * 100, 2)} cm")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


# ─── Holding the cube ──────────────────────────────────────────────────────────

HALF = 1.5  # half the cube's side, cube units
UNIT = 0.88 / 0.019  # cube units per metre: a cube of 5.7 cm, hands a little smaller than the mesh's
# The rest hand's axes in the cuber's frame, a right hand: radial (+x) up, palmar (+y) towards the cube
# (-x), proximal (+z) towards the cuber (-y); its fingers along the right face towards the far one
REST = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
# The curls tried per finger (degrees): the knuckle's, a little back to well forward, and the middle joint's
# (the last joint follows it, two thirds of it, as in a real finger)
KNUCKLE = ((-20.0, 50.0), (-10.0, 90.0))  # the thumb's, the fingers'
MIDDLE = ((0.0, 80.0), (0.0, 110.0))
STEP = 6.0  # the search's first pass, every STEP degrees; then a quarter of it round the best
LAST = 0.7
OFF = 1.4  # the palm's distance off the side face (cube units)


def _box_distance(p: np.ndarray) -> np.ndarray:
    """Signed distance of points to the cube (rounded edges)."""
    q = np.abs(p) - (HALF - 0.1)
    out = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    return out + np.minimum(q.max(axis=-1), 0.0) - 0.1


def _turn(v: np.ndarray, axis: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    """Vectors v turned about the unit axes by each angle (Rodrigues), broadcast: v and axis (..., 3)."""
    t = np.radians(degrees)[..., None]
    return v * np.cos(t) + np.cross(axis, v) * np.sin(t) + axis * np.sum(axis * v, axis=-1, keepdims=True) * (1 - np.cos(t))


def _reach(base: np.ndarray, bones: np.ndarray, axis: np.ndarray, target: np.ndarray, normal: np.ndarray,
           radii: list[float], thumb: bool = False, extra: np.ndarray | None = None) -> tuple[int, float, float]:
    """The knuckle's and middle joint's curls bringing the tip of a chain (base, then 3 bones, straight)
    to the target, the phalanges outside the cube and on their hand's half, for the best of D chains
    (base and axis (D, 3), bones (3, D, 3); ``extra``: a cost per chain): which, and its two curls.
    Every STEP degrees, then finer round the best."""
    d, k1, k2 = _search(base, bones, axis, target, normal, radii, thumb, extra, None)
    one = lambda x: x[d:d + 1]  # noqa: E731
    _, k1, k2 = _search(one(base), bones[:, d:d + 1], one(axis), target, normal, radii, thumb, None, (k1, k2))
    return d, k1, k2


def _search(base, bones, axis, target, normal, radii, thumb, extra, near) -> tuple[int, float, float]:
    """``_reach`` over a grid of curls: every STEP degrees, or finer round ``near``."""
    (a1, b1), (a2, b2) = KNUCKLE[not thumb], MIDDLE[not thumb]
    if near is None:
        k1, k2 = np.meshgrid(np.arange(a1, b1 + 1e-6, STEP), np.arange(a2, b2 + 1e-6, STEP), indexing="ij")
    else:
        fine = np.arange(-STEP, STEP + 1e-6, STEP / 4)
        k1, k2 = np.meshgrid(np.clip(near[0] + fine, a1, b1), np.clip(near[1] + fine, a2, b2), indexing="ij")
    at = lambda x: x[:, None, None, :]  # noqa: E731 (D, 1, 1, 3): against the curls (A, B)
    ax = at(axis)
    p0 = at(base)
    p1 = p0 + _turn(at(bones[0]), ax, k1)
    p2 = p1 + _turn(at(bones[1]), ax, k1 + k2)
    p3 = p2 + _turn(at(bones[2]), ax, k1 + k2 * (1 + LAST))
    # On the target's face above all (its normal), near the target on it
    cost = ((p3 - target) @ normal) ** 2 + 0.15 * np.sum((p3 - target) ** 2, axis=-1)
    share = np.array([0.3, 0.65, 1.0])[:, None]
    for k, ((a_, b_), r) in enumerate(zip(((p0, p1), (p1, p2), (p2, p3)), radii, strict=True)):
        pts = a_[..., None, :] + (b_ - a_)[..., None, :] * share  # (D, A, B, 3, 3)
        gap = _box_distance(pts) - r
        across = np.maximum(0.0, 0.5 * r - pts[..., 0])
        cost = cost + 40 * np.sum(np.maximum(0.0, -gap) ** 2 + across ** 2, axis=-1)
        if k:
            # The last two phalanges along the cube, not in a loop away from it
            cost = cost + 0.15 * np.sum(np.maximum(0.0, gap - 0.15) ** 2, axis=-1)
    # A little curl rather than a knuckle bent backwards, all else alike
    cost = cost + 0.3 * np.maximum(0.0, -k1) / 20 + 0.02 * (np.abs(k1) / 60) + 0.05 * (k2 / 120)
    if extra is not None:
        cost = cost + extra[:, None, None]
    d, i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return int(d), float(k1[i, j]), float(k2[i, j])


def _sphere(n: int) -> np.ndarray:
    """n directions spread evenly (Fibonacci)."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    a = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z], axis=1)


DIRECTIONS = _sphere(200)


@dataclass
class Pose:
    transforms: np.ndarray  # (16, 3, 4): rest (metres, right hand) to the cuber's frame
    wrist: np.ndarray  # where the forearm leaves the mesh, and its direction (towards the elbow)
    forearm: np.ndarray


def targets(hand, radii: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Where each fingertip goes (a point, and its face's normal), a right hand: the thumb on the near face, the fingers on the far one
    (Finger: across is the height, reach how far onto the face from the right edge), an index on the
    top face when its target is "top"; short of the middle, where the other hand's fingers are."""
    thumb = hand.thumb
    out = [(np.array([float(np.clip(1.05 - 0.3 * thumb.reach, 0.45, 1.0)), -HALF - radii[0], float(np.clip(thumb.across, -0.9, 1.1))]),
            np.array([0.0, -1.0, 0.0]))]
    for k in range(4):
        finger = hand.fingers[k] if k < len(hand.fingers) else None
        across = finger.across if finger else 1.05 - 0.68 * k
        reach = finger.reach if finger else 0.5
        x = max(0.55, HALF - 0.15 - 0.6 * reach)
        if finger is not None and finger.target == "top":
            out.append((np.array([x, float(np.clip(across, -0.4, 1.2)), HALF + radii[k + 1]]), np.array([0.0, 0.0, 1.0])))
        else:
            out.append((np.array([x, HALF + radii[k + 1], float(np.clip(across, -1.7, 1.4))]), np.array([0.0, 1.0, 0.0])))
    return out


_GRIPS: dict = {}


def _grip(rig: Rig, hand, seed: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The still grip (``hold`` without the tricks): the transforms, each finger's knuckle and tip; kept
    (some 20 ms each: see ``prime``). One rig: the asset's."""
    key = (repr(hand), seed)
    if key in _GRIPS:
        return _GRIPS[key]
    if len(_GRIPS) > 20000:
        _GRIPS.clear()
    rng = None if seed is None else np.random.default_rng(seed)
    scale = UNIT * float(np.clip(hand.palm_scale, 0.85, 1.1))  # hands of 15 to 19 cm
    # The hand pitched up, the forearm coming from below (more when the hand is low)
    pitch = float(np.clip(25 - 20 * hand.palm_height, 10, 50))
    rot = rotation(np.array([1.0, 0.0, 0.0]), pitch) @ REST
    gap = rig.radii[2] * scale + 0.05
    # The knuckles off the far edge, the palm off the side (the cube held in the fingers, as cubers do):
    # the fingers curl round onto the far face
    target = np.array([HALF + gap + OFF, HALF + 0.2 * hand.palm_depth, -0.5 + 0.35 * hand.palm_height])
    move = target - scale * rot @ rig.joints[2, 0]
    transforms = np.zeros((16, 3, 4))
    knuckles, tips = np.zeros((5, 3)), np.zeros((5, 3))

    def place(r: np.ndarray, anchor_rest: np.ndarray, anchor_world: np.ndarray) -> np.ndarray:
        """Rest to world: rotation r, the rest point anchor_rest at anchor_world."""
        m = np.zeros((3, 4))
        m[:, :3] = r * scale
        m[:, 3] = anchor_world - m[:, :3] @ anchor_rest
        return m

    transforms[0] = place(rot, np.zeros(3), move)
    world = lambda p: scale * rot @ p + move  # noqa: E731
    radii = rig.radii * scale
    aims = targets(hand, radii)
    for c in range(5):
        joints = rig.joints[c]
        base = world(joints[0])
        aim, normal = aims[c]
        if rng is not None:
            aim = aim + rng.normal(0, 0.06, 3) * np.array([1, 0, 1])
        if c == 0:
            # The thumb: its first bone (in the palm) turned any way within 55 degrees of its rest, the
            # two others curled; the turn bringing the tip best onto its target, the least turn alike
            rest = rot @ (joints[1] - joints[0])
            rest /= np.linalg.norm(rest)
            tube = [float(radii[0]) * (1 - 0.08 * k) for k in range(3)]
            ds = DIRECTIONS[DIRECTIONS @ rest > math.cos(math.radians(55))]
            rs = np.array([swing(rest, d) @ rot for d in ds])  # (D, 3, 3)
            starts = base + rs @ (joints[1] - joints[0]) * scale
            mids = (base + starts) / 2
            inside = np.maximum(0.0, tube[0] - _box_distance(np.stack([mids, starts]))) ** 2
            axes = rs @ rig.axes[0]
            bones = np.stack([rs @ (joints[2] - joints[1]), rs @ (joints[3] - joints[2]), rs @ (joints[3] - joints[2]) * 1e-3]) * scale
            extra = 40 * inside.sum(0) + 0.3 * (1 - ds @ rest)
            d, k1, k2 = _reach(starts, bones, axes, aim, normal, [*tube[1:], tube[2]], thumb=True, extra=extra)
            r, start, axis = rs[d], starts[d], axes[d]
            transforms[1] = place(r, joints[0], base)
            first, bends = 1, (k1, k1 + k2)
        else:
            # Spread (about the axis across the palm: abduction) so that the target is in the plane the
            # finger curls in, as far as a knuckle goes sideways; then its curls bring the tip there
            curl = rot @ rig.axes[c]
            n = np.cross(curl, rot @ (joints[3] - joints[0]))
            n /= np.linalg.norm(n)
            v = aim - base
            phi = math.degrees(math.atan2(-float(curl @ v), float(np.cross(n, curl) @ v)))
            phi = (phi + 90) % 180 - 90
            r = rotation(n, float(np.clip(phi, -25.0, 25.0))) @ rot
            axis = r @ rig.axes[c]
            bones = [r @ (joints[k + 1] - joints[k]) * scale for k in range(3)]
            tube = [float(radii[c]) * (1 - 0.08 * k) for k in range(3)]
            _, k1, k2 = _reach(base[None], np.array(bones)[:, None], axis[None], aim, normal, tube)
            first, start, bends = 0, base, (k1, k1 + k2, k1 + k2 * (1 + LAST))
        point = start
        for k, bend in zip(range(first, 3), bends, strict=True):
            g = rotation(axis, bend) @ r
            transforms[1 + 3 * c + k] = place(g, joints[k], point)
            point = point + g @ (joints[k + 1] - joints[k]) * scale
        knuckles[c], tips[c] = base, point
    _GRIPS[key] = (transforms, knuckles, tips)
    return _GRIPS[key]


_RIG: Rig | None = None


def _grip_job(job: tuple) -> tuple:
    global _RIG
    if _RIG is None:
        _RIG = Rig.load()
    return _grip(_RIG, *job)


def prime(jobs: list[tuple], workers: int) -> None:
    """The grips of many (hand, seed) solved ahead, in ``workers`` processes: a video's, before its frames
    (the asset's rig: the only one there is)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    todo = list({(repr(h), s): (h, s) for h, s in jobs if (repr(h), s) not in _GRIPS}.values())
    if not todo:
        return
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        for (h, s), grip in zip(todo, pool.map(_grip_job, todo, chunksize=max(1, len(todo) // (8 * workers))), strict=True):
            _GRIPS[(repr(h), s)] = grip


def hold(rig: Rig, hand, tricks=(), seed: int | None = None) -> Pose:
    """The hand ``hand`` (sampler.Hand) holding the cube, as a right hand (a left one is its mirror:
    see ``mirror_trick``): the palm placed by its depth, height and size, then each finger spread towards
    its target (``targets``) and curled to reach it without going into the cube. ``tricks``: this hand's
    pushes and wrist turn (Scenario.tricks), in the right hand's frame; ``seed``: the targets' jitter."""
    still, knuckles, tips = _grip(rig, hand, seed)
    transforms = still.copy()
    wrist_trick = next((t for t in tricks if t["finger"] < 0), None)
    if wrist_trick:
        # The wrist's turn (about the cube's centre): the whole hand
        whole = rotation(np.array(wrist_trick["axis"], float), wrist_trick["angle"] * wrist_trick["weight"])
        transforms = np.einsum("ij,bjk->bik", whole, transforms)
        knuckles, tips = knuckles @ whole.T, tips @ whole.T
    for c in range(5):
        push = next((t for t in tricks if t["finger"] == c), None)
        if push:
            # Pushing a layer: the finger swung about its knuckle so that its tip goes round with the layer
            q = rotation(np.array(push["axis"], float), push["angle"] * push["weight"])
            knuckle, tip = knuckles[c], tips[c]
            s = swing(tip - knuckle, q @ tip - knuckle)
            for k in range(3):
                m = transforms[1 + 3 * c + k]
                m[:, :3], m[:, 3] = s @ m[:, :3], knuckle + s @ (m[:, 3] - knuckle)
    top = np.array([0.0, 0.0, rig.verts[:, 2].max()])
    wrist = transforms[0][:, :3] @ (top - np.array([0.0, 0.0, 0.02])) + transforms[0][:, 3]
    forearm = transforms[0][:, :3] @ np.array([0.0, 0.0, 1.0])
    return Pose(transforms, wrist, forearm / np.linalg.norm(forearm))


MIRROR = np.diag([-1.0, 1.0, 1.0])


def mirror_trick(trick: dict) -> dict:
    """A left hand's push in the right hand's frame (x mirrored): axis (x, -y, -z), the same angle."""
    a = trick["axis"]
    return {**trick, "axis": [a[0], -a[1], -a[2]]}
