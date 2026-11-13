"""
Random scenarios for the synthetic images (no Blender here, so the sampling
is testable on its own). Every random choice of one image lives in a
``Scenario``; the renderer only draws it. Domain randomization: whatever
varies in real videos (cube brand and style, colours, light, viewpoint,
hands, background, motion, a layer caught mid-turn) varies here, so that the
model can rely on nothing but the cube.
"""
from __future__ import annotations

import colorsys
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..cube import FACE_MOVES, Cube, Move, invert_alg, parse_move

# Colours of the six plastics in sRGB, U R F D L B: white, red, green, yellow, orange, blue
BASE_COLORS = np.array([
    [0.93, 0.93, 0.91],
    [0.78, 0.06, 0.10],
    [0.05, 0.68, 0.28],
    [0.97, 0.86, 0.06],
    [1.00, 0.45, 0.03],
    [0.05, 0.40, 0.88],
])
#: How far each plastic's hue may drift (share of a turn): red, orange and yellow stay apart;
#: greens and blues differ a lot between brands (a real stickerless blue measured at 200-213°)
HUE_SPREAD = (0.0, 0.018, 0.035, 0.018, 0.018, 0.035)

# Algorithms that keep the first two layers: their mixes with U turns cover last-layer states
_LL_ALGS = (
    "R U R' U R U2 R'",                      # Sune
    "R U2 R' U' R U' R'",                    # Antisune
    "R U R' U' R' F R2 U' R' U' R U R' F'",  # T perm
    "F R U' R' U' R U R' F' R U R' U' R' F R F'",  # Y perm
    "R2 U R U R' U' R' U' R' U R'",          # Ua perm
    "F R U R' U' F'",                        # OLL 45
    "r U R' U' r' F R F'",                   # OLL 33-like (wide)
)

# Skin tones in sRGB, light to dark
SKIN_TONES = np.array([
    [0.96, 0.80, 0.69], [0.92, 0.72, 0.58], [0.85, 0.63, 0.49],
    [0.76, 0.53, 0.38], [0.62, 0.42, 0.29], [0.45, 0.30, 0.21], [0.30, 0.20, 0.14],
])


@dataclass
class Turn:
    """A layer drawn rotated by ``angle`` degrees (right-handed about +axis) from ``Scenario.stickers``."""
    move: str
    angle: float
    #: Far from a resting position: the side faces of the layer cannot be read
    turning: bool


@dataclass
class Finger:
    #: Face the finger lies on after wrapping round the cube: "far" (away from the cuber) or "top"
    target: str
    #: Position across that face, in cube units from the centre (-1.5 .. 1.5)
    across: float
    #: How far the finger reaches onto that face from the edge (cube units)
    reach: float


@dataclass
class Hand:
    """A hand holding the cube, in the cuber's frame: palm on the left or right face."""
    side: int  # -1 left hand (on L), +1 right hand (on R)
    palm_depth: float  # palm centre towards the far face (+) or the cuber (-), cube units
    palm_height: float
    palm_scale: float
    finger_radius: float
    fingers: list[Finger]  # index to little finger
    thumb: Finger  # on the near face (towards the cuber)
    skin: list[float]
    #: "blobs": fingers of three ellipsoids; "capsules": round phalanges and knuckles, tapering, with nails
    model: str = "blobs"


@dataclass
class Light:
    azimuth: float
    elevation: float
    distance: float
    energy: float
    kelvin: float
    size: float
    #: RGB filter of a coloured lamp (LED strips, RGB lighting), white when [1, 1, 1]
    tint: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class Scenario:
    index: int
    #: How the state was made (for debugging), and the stickers drawn (StickerCube order)
    recipe: str
    stickers: list[int]
    turn: Turn | None
    #: Unit quaternion (w, x, y, z), cube axes (x R, y U, z F): how the cuber holds the cube.
    #: The cuber looks along -z: the near face is +z, the far face -z, the top +y.
    orientation: list[float]
    #: Camera direction from the cube: azimuth 0 = behind the cuber's head (sees the near face),
    #: 180 = facing the cuber (sees the far face), elevation above the horizon
    azimuth: float
    elevation: float
    #: Camera distance in cube units (a cubie ≈ 19 mm: 12 ≈ 23 cm, 40 ≈ 75 cm)
    distance: float
    #: Full frames: share of the image's longer side the cube spans
    frame_fraction: float
    #: Where the cube sits off the image centre, as shares of the image's width and height
    offset: list[float]
    roll: float
    style: str  # stickerless | stickered | stickered_white
    palette: list[list[float]]
    body_color: list[float]
    sticker_size: float
    bevel: int
    roughness: float
    hands: list[Hand]
    occluders: list[dict]
    background: dict
    table: dict | None
    lights: list[Light]
    ambient: list[float]
    exposure: float
    #: Tone mapping of the camera ("AgX" or "Standard")
    tone: str
    #: Brightness of each of the 54 stickers around its plastic's colour (wear, light falloff)
    tile_jitter: list[float]
    shutter: float
    spin: list[float]  # axis (3) and degrees per frame of the whole cube
    turn_speed: float  # degrees per frame of the turning layer
    #: Centre caps (see _caps), None for plain centres
    caps: dict | None = None
    #: "crop": the image is the crop the app gives the reader (square, the cube's box times
    #: ``crop_margin``, off by ``offset`` as a detector would leave it); "full": a whole frame
    #: of ``aspect`` (width / height) as the camera films it, for the detector
    framing: str = "crop"
    crop_margin: float = 1.5
    aspect: float = 1.0
    #: Full frames without the cube (hands and scene only): the detector must say "no cube"
    cube_hidden: bool = False
    #: The cuber's forearms, torso and head (see _body), None to leave them out
    body: dict | None = None
    #: Things around the cube that are not the cube (see _distractors): the detector must not take them for it
    distractors: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    #: A second layer already turning while ``turn`` ends (corner cutting): the cubies both layers
    #: share stay with ``turn``
    turn_next: Turn | None = None
    #: Fingers pushing the turning layers (finger tricks, see scene._trick), in the cuber's frame:
    #: {"side": ±1, "finger": -1 the whole hand (a wrist turn) | 0 the thumb | 1.. index..., "target",
    #:  "across", "reach" (where the finger lies, see Finger), "axis" (3, cube's axes), "angle" (degrees
    #:  turned with the layer), "weight" (0..1: from the hand's own pose to the push)}
    tricks: list[dict] = field(default_factory=list)
    #: Clear coat on the tiles (glossy stickers, polished plastic), None for the style's own
    coat: float | None = None
    #: A pattern printed in the tiles (carbon, frosted, brushed stickers): {"scale", "amount"}, None for plain
    texture: dict | None = None
    #: The layers along one axis a few degrees off their resting place, as on a real cube (see _loose):
    #: {"axis": 0..2, "angles": degrees of the layers at -2, 0, 2}; the turning layers keep their turn
    slack: dict | None = None
    #: A camera out of focus (see _harsh): sharp at this share of the distance to the cube, at this f-number
    focus: float | None = None
    fstop: float = 1.0


def _random_moves(rng: np.random.Generator, count: int) -> list[str]:
    moves: list[str] = []
    while len(moves) < count:
        m = FACE_MOVES[rng.integers(len(FACE_MOVES))]
        if moves and moves[-1][0] == m[0]:
            continue
        moves.append(m)
    return moves


def _state(rng: np.random.Generator) -> tuple[str, Cube]:
    """A cube state, mixing random states with the near-solved states of the end of a solve."""
    kind = rng.choice(["random", "path", "last_layer", "near_solved", "solved"], p=[0.3, 0.2, 0.2, 0.2, 0.1])
    if kind == "random":
        alg = " ".join(_random_moves(rng, 22))
    elif kind == "path":
        scramble = _random_moves(rng, int(rng.integers(15, 26)))
        undo = invert_alg(" ".join(scramble)).split()
        alg = " ".join(scramble + undo[: int(rng.integers(0, len(undo) + 1))])
    elif kind == "last_layer":
        parts = []
        for _ in range(int(rng.integers(1, 5))):
            parts.append(["", "U", "U2", "U'"][rng.integers(4)])
            parts.append(_LL_ALGS[rng.integers(len(_LL_ALGS))])
        # The last layer may sit on any face: the solver can hold the cube any way
        alg = " ".join(p for p in parts if p)
    elif kind == "near_solved":
        alg = " ".join(_random_moves(rng, int(rng.integers(1, 9))))
    else:
        alg = ["", "U", "U2", "U'"][rng.integers(4)]
    return f"{kind}: {alg}", Cube().apply(alg) if alg else Cube()


#: The site's 3×3 algorithms by set (scripts/vision/export-algs.ts)
ALGS: dict[str, list[str]] = json.loads((Path(__file__).parent / "algs.json").read_text())
#: The 24 ways to hold a cube: the step's colour (cross, last layer) on any face, colour neutral
_HOLDS24 = [f"{a} {b}".strip() for a in ("", "x", "x2", "x'", "z", "z'") for b in ("", "y", "y2", "y'")]
#: The four F2L slots, from the front-right one: the algorithms for it turned to the others
_SLOTS = ("", "y", "y2", "y'")
#: The steps of a solve whose states are drawn (_stage_state), and how often
STAGES = {"scrambled": 0.24, "cross": 0.1, "f2l": 0.17, "last_slot": 0.08, "oll": 0.13, "pll": 0.11, "auf": 0.03,
          "roux": 0.06, "mid_alg": 0.08}


def _auf(rng: np.random.Generator) -> str:
    return ["", "U", "U2", "U'"][rng.integers(4)]


def _pairs(rng: np.random.Generator, slots: list[int], count: int) -> list[str]:
    """F2L algorithms (from the site's sets) on the given slots, between turns of the top: the pairs of those
    slots out, the cross and the other slots kept."""
    out = []
    for _ in range(count):
        slot = _SLOTS[slots[rng.integers(len(slots))]]
        alg = str(rng.choice(ALGS["f2l"] if rng.random() < 0.6 else ALGS["zbls"] + ALGS["vls"] + ALGS["wv"]))
        out += [_auf(rng), f"{slot} {alg} {invert_alg(slot)}".strip() if slot else alg]
    return out


def _last_layer(rng: np.random.Generator, sets: tuple[str, ...], count: int) -> list[str]:
    out = []
    for _ in range(count):
        out += [_auf(rng), str(rng.choice(ALGS[str(rng.choice(sets))]))]
    return out


def _stage_state(rng: np.random.Generator) -> tuple[str, Cube]:
    """A state of a real solve (CFOP or Roux), the step's colour on any face: scrambled, the cross done, some or
    all of the F2L pairs, the last layer oriented or not, or caught in the middle of an algorithm."""
    stage = str(rng.choice(list(STAGES), p=np.array(list(STAGES.values())) / sum(STAGES.values())))
    if stage == "mid_alg":
        stage, cut = str(rng.choice(["f2l", "last_slot", "oll", "pll"])), True
    else:
        cut = False
    if stage == "scrambled":
        # A scramble, or the cross being built (a few moves off a done cross)
        moves = _random_moves(rng, 22) if rng.random() < 0.6 else _pairs(rng, [0, 1, 2, 3], 8) + _random_moves(rng, int(rng.integers(1, 6)))
    elif stage == "cross":
        moves = _pairs(rng, [0, 1, 2, 3], int(rng.integers(6, 12))) + _last_layer(rng, ("oll", "pll"), 1)
    elif stage == "f2l":
        open_slots = sorted(rng.choice(4, size=int(rng.integers(2, 4)), replace=False).tolist())
        moves = _pairs(rng, open_slots, int(rng.integers(3, 8))) + _last_layer(rng, ("oll", "pll"), 1)
    elif stage == "last_slot":
        moves = _pairs(rng, [int(rng.integers(4))], int(rng.integers(1, 3))) + _last_layer(rng, ("oll", "pll"), 1)
    elif stage == "oll":
        moves = _last_layer(rng, ("oll", "pll", "coll", "ollcp", "two_look"), int(rng.integers(1, 4)))
    elif stage == "pll":
        moves = _last_layer(rng, ("pll",), int(rng.integers(1, 3)))
    elif stage == "auf":
        moves = [_auf(rng)]
    else:
        # Roux: the blocks done, the corners and the M slice left (CMLL, then LSE)
        moves = _last_layer(rng, ("cmll",), int(rng.integers(1, 3))) + [str(rng.choice(["M", "M'", "M2", "M U M'", "M' U2 M"]))
                                                                        for _ in range(int(rng.integers(0, 4)))]
    # The solver's moves undo these: the state is reached by their inverse (the case they solve)
    alg = invert_alg(" ".join(m for m in moves if m))
    if cut:
        # Partway through the next algorithm
        nxt = str(rng.choice(ALGS["f2l" if stage in ("f2l", "last_slot") else "oll" if stage == "oll" else "pll"])).split()
        alg = f"{alg} {' '.join(nxt[: int(rng.integers(1, len(nxt)))])}"
    hold = _HOLDS24[rng.integers(24)]
    return f"{stage}{'+cut' if cut else ''}: {hold} | {alg}", Cube().apply(f"{hold} {alg} {invert_alg(hold)}" if hold else alg)


#: Within this many degrees of a resting position, a layer counts as settled
SETTLED = 7.0


def _turn(rng: np.random.Generator, cube: Cube) -> tuple[Cube, Turn | None]:
    """Maybe catch a layer mid-turn. Returns the cube to draw and the residual turn."""
    if rng.random() > 0.45:
        return cube, None
    r = rng.random()
    if r < 0.85:
        token = FACE_MOVES[rng.integers(len(FACE_MOVES))]
    elif r < 0.95:
        token = ["M", "M'", "M2", "E", "E'", "S", "S'"][rng.integers(7)]
    else:
        token = ["Rw", "Rw'", "Lw", "Lw'", "Uw", "Fw'"][rng.integers(6)]
    move = parse_move(token)
    assert move is not None
    theta = rng.random() * move.angle()
    # Draw from the nearest resting position so that the residual stays within ±45°
    quarters = int(round(theta / 90.0))
    residual = theta - 90.0 * quarters
    if quarters:
        quarter = Move(move.family, move.first, move.last, 1)
        if math.copysign(1, quarter.angle()) != math.copysign(1, quarters):
            quarter = Move(move.family, move.first, move.last, 3)
        cube = cube.apply([quarter] * abs(quarters))
    return cube, Turn(token, residual, abs(residual) > SETTLED)


def _quaternion_multiply(a: list[float], b: list[float]) -> list[float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _axis_angle(axis: np.ndarray, degrees: float) -> list[float]:
    axis = axis / (np.linalg.norm(axis) or 1.0)
    h = math.radians(degrees) / 2
    return [math.cos(h), *(float(c) * math.sin(h) for c in axis)]


def _random_rotation(rng: np.random.Generator) -> list[float]:
    """Uniform random unit quaternion (Shoemake)."""
    u1, u2, u3 = rng.random(3)
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    return [b * math.cos(2 * math.pi * u3), a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2), b * math.sin(2 * math.pi * u3)]


#: The 24 ways to hold a cube, as quaternions (products of quarter turns about x, y, z)
def _holds() -> list[list[float]]:
    quarters = [_axis_angle(np.array(a, dtype=float), d) for a in ([1, 0, 0], [0, 1, 0], [0, 0, 1]) for d in (0, 90, 180, 270)]
    found: list[list[float]] = []
    for a in quarters:
        for b in quarters:
            q = _quaternion_multiply(a, b)
            # q and -q are the same rotation
            if not any(abs(sum(x * y for x, y in zip(q, f, strict=True))) > 1 - 1e-6 for f in found):
                found.append(q)
    return found


HOLDS = _holds()


def _orientation(rng: np.random.Generator) -> list[float]:
    """Held with a face to each side (any of the 24 ways), tilted a little; sometimes caught mid-rotation."""
    if rng.random() < 0.2:
        return _random_rotation(rng)
    hold = HOLDS[rng.integers(len(HOLDS))]
    tilt = _axis_angle(rng.normal(size=3), float(rng.normal(0, 10)))
    return _quaternion_multiply(tilt, hold)


def _palette(rng: np.random.Generator, style: str) -> list[list[float]]:
    """The six plastics of one cube: each colour shifted a little, never into its neighbour."""
    out = []
    vivid = style == "stickerless"
    for k, rgb in enumerate(BASE_COLORS):
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        if k == 0:  # white: may be greyish, bluish or warm
            s = rng.uniform(0.0, 0.08)
            h = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.82, 1.0)
        else:
            h = (h + rng.uniform(-HUE_SPREAD[k], HUE_SPREAD[k])) % 1.0
            s = float(np.clip(s * rng.uniform(0.8 if not vivid else 0.95, 1.1), 0, 1))
            v = float(np.clip(v * rng.uniform(0.8, 1.08), 0, 1))
        out.append(list(colorsys.hsv_to_rgb(h, s, v)))
    return out


def _hand(rng: np.random.Generator, side: int, skin: np.ndarray) -> Hand:
    fingers = [
        Finger(
            target="top" if (k == 0 and rng.random() < 0.5) or rng.random() < 0.12 else "far",
            across=float(np.clip(0.9 - 0.75 * k + rng.normal(0, 0.25), -1.6, 1.6)),
            reach=float(rng.uniform(0.2, 2.2)),
        )
        for k in range(int(rng.choice([2, 3, 4], p=[0.2, 0.3, 0.5])))
    ]
    return Hand(
        side=side,
        palm_depth=float(rng.uniform(-0.6, 0.9)),
        palm_height=float(rng.uniform(-1.2, 0.6)),
        palm_scale=float(rng.uniform(0.85, 1.2)),
        finger_radius=float(rng.uniform(0.32, 0.44)),
        fingers=fingers,
        thumb=Finger("near", float(rng.uniform(-1.2, 1.0)), float(rng.uniform(0.2, 2.2))),
        skin=list(np.clip(skin * rng.uniform(0.93, 1.05, size=3), 0, 1)),
    )


def _real_hand(rng: np.random.Generator, side: int, skin: np.ndarray) -> Hand:
    """A cuber's hand as filmed: four fingers on the far face (the index often on top), their tips
    mostly near the edge, the thumb on the near face; slimmer than the blobs."""
    fingers = []
    for k in range(4):
        reach = float(rng.uniform(0.1, 0.9) if rng.random() < 0.75 else rng.uniform(0.9, 1.8))
        top = (k == 0 and rng.random() < 0.4) or rng.random() < 0.08
        fingers.append(Finger("top" if top else "far", float(np.clip(1.05 - 0.68 * k + rng.normal(0, 0.15), -1.5, 1.5)), reach))
    return Hand(
        side=side,
        palm_depth=float(rng.uniform(-0.6, 0.6)),
        palm_height=float(rng.uniform(-1.1, 0.3)),
        palm_scale=float(rng.uniform(0.85, 1.15)),
        finger_radius=float(rng.uniform(0.3, 0.38)),
        fingers=fingers,
        thumb=Finger("near", float(rng.uniform(-1.0, 0.8)), float(rng.uniform(0.1, 1.2))),
        skin=list(np.clip(skin * rng.uniform(0.93, 1.05, size=3), 0, 1)),
        model="capsules",
    )


def _vary(rng: np.random.Generator, sc: Scenario) -> None:
    """What differs between real cubes and rooms beyond the base draws: faded or worn cubes, a window
    behind the cube, a dim room, one hard spot light."""
    if rng.random() < 0.15:  # faded plastics (old cube, sun)
        fade, dim = rng.uniform(0.55, 0.8), rng.uniform(0.85, 1.0)
        sc.palette = [list(colorsys.hsv_to_rgb(h, s * fade, v * dim)) for h, s, v in (colorsys.rgb_to_hsv(*c) for c in sc.palette)]
        sc.meta["variety"] = "faded"
    if rng.random() < 0.1:  # worn or dirty stickers
        worn = rng.random(54) < rng.uniform(0.1, 0.4)
        sc.tile_jitter = [float(j * rng.uniform(0.7, 0.9)) if w else j for j, w in zip(sc.tile_jitter, worn, strict=True)]
        sc.meta["variety"] = "worn"
    light = rng.random()
    if light < 0.15:  # against the light
        sc.background["glow"] = float(rng.uniform(1.5, 8.0))
        sc.meta["light"] = "backlit"
    elif light < 0.25:  # a dim room
        sc.exposure = float(rng.uniform(-2.5, -1.2))
        sc.meta["light"] = "dim"
    elif light < 0.35:  # one hard spot: sharp shadows, glare on the stickers
        sc.lights = [dataclasses.replace(sc.lights[0], size=float(rng.uniform(0.2, 0.8)), energy=float(rng.uniform(1500, 4000)))]
        sc.meta["light"] = "spot"


#: The cuber's axes x (right), y (up), z (towards the cuber) in Blender's world (see scene.py): a whole-cube
#: rotation x, y or z spins about one of them
CUBER_AXES_BL = ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0])


def _wild(rng: np.random.Generator, sc: Scenario) -> None:
    """Footage as it comes, beyond _vary: the cube never held straight and always moving, whole-cube
    rotations caught mid-way, glare and blown highlights, very bright rooms, glossy, frosted, printed
    and fluo stickers, faces seen almost edge-on, and the detector's crop further off."""
    tags = []
    # Never straight: a tilt of up to ~45° on top of the hold's own (the cuber looks, regrips, wobbles)
    if rng.random() < 0.7:
        sc.orientation = _quaternion_multiply(_axis_angle(rng.normal(size=3), float(np.clip(rng.normal(0, 22), -50, 50))), sc.orientation)
    # Always moving: still, wobbling, or a whole-cube rotation x, y, z mid-way (fast, blurred)
    motion = rng.random()
    if motion < 0.25:
        axis = CUBER_AXES_BL[int(rng.integers(3))]
        sc.spin = [*axis, float(rng.uniform(8, 35))]
        sc.shutter = float(rng.uniform(0.3, 1.0))
        tags.append("rotation")
    elif motion < 0.8:
        sc.spin = [*(float(x) for x in rng.normal(0, 1, size=3)), float(rng.uniform(1, 9))]
        sc.shutter = max(sc.shutter, float(rng.uniform(0.2, 1.0)))
        tags.append("wobble")
    # Cameras from anywhere: from below the cube to straight above it, rolled further
    if rng.random() < 0.35:
        sc.elevation = float(rng.uniform(-45, 85))
    if rng.random() < 0.2:
        sc.roll = float(rng.uniform(-70, 70))
    # Light
    light = rng.random()
    if light < 0.12:  # a lamp near the camera: glare on glossy faces
        sc.lights[0] = dataclasses.replace(sc.lights[0], azimuth=float(rng.uniform(-30, 30)), elevation=float(rng.uniform(10, 55)),
                                           size=float(rng.uniform(0.3, 1.5)), energy=float(rng.uniform(2000, 6000)))
        sc.roughness = float(rng.uniform(0.04, 0.2))
        sc.coat = float(rng.uniform(0.5, 1.0))
        tags.append("glare")
    elif light < 0.2:  # blown highlights: white and yellow close, every colour washed out
        sc.exposure = float(rng.uniform(2.0, 3.5))
        tags.append("overexposed")
    elif light < 0.3:  # a very bright room, daylight from everywhere
        sc.ambient = [float(a) for a in rng.uniform(0.8, 1.6) * rng.uniform(0.9, 1.1, size=3)]
        tags.append("bright")
    elif light < 0.38:  # straight against a bright window
        sc.background["glow"] = float(rng.uniform(6, 20))
        sc.exposure = float(rng.uniform(-1.0, 1.0))
        tags.append("backlit")
    # The cube
    finish = rng.random()
    if finish < 0.25:
        sc.coat = float(rng.uniform(0.4, 1.0)) if sc.coat is None else sc.coat
        sc.roughness = min(sc.roughness, float(rng.uniform(0.05, 0.3)))
        tags.append("glossy")
    elif finish < 0.4:
        sc.coat, sc.roughness = 0.0, float(rng.uniform(0.6, 0.9))
        tags.append("frosted")
    if rng.random() < 0.15:
        sc.texture = {"scale": float(np.exp(rng.uniform(np.log(15), np.log(250)))), "amount": float(rng.uniform(0.08, 0.35))}
        tags.append("printed")
    if rng.random() < 0.1:  # fluo stickers: brighter, lime green, pinkish red, sky blue
        shift = {1: -0.025, 2: -0.06, 5: -0.04}
        sc.palette = [list(colorsys.hsv_to_rgb((h + shift.get(k, 0.0) * rng.random()) % 1.0, min(1.0, s * 1.1), min(1.0, v * 1.15)))
                      for k, (h, s, v) in enumerate(colorsys.rgb_to_hsv(*c) for c in sc.palette)]
        tags.append("fluo")
    if sc.style != "stickerless":
        sc.sticker_size = float(rng.uniform(0.72, 0.94))
    # The detector further off: a looser or tighter crop, off centre
    if sc.framing == "crop":
        sc.crop_margin = float(rng.uniform(1.2, 1.95))
        sc.offset = [float(x) for x in rng.uniform(-0.12, 0.12, size=2)]
    sc.meta["wild"] = "+".join(tags) or "none"


def _loose(rng: np.random.Generator, sc: Scenario) -> None:
    """A real cube as it is solved, beyond _wild: its layers never quite aligned (a few degrees off, still
    settled), turns and whole-cube rotations faster (longer blur, still settled for a rotation)."""
    tags = []
    # Off by a few degrees along the turning layer's axis (the other axes are locked while it turns)
    if not sc.turn_next and rng.random() < 0.75:
        move = parse_move(sc.turn.move) if sc.turn else None
        axis = move.axis if move else int(rng.integers(3))
        sc.slack = {"axis": axis, "angles": [float(np.clip(rng.normal(0, 2.5), -SETTLED + 1, SETTLED - 1)) for _ in range(3)]}
        tags.append("slack")
    if sc.turn and sc.turn.turning:
        sc.turn_speed = float(rng.uniform(10, 70))
    elif rng.random() < 0.4:
        # A fast whole-cube rotation (x, y, z, or a glance at another side) or regrip, smeared, its layers
        # settled: not a turn
        sc.spin = [*CUBER_AXES_BL[int(rng.integers(3))], float(rng.uniform(10, 60))] if rng.random() < 0.6 \
            else [*(float(x) for x in rng.normal(0, 1, size=3)), float(rng.uniform(8, 35))]
        sc.shutter = float(rng.uniform(0.5, 1.0))
        tags.append("smear")
    sc.meta["loose"] = "+".join(tags) or "none"


def _harsh(rng: np.random.Generator, sc: Scenario) -> None:
    """The worst footage, beyond _loose: glances (fast x, y, z, a look at another side) and shakes smeared over
    the frame, cameras out of focus, burnt, oversaturated or dark colours."""
    tags = []
    if not (sc.turn and sc.turn.turning):
        if rng.random() < 0.55:
            sc.spin = [*CUBER_AXES_BL[int(rng.integers(3))], float(rng.uniform(20, 70))] if rng.random() < 0.7 \
                else [*(float(x) for x in rng.normal(0, 1, size=3)), float(rng.uniform(15, 45))]
            sc.shutter = float(rng.uniform(0.7, 1.0))
            tags.append("glance")
    elif rng.random() < 0.35:  # the hand shakes while it turns
        sc.spin = [*(float(x) for x in rng.normal(0, 1, size=3)), float(rng.uniform(5, 20))]
        sc.shutter = max(sc.shutter, float(rng.uniform(0.6, 1.0)))
        tags.append("shake")
    if rng.random() < 0.3:
        sc.focus = float(rng.uniform(0.4, 0.75)) if rng.random() < 0.5 else float(rng.uniform(1.4, 3.0))
        sc.fstop = float(np.exp(rng.uniform(np.log(0.6), np.log(3.0))))
        tags.append("defocus")
    light = rng.random()
    if light < 0.15:
        sc.exposure = float(rng.uniform(1.8, 3.5))
        tags.append("burnt")
    elif light < 0.3:
        boost, bright = float(rng.uniform(1.2, 1.7)), float(rng.uniform(1.0, 1.2))
        sc.palette = [list(colorsys.hsv_to_rgb(h, min(1.0, s * boost), min(1.0, v * bright)))
                      for h, s, v in (colorsys.rgb_to_hsv(*c) for c in sc.palette)]
        tags.append("saturated")
    elif light < 0.38:
        sc.exposure = float(rng.uniform(-3.0, -1.8))
        tags.append("dark")
    sc.meta["harsh"] = "+".join(tags) or "none"


def _azimuth(rng: np.random.Generator) -> float:
    """Mostly facing the cuber (a webcam, a phone on the desk) or over their shoulder; rarely from a side."""
    kind = rng.choice(["facing", "behind", "side"], p=[0.5, 0.3, 0.2])
    centre = {"facing": 180.0, "behind": 0.0, "side": float(rng.choice([90.0, 270.0]))}[kind]
    return float((centre + rng.uniform(-55, 55)) % 360)


def _occluder(rng: np.random.Generator, skin: np.ndarray) -> dict:
    """A blob near the cube's surface (a knuckle, a sleeve, the other hand passing by)."""
    direction = rng.normal(size=3)
    direction /= np.abs(direction).max() or 1.0
    color = np.clip(skin * rng.uniform(0.9, 1.05), 0, 1) if rng.random() < 0.7 else rng.random(3)
    return {
        "position": [float(x) for x in direction * rng.uniform(1.9, 2.6)],
        "size": [float(x) for x in rng.uniform(0.3, 1.0, size=3)],
        "color": [float(x) for x in color],
    }


def _caps(rng: np.random.Generator) -> dict | None:
    """Real centres often differ from the other stickers: an embossed disc on each, a logo on one (often white)."""
    if rng.random() < 0.4:
        return None
    logo = int(rng.choice(6, p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])) if rng.random() < 0.6 else None
    logo_color = [[0.92] * 3, [0.03] * 3, [0.55] * 3, [float(x) for x in rng.random(3)]][int(rng.choice(4, p=[0.3, 0.35, 0.2, 0.15]))]
    return {
        "all": bool(logo is None or rng.random() < 0.5),  # a disc on every centre
        "size": float(rng.uniform(0.45, 0.85)),  # diameter, share of the tile
        "height": float(rng.uniform(0.002, 0.012)),
        "shade": float(rng.uniform(0.8, 1.12)),  # colour of the discs against their tile
        "gloss": float(rng.uniform(0.4, 1.4)),
        "logo": logo,  # colour of the centre that carries it
        "logo_color": logo_color,
    }


def _body(rng: np.random.Generator, skin: np.ndarray) -> dict:
    """The cuber around the cube: forearms from the wrists (bare or in sleeves), torso and head."""
    clothing = [float(x) for x in (rng.random(3) * rng.uniform(0.3, 1.0))]
    return {
        "clothing": clothing,
        "sleeves": bool(rng.random() < 0.45),
        # Per hand (left, right): direction jitter, length and radius of the forearm, cube units
        "forearms": [
            {"spread": float(rng.uniform(0.0, 0.6)), "drop": float(rng.uniform(0.2, 0.9)),
             "length": float(rng.uniform(9, 13)), "radius": float(rng.uniform(1.0, 1.5))}
            for _ in range(2)
        ],
        "chest": float(rng.uniform(9, 16)),  # distance from the cube to the chest
        "height": float(rng.uniform(4, 9)),  # chest centre below the cube
        "skin": [float(x) for x in np.clip(skin * rng.uniform(0.9, 1.05, size=3), 0, 1)],
    }


def _ambient(rng: np.random.Generator) -> list[float]:
    """Light from the room around the cube: a grey level, from a dim lamp to a window in daylight, barely tinted."""
    level = rng.uniform(0.03, 0.6)
    return [float(level * t) for t in rng.uniform(0.85, 1.15, size=3)]


#: Hues of coloured room lighting (RGB LEDs): magenta, purple, blue, cyan, pink, red, green
LED_HUES = (300, 275, 235, 195, 330, 0, 120)


def _coloured(rng: np.random.Generator, sc: Scenario) -> None:
    """Now and then a room lit by coloured LEDs (a real solve filmed under magenta light turned blue to
    violet and white to pink): a hue on some lamps, and the room's light of that hue too."""
    if rng.random() >= 0.3:
        return
    hue = (LED_HUES[int(rng.integers(len(LED_HUES)))] + rng.uniform(-15, 15)) % 360 / 360
    tint = [float(c) for c in colorsys.hsv_to_rgb(hue, rng.uniform(0.3, 0.85), 1.0)]
    lit = [light for light in sc.lights if rng.random() < 0.7] or sc.lights[:1]
    for light in lit:
        light.tint = tint
    mix = rng.uniform(0.4, 1.0)
    sc.ambient = [a * (1 - mix + mix * t) for a, t in zip(sc.ambient, tint, strict=True)]
    sc.meta["lighting"] = "coloured"


#: Look-alikes and things of a room: boxes covered with squares of colour (2×2 to 8×8 cells, never the
#: cube's 3×3), plain boxes, bottles and cups, posters and books
DISTRACTOR_KINDS = ("tiles", "plain", "cylinder", "panel")


def _distractor(rng: np.random.Generator, framing: str) -> dict:
    kind = str(rng.choice(DISTRACTOR_KINDS, p=[0.4, 0.2, 0.2, 0.2]))
    if kind == "cylinder":
        size = [float(rng.uniform(0.8, 3)), 0.0, float(rng.uniform(2, 8))]
        size[1] = size[0]
    elif kind == "panel":
        size = [float(rng.uniform(2, 9)), float(rng.uniform(2, 9)), float(rng.uniform(0.05, 0.6))]
    else:
        size = [float(x) for x in rng.uniform(1.5, 7, size=3)]
    cells = [int(c) for c in rng.choice([2, 4, 5, 6, 8], size=3)] if kind in ("tiles", "panel") else [1, 1, 1]
    if kind == "cylinder" and rng.random() < 0.5:
        cells = [1, 1, int(rng.integers(2, 7))]  # bands of colour, a label
    vivid = rng.random() < 0.7
    colours = [[float(c) for c in (rng.uniform(0, 1, 3) ** (2 if vivid else 0.5))] for _ in range(6)]
    if kind == "plain":
        colours = [colours[0]] * 6
    behind = framing == "crop" or rng.random() < 0.75
    return {
        "kind": kind,
        "size": size,
        "rotation": [float(x) for x in rng.normal(size=4)],
        # Behind the cube (cube units along the view), or beside it; across the view (shares of its half-sides)
        "depth": float(rng.uniform(1.5, 10) if behind else rng.uniform(-2.5, 1.0)),
        "across": [float(x) for x in (rng.uniform(-1, 1, 2) if behind else rng.choice([-1, 1], 2) * rng.uniform(0.5, 1, 2))],
        "cells": cells,
        "gap": float(rng.uniform(0.02, 0.12)) if rng.random() < 0.7 else 0.0,
        "colours": colours,
        "seed": float(rng.uniform(0, 100)),
        "roughness": float(rng.uniform(0.1, 0.8)),
    }


def _framing(rng: np.random.Generator, framing: str) -> dict:
    if framing == "crop":
        # The app crops a square of the detected box's longer side times 1.5 (src/lib/vision/detector.ts)
        return {"crop_margin": float(rng.uniform(1.3, 1.7)), "aspect": 1.0, "cube_hidden": False}
    aspect = float(rng.choice([16 / 9, 4 / 3, 9 / 16, 3 / 4, 1.0], p=[0.5, 0.15, 0.25, 0.05, 0.05]))
    return {
        "frame_fraction": float(np.exp(rng.uniform(np.log(0.04), np.log(0.6)))),
        "aspect": aspect,
        "cube_hidden": bool(rng.random() < 0.04),
    }


def sample(seed: int, index: int, *, backgrounds: int = 0, framing: str = "crop", distractors: bool = False,
           real_hands: bool = False, variety: bool = False, wild: bool = False, loose: bool = False,
           stages: bool = False, harsh: bool = False) -> Scenario:
    """Scenario number ``index`` of a dataset with seed ``seed`` (deterministic). ``distractors``: things
    around the cube that are not it, and more frames without the cube (drawn after all the rest, so the
    scenario is otherwise the same)."""
    if framing not in ("crop", "full"):
        raise ValueError(f"unknown framing {framing}")
    rng = np.random.default_rng([seed, index])
    recipe, cube = _state(rng)
    if stages:
        # The states of real solves, drawn apart: the rest of the scenario stays the same
        recipe, cube = _stage_state(np.random.default_rng([seed, index, 105]))
    cube, turn = _turn(rng, cube)
    style = str(rng.choice(["stickerless", "stickered", "stickered_white"], p=[0.55, 0.4, 0.05]))
    skin = SKIN_TONES[rng.integers(len(SKIN_TONES))]
    hand_count = int(rng.choice([0, 1, 2], p=[0.15, 0.2, 0.65]))
    sides = [-1, 1] if hand_count == 2 else [int(rng.choice([-1, 1]))] if hand_count == 1 else []
    bg_kinds = ["procedural", "skin", "wall"] + (["image"] if backgrounds else [])
    bg_p = [0.45, 0.25, 0.3] if not backgrounds else [0.3, 0.2, 0.15, 0.35]
    bg_kind = str(rng.choice(bg_kinds, p=bg_p))
    lights = [
        Light(
            azimuth=float(rng.uniform(-150, 150)),
            elevation=float(rng.uniform(5, 80)),
            distance=float(rng.uniform(8, 20)),
            energy=float(rng.uniform(300, 2500)),
            kelvin=float(rng.uniform(2700, 7500)),
            size=float(rng.uniform(1, 8)),
        )
        for _ in range(int(rng.integers(1, 4)))
    ]
    blur = rng.random() < 0.6
    sc = Scenario(
        index=index,
        recipe=recipe,
        stickers=cube.stickers.tolist(),
        turn=turn,
        orientation=_orientation(rng),
        azimuth=_azimuth(rng),
        elevation=float(rng.uniform(-25, 70)),
        distance=float(rng.uniform(12, 40)),
        frame_fraction=float(rng.uniform(0.45, 0.8)),
        offset=[float(x) for x in rng.uniform(-0.12, 0.12, size=2)],
        roll=float(np.clip(rng.normal(0, 15), -40, 40)),
        style=style,
        palette=_palette(rng, style),
        body_color=(
            [float(rng.uniform(0.88, 0.97))] * 3 if style == "stickered_white"
            else [float(rng.uniform(0.01, 0.06))] * 3
        ),
        sticker_size=float(rng.uniform(0.94, 0.985) if style == "stickerless" else rng.uniform(0.78, 0.92)),
        bevel=int(rng.integers(3)),
        roughness=float(rng.uniform(0.15, 0.6)),
        hands=[_hand(rng, s, skin) for s in sides],
        occluders=[_occluder(rng, skin) for _ in range(int(rng.choice([0, 1, 2, 3], p=[0.55, 0.25, 0.12, 0.08])))],
        background={
            "kind": bg_kind,
            "colors": [[float(x) for x in rng.random(3)] for _ in range(3)],
            "skin": list(np.clip(skin * rng.uniform(0.85, 1.05), 0, 1)),
            "scale": float(rng.uniform(1, 30)),
            "image": int(rng.integers(backgrounds)) if backgrounds else -1,
            "distance": float(rng.uniform(4, 25)),
        },
        table={"color": [float(x) for x in rng.random(3)], "height": float(rng.uniform(2.5, 6))} if rng.random() < 0.4 else None,
        lights=lights,
        ambient=_ambient(rng),
        exposure=float(np.clip(rng.normal(0.8, 0.7), -1.5, 2.5)),
        tone=str(rng.choice(["AgX", "Standard"], p=[0.6, 0.4])),
        tile_jitter=[float(x) for x in rng.uniform(0.95, 1.04, size=54)],
        shutter=float(rng.uniform(0.2, 1.0)) if blur else 0.0,
        spin=[*(float(x) for x in rng.normal(0, 1, size=3)), float(abs(rng.normal(0, 6)))],
        turn_speed=float(rng.uniform(10, 45)) if turn and turn.turning else float(rng.uniform(0, 8)),
        caps=_caps(rng),
        meta={"hand_count": hand_count},
    )
    # Drawn last, so that the draws above stay the same whatever the framing
    sc.body = _body(rng, skin) if rng.random() < 0.85 else None
    extra = _framing(rng, framing)
    sc.framing = framing
    sc.crop_margin = extra.get("crop_margin", 1.5)
    sc.aspect = extra["aspect"]
    sc.cube_hidden = extra["cube_hidden"]
    if framing == "crop":
        # A detector's error: the crop a little off the cube's centre
        sc.offset = [x * 2 / 3 for x in sc.offset]
    else:
        sc.frame_fraction = extra["frame_fraction"]
        # Anywhere in the frame, now and then partly out of it
        room = 0.5 - 0.3 * sc.frame_fraction * np.array([1 / min(1.0, sc.aspect), max(1.0, sc.aspect)])
        sc.offset = [float(x) for x in rng.uniform(-room, room)]
    # Drawn after all the others, which stay the same as before it existed
    _coloured(rng, sc)
    if distractors:
        count = int(rng.choice([0, 1, 2, 3, 4], p=[0.2, 0.25, 0.25, 0.18, 0.12] if framing == "full" else [0.5, 0.3, 0.2, 0.0, 0.0]))
        sc.distractors = [_distractor(rng, framing) for _ in range(count)]
        if framing == "full" and not sc.cube_hidden and rng.random() < 0.06:
            sc.cube_hidden = True  # 10 % in all: a room, hands, look-alikes, and no cube
    if real_hands:
        # Their own draws: the rest of the scenario stays the same
        hands = np.random.default_rng([seed, index, 101])
        sc.hands = [_real_hand(hands, h.side, np.array(h.skin)) for h in sc.hands]
    if variety:
        _vary(np.random.default_rng([seed, index, 102]), sc)
    if wild:
        _wild(np.random.default_rng([seed, index, 103]), sc)
    if loose:
        _loose(np.random.default_rng([seed, index, 104]), sc)
    if harsh:
        _harsh(np.random.default_rng([seed, index, 106]), sc)
    return sc
