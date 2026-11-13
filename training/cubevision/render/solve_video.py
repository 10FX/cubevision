"""
Synthetic solve videos, with the truth known: a fixed scene per video (cube,
hands, cuber, light, background, camera), then a CFOP solve played at a
human's pace, written as the cuber turns it (rotations, wide moves and slices
included), the cube tilting in the hands all along. They measure the whole
system end to end (detector, reader, search, the cuber's rotations, timings):
see ml2/HANDOFF.md (phase 2) and scripts/vision/eval.test.ts.

    python -m cubevision.render.solve_video --out evals/synthetic --count 20 --seed 21

Writes ``<name>.webm`` (VP9, as a browser records), ``<name>.truth.json`` and
``synthetic.json`` (the list of the videos, their scrambles and their truth files).

The solve is built backwards, so that every stage's end is known: a cross,
four F2L pairs (with y between some of them), an OLL and a PLL, each from real
algorithms that keep what the earlier stages solved; the scramble is its
inverse. The stickers stay in the cube's own frame: the cuber's rotations only
change how the cube is held (Scenario.orientation), and each move written in
the cuber's frame is turned on the layer of the cube that faces that way.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..cube import Cube, Move, _format, invert_alg, parse_move
from . import hand_mesh
from .sampler import SKIN_TONES, Finger, Hand, Scenario, Turn, _axis_angle, _hand, _quaternion_multiply, _real_hand, sample

# ─── The solve, written as the cuber turns it (cross on D, after an x2) ──────

#: Insertions of a pair into the front-right slot: they keep the cross
F2L_ALGS = (
    "R U R'", "U R U' R'", "R U' R' U R U' R'", "U' R U R' U R U R'", "R U2 R' U' R U R'", "U' R U' R' U R U R'",
    "F' U' F", "U' F' U F", "U R U2 R' U R U' R'", "R U R' U' R U R' U' R U R'", "U' R U2 R' U2 R U' R'",
)
#: Orientations of the last layer: they keep the first two layers (two with wide moves)
OLL_ALGS = (
    "R U R' U R U2 R'", "R U2 R' U' R U' R'", "F R U R' U' F'", "F R U R' U' R U R' U' F'", "r U R' U' r' F R F'",
    "R U R' U' R' F R F'", "r U R' U R U2 r'", "F U R U' R' F'", "R U2 R2 U' R2 U' R2 U2 R",
)
#: Permutations of the last layer: they keep its orientation (three with M slices, one with an x rotation)
PLL_ALGS = (
    "R U R' U' R' F R2 U' R' U' R U R' F'", "R U' R U R U R U' R' U' R2", "R2 U R U R' U' R' U' R' U R'",
    "M2 U M U2 M' U M2", "M2 U M2 U2 M2 U M2", "M' U M2 U M2 U M' U2 M2", "x R' U R' D2 R U' R' D2 R2 x'",
    "F R U' R' U' R U R' F' R U R' U' R' F R F'", "R U R' F' R U R' U' R' F R2 U' R'",
)
AUF = ("", "U", "U'", "U2")
FACE_TURNS = tuple(f + s for f in "URFDLB" for s in ("", "'", "2"))

# Faces by (axis, side) and slices by axis, in any frame: x R, y U, z F
_DIRECTION = {"R": (0, 1), "L": (0, -1), "U": (1, 1), "D": (1, -1), "F": (2, 1), "B": (2, -1)}
_FACE_AT = {v: k for k, v in _DIRECTION.items()}
#: A slice turns as the face it follows: M as L, E as D, S as F
_SLICE = {"M": (0, -1), "E": (1, -1), "S": (2, 1)}
_SLICE_ON = {0: "M", 1: "E", 2: "S"}


def rotation_matrix(axis: int, degrees: float) -> np.ndarray:
    """Right-handed rotation about a frame axis, rounded to integers for quarter turns."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    m = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return np.rint(m) if abs(degrees) % 90 == 0 else m


def in_cube_frame(token: str, hold: np.ndarray) -> str | None:
    """A move written in the cuber's frame, as the move of the cube's own layers (None for a rotation).
    ``hold`` takes the cube's axes to the cuber's (the cube's x, y, z in its columns)."""
    move = parse_move(token)
    if move is None:
        raise ValueError(token)
    if move.family in ("x", "y", "z"):
        return None
    slice_ = move.family in _SLICE
    axis, side = (_SLICE if slice_ else _DIRECTION)[move.family]
    direction = np.zeros(3)
    direction[axis] = side
    seen = hold.T @ direction  # where that side of the cube is, in the cube's frame
    axis = int(np.argmax(np.abs(seen)))
    side = int(np.sign(seen[axis]))
    if not slice_:
        return _format(Move(_FACE_AT[(axis, side)], move.first, move.last, move.turns), move.turns)
    family = _SLICE_ON[axis]
    turns = move.turns if side == _SLICE[family][1] else (4 - move.turns) % 4
    return _format(Move(family, 0, 0, turns), turns)


def in_cuber_frame(token: str, hold: np.ndarray) -> str:
    """A move of the cube's own layers, as the cuber writes it holding the cube ``hold`` (in_cube_frame's inverse)."""
    move = parse_move(token)
    if move is None or move.family in ("x", "y", "z"):
        raise ValueError(token)
    slice_ = move.family in _SLICE
    axis, side = (_SLICE if slice_ else _DIRECTION)[move.family]
    direction = np.zeros(3)
    direction[axis] = side
    seen = hold @ direction  # where that side of the cube is, in the cuber's frame
    axis = int(np.argmax(np.abs(seen)))
    side = int(np.sign(seen[axis]))
    if not slice_:
        return _format(Move(_FACE_AT[(axis, side)], move.first, move.last, move.turns), move.turns)
    family = _SLICE_ON[axis]
    turns = move.turns if side == _SLICE[family][1] else (4 - move.turns) % 4
    return _format(Move(family, 0, 0, turns), turns)


#: Whole-cube rotations a cuber does mid-solve (y most, then x, z rarest), and their weights
ROTATIONS = ("y", "y'", "y2", "x", "x'", "x2", "z", "z'", "z2")
ROTATION_P = np.array([5, 5, 3, 3, 3, 2, 2, 2, 1], dtype=float) / 26


def reheld(rng: np.random.Generator, tokens: list[tuple[str, str]], rate: float) -> list[tuple[str, str]]:
    """The same solve turned by a cuber who also rotates the whole cube now and then (``rate``: a rotation
    before that share of the moves): every cube layer turns as before, written in the hold of the moment,
    so the scramble and the stages stay the same."""
    old, new = np.eye(3), np.eye(3)
    out: list[tuple[str, str]] = []
    for k, (token, stage) in enumerate(tokens):
        if k and rng.random() < rate and out[-1][0][0] not in "xyz" and token[0] not in "xyz":
            rotation = str(rng.choice(ROTATIONS, p=ROTATION_P))
            new = turned_hold(rotation, new)
            out.append((rotation, stage))
        if token[0] in "xyz":
            old, new = turned_hold(token, old), turned_hold(token, new)
            out.append((token, stage))
        else:
            out.append((in_cuber_frame(in_cube_frame(token, old), new), stage))
    return out


#: Looks at another side of the cube, turned back after: x' to see the back, y to see a side (and their weights)
PEEKS = ("x", "x'", "y", "y'", "y2", "z", "z'")
PEEK_P = np.array([4, 3, 3, 3, 1, 1, 1], dtype=float) / 16


def peeked(rng: np.random.Generator, tokens: list[tuple[str, str]], rate: float) -> list[tuple[str, str]]:
    """The same solve by a cuber who now and then turns the cube to look at another side, then turns it back
    (``rate``: a look before that share of the moves): the hold is the same after, and so are the moves."""
    out: list[tuple[str, str]] = []
    for k, (token, stage) in enumerate(tokens):
        if k and rng.random() < rate and out[-1][0][0] not in "xyz" and token[0] not in "xyz":
            look = str(rng.choice(PEEKS, p=PEEK_P))
            out += [(look, stage), (invert_alg(look), stage)]
        out.append((token, stage))
    return out


def turned_hold(token: str, hold: np.ndarray) -> np.ndarray:
    """The hold after a rotation written in the cuber's frame (x, y2, z'…)."""
    move = parse_move(token)
    assert move is not None and move.family in ("x", "y", "z")
    return rotation_matrix(move.axis, move.angle()) @ hold


def quaternion_of(m: np.ndarray) -> list[float]:
    """Rotation matrix → unit quaternion (w, x, y, z), from its largest component (Shepperd): right for
    half turns too, about any axis (the signs of the others come from it, not from the matrix's zeros)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    k = int(np.argmax([trace, m[0, 0], m[1, 1], m[2, 2]]))
    if k == 0:
        r = math.sqrt(1 + trace) * 2
        q = [r / 4, (m[2, 1] - m[1, 2]) / r, (m[0, 2] - m[2, 0]) / r, (m[1, 0] - m[0, 1]) / r]
    elif k == 1:
        r = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / r, r / 4, (m[0, 1] + m[1, 0]) / r, (m[0, 2] + m[2, 0]) / r]
    elif k == 2:
        r = math.sqrt(1 - m[0, 0] + m[1, 1] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / r, (m[0, 1] + m[1, 0]) / r, r / 4, (m[1, 2] + m[2, 1]) / r]
    else:
        r = math.sqrt(1 - m[0, 0] - m[1, 1] + m[2, 2]) * 2
        q = [(m[1, 0] - m[0, 1]) / r, (m[0, 2] + m[2, 0]) / r, (m[1, 2] + m[2, 1]) / r, r / 4]
    if q[0] < 0:
        q = [-v for v in q]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def slerp(a: list[float], b: list[float], t: float) -> list[float]:
    dot = sum(p * q for p, q in zip(a, b, strict=True))
    if dot < 0:
        b, dot = [-q for q in b], -dot
    if dot > 0.9995:
        out = [p + t * (q - p) for p, q in zip(a, b, strict=True)]
    else:
        theta = math.acos(dot)
        out = [(math.sin((1 - t) * theta) * p + math.sin(t * theta) * q) / math.sin(theta) for p, q in zip(a, b, strict=True)]
    n = math.sqrt(sum(v * v for v in out))
    return [v / n for v in out]


@dataclass
class Step:
    """A token written by the cuber, played from ``start`` to ``end`` (ms)."""
    token: str
    start: float
    end: float
    stage: str  # inspection, cross, f2l, oll, pll


@dataclass
class Solve:
    scramble: str
    steps: list[Step]
    #: End of each stage (ms): cross, f2l, oll, pll (the solve's end)
    stages: dict[str, float]
    tps: float
    #: Tilt of the cube in the hands: amplitude (degrees) and how fast it wanders
    tilt: float
    tilt_rate: float
    tilt_axes: list[list[float]] = field(default_factory=list)
    #: Sudden moves of the cube in the hands, gone again: (start ms, length ms, axis, degrees)
    jerks: list[tuple[float, float, list[float], float]] = field(default_factory=list)

    @property
    def written(self) -> str:
        return " ".join(s.token for s in self.steps)

    @property
    def duration(self) -> float:
        return self.steps[-1].end + 600.0  # the cube held still a moment, solved


def _cross(rng: np.random.Generator) -> list[str]:
    moves: list[str] = []
    while len(moves) < int(rng.integers(5, 9)):
        m = FACE_TURNS[int(rng.integers(len(FACE_TURNS)))]
        if not moves or moves[-1][0] != m[0]:
            moves.append(m)
    return moves


def plan(rng: np.random.Generator, *, wide: bool = True) -> tuple[list[tuple[str, str]], str]:
    """The tokens of a solve with their stage, and its scramble (in the scrambling orientation)."""
    tokens: list[tuple[str, str]] = [("x2", "cross")]
    tokens += [(m, "cross") for m in _cross(rng)]

    def add(stage: str, draw) -> None:
        # A stage does not start on the layer the last one ended on (no F' then F across two stages)
        for _ in range(50):
            moves = draw().split()
            if not moves or _layer(moves[0]) != _layer(tokens[-1][0]):
                break
        tokens.extend((m, stage) for m in moves)

    for pair in range(4):
        if pair and rng.random() < 0.45:
            tokens.append((str(rng.choice(["y", "y'", "y2"])), "f2l"))
        add("f2l", lambda: AUF[int(rng.integers(4))] + " " + F2L_ALGS[int(rng.integers(len(F2L_ALGS)))])
    olls = OLL_ALGS if wide else tuple(a for a in OLL_ALGS if "r" not in a)
    plls = PLL_ALGS if wide else tuple(a for a in PLL_ALGS if "M" not in a)
    for stage, algs in (("oll", olls), ("pll", plls)):
        add(stage, lambda algs=algs: AUF[int(rng.integers(4))] + " " + algs[int(rng.integers(len(algs)))])
    end_auf = AUF[int(rng.integers(4))]
    if end_auf and _layer(end_auf) != _layer(tokens[-1][0]):
        tokens.append((end_auf, "pll"))
    tokens = merged(tokens)
    # The moves in the cube's frame, whose inverse is the scramble
    hold = np.eye(3)
    cube_moves = []
    for token, _ in tokens:
        if parse_move(token).family in ("x", "y", "z"):  # type: ignore[union-attr]
            hold = turned_hold(token, hold)
        else:
            cube_moves.append(in_cube_frame(token, hold))
    return tokens, invert_alg(" ".join(cube_moves))


def _layer(token: str) -> tuple:
    move = parse_move(token)
    assert move is not None
    return move.family, move.first, move.last


def merged(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Consecutive turns of the same layer within a stage as one (U' then U2 is U): no cuber turns U' U.
    Across stages they stay apart, so that each stage still ends with its own last move."""
    out: list[tuple[str, str]] = []
    for token, stage in tokens:
        move = parse_move(token)
        assert move is not None
        if out and out[-1][1] == stage:
            last = parse_move(out[-1][0])
            assert last is not None
            if (last.family, last.first, last.last) == (move.family, move.first, move.last):
                turns = (last.turns + move.turns) % 4
                out.pop()
                if turns:
                    out.append((_format(move, turns), stage))
                continue
        out.append((token, stage))
    return out


def timed(rng: np.random.Generator, tokens: list[tuple[str, str]], scramble: str, tps: float | None = None, cut: float = 0.0) -> Solve:
    """A human's pace: inspection, quick bursts of moves, pauses to look for the next pair or case.
    ``cut``: how often a turn that follows another without a pause starts before it ends (corner cutting)."""
    drawn = float(rng.uniform(3, 8))
    tps = drawn if tps is None else tps
    t = float(rng.uniform(700, 1400))  # inspection, the cube held still: the app learns where the camera is
    steps: list[Step] = []
    previous = "inspection"
    for k, (token, stage) in enumerate(tokens):
        move = parse_move(token)
        assert move is not None
        if stage != previous and previous != "inspection":
            t += float(rng.uniform(250, 1200))  # recognising the next case
        elif k and token[0] in "xyz" and tokens[k - 1][0] == invert_alg(token):
            t += float(rng.uniform(150, 700))  # a look at the side turned to, then the cube turned back
        elif k and tokens[k - 1][0][0] in "xyz" or token[0] in "xyz":
            t += float(rng.uniform(60, 200))
        elif stage == "f2l" and k and token[0] == "U" and tokens[k - 1][0][0] != "U" and rng.random() < 0.3:
            t += float(rng.uniform(150, 600))  # looking for the next pair
        rotation = move.family in ("x", "y", "z")
        quarter = float(rng.uniform(80, 150)) if not rotation else float(rng.uniform(180, 320))
        length = quarter * (1.5 if move.turns == 2 else 1.0)
        prev = steps[-1] if steps else None
        if cut and prev and not rotation and prev.token[0] not in "xyz" and t <= prev.end + 40 and rng.random() < cut:
            # Corner cutting: this layer starts while the last one ends, and ends after it
            span = prev.end - prev.start
            t = max(prev.start + 0.5 * span, prev.end - float(rng.uniform(0.15, 0.45)) * span, prev.end + 15 - length)
        steps.append(Step(token, t, t + length, stage))
        # Moves overlap little: the next starts after this one ends, at the solve's pace
        t += max(length, 1000.0 / tps * float(rng.uniform(0.7, 1.3)))
        previous = stage
    stages = {}
    for stage in ("cross", "f2l", "oll", "pll"):
        stages[stage] = max(s.end for s in steps if s.stage == stage)
    axes = [list(v / np.linalg.norm(v)) for v in rng.normal(size=(3, 3))]
    return Solve(scramble, steps, stages, tps, float(rng.uniform(10, 30)), float(rng.uniform(0.3, 1.2)), [[float(x) for x in a] for a in axes])


# ─── Frame by frame ──────────────────────────────────────────────────────────

def _ease(p: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, p)))


def _tilt(solve: Solve, t: float) -> list[float]:
    """The cube's wobble in the hands: three slow sinusoids about fixed axes, under ``tilt`` degrees in all,
    and the jerks under way."""
    q = [1.0, 0.0, 0.0, 0.0]
    for k, axis in enumerate(solve.tilt_axes):
        angle = solve.tilt / math.sqrt(3) * math.sin(2 * math.pi * solve.tilt_rate * (k + 1) * 0.37 * t / 1000 + k)
        q = _quaternion_multiply(_axis_angle(np.array(axis), angle), q)
    for start, length, axis, degrees in solve.jerks:
        if start <= t <= start + length:
            q = _quaternion_multiply(_axis_angle(np.array(axis), degrees * math.sin(math.pi * (t - start) / length)), q)
    return q


def agitated(rng: np.random.Generator, solve: Solve) -> Solve:
    """A cuber who never holds the cube still: a wider, quicker wobble, the cube turned this way and that
    during the inspection (looking at its sides), jerks now and then while solving (regrips, a look)."""
    jerks = []
    inspection = solve.steps[0].start
    t = 50.0
    while t < inspection - 400:
        length = float(rng.uniform(300, 900))
        jerks.append((t, length, [float(x) for x in rng.normal(size=3)], float(rng.uniform(20, 70))))
        t += length + float(rng.uniform(0, 300))
    t = inspection
    while t < solve.duration:
        t += float(rng.exponential(1500))
        jerks.append((t, float(rng.uniform(150, 500)), [float(x) for x in rng.normal(size=3)], float(rng.uniform(8, 35))))
    return dataclasses.replace(solve, tilt=float(rng.uniform(15, 40)), tilt_rate=float(rng.uniform(0.5, 2.0)), jerks=jerks)


@dataclass
class Pose:
    stickers: list[int]
    turn: Turn | None
    #: How the cube is held (Scenario.orientation)
    orientation: list[float]
    #: The next layer, already turning (corner cutting: Scenario.turn_next)
    turn_next: Turn | None = None


def poses(solve: Solve, times: list[float]) -> list[Pose]:
    """The cube at each time: its stickers, the layer being turned, and how it is held."""
    out = []
    cube = Cube().apply(solve.scramble)
    hold = np.eye(3)
    k = 0  # next step not yet finished
    for t in times:
        # Finish every step that ended before t
        while k < len(solve.steps) and solve.steps[k].end <= t:
            token = solve.steps[k].token
            if token[0] in "xyz":
                hold = turned_hold(token, hold)
            else:
                cube = cube.apply(in_cube_frame(token, hold))
            k += 1
        turn = None
        held = quaternion_of(hold)
        if k < len(solve.steps) and solve.steps[k].start <= t:
            step = solve.steps[k]
            p = _ease((t - step.start) / (step.end - step.start))
            if step.token[0] in "xyz":
                held = slerp(held, quaternion_of(turned_hold(step.token, hold)), p)
            else:
                body = in_cube_frame(step.token, hold)
                move = parse_move(body)
                assert move is not None
                turn = Turn(body, p * move.angle(), True)
        turn_next = None
        if turn and k + 1 < len(solve.steps) and solve.steps[k + 1].start <= t and solve.steps[k + 1].token[0] not in "xyz":
            step = solve.steps[k + 1]
            body = in_cube_frame(step.token, hold)
            move = parse_move(body)
            assert move is not None
            turn_next = Turn(body, _ease((t - step.start) / (step.end - step.start)) * move.angle(), True)
        out.append(Pose(cube.stickers.tolist(), turn, _quaternion_multiply(_tilt(solve, t), held), turn_next))
    return out


#: Who pushes each layer, as the cuber turns it (family, quarter turns 1 / 2 / 3 = '): hand side, finger
#: (-1 the whole hand turned at the wrist, 0 the thumb, 1 index, 3 ring), where it lies (target, across, reach)
PUSHERS = {
    "R": {1: (1, -1), 2: (1, -1), 3: (1, -1)}, "L": {1: (-1, -1), 2: (-1, -1), 3: (-1, -1)},
    "U": {1: (1, 1, "far", 1.0, 1.0), 2: (1, 1, "far", 1.0, 1.0), 3: (-1, 1, "far", 1.0, 1.0)},
    "D": {1: (-1, 3, "far", -1.0, 1.0), 2: (-1, 3, "far", -1.0, 1.0), 3: (1, 3, "far", -1.0, 1.0)},
    "F": {1: (1, 0, "near", 0.8, 1.0), 2: (1, 0, "near", 0.8, 1.0), 3: (-1, 0, "near", 0.8, 1.0)},
    "B": {1: (-1, 3, "far", 0.0, 0.8), 2: (-1, 3, "far", 0.0, 0.8), 3: (1, 3, "far", 0.0, 0.8)},
    "M": {1: (-1, 3, "far", 0.0, 1.5), 2: (-1, 3, "far", 0.0, 1.5), 3: (-1, 3, "far", 0.0, 1.5)},
}
#: A finger comes onto the layer this long before the turn, and leaves it this long after, ms
TRICK_IN, TRICK_OUT = 60.0, 80.0


def tricks_at(solve: Solve, t: float) -> list[dict]:
    """The fingers pushing the layers turning at ``t`` (Scenario.tricks), in the cuber's frame."""
    out = []
    for step in solve.steps:
        if step.token[0] in "xyz" or not (step.start - TRICK_IN <= t <= step.end + TRICK_OUT):
            continue
        move = parse_move(step.token)
        assert move is not None
        family = move.family.upper()
        if family not in PUSHERS:
            continue
        who = PUSHERS[family][move.turns]
        p = _ease((t - step.start) / (step.end - step.start))
        weight = min(1.0, (t - step.start + TRICK_IN) / TRICK_IN, 1.0 - (t - step.end) / TRICK_OUT)
        axis = [0.0, 0.0, 0.0]
        axis[move.axis] = 1.0
        trick = {"side": who[0], "finger": who[1], "axis": axis, "angle": p * move.angle(), "weight": max(0.0, weight)}
        if len(who) > 2:
            trick.update(target=who[2], across=who[3], reach=who[4])
        out.append(trick)
    return out


def _regrip(rng: np.random.Generator, hand: Hand) -> Hand:
    """The same hand a moment later: the palm slid a little, every finger moved along the cube."""
    def finger(f: Finger) -> Finger:
        return dataclasses.replace(f, across=float(np.clip(f.across + rng.normal(0, 0.25), -1.6, 1.6)),
                                   reach=float(np.clip(f.reach + rng.normal(0, 0.5), 0.2, 2.2)))
    return dataclasses.replace(hand, palm_depth=float(np.clip(hand.palm_depth + rng.normal(0, 0.2), -0.6, 0.9)),
                               palm_height=float(np.clip(hand.palm_height + rng.normal(0, 0.2), -1.2, 0.6)),
                               fingers=[finger(f) for f in hand.fingers], thumb=finger(hand.thumb))


def _blend(a: Hand, b: Hand, p: float) -> Hand:
    lerp = lambda x, y: x + (y - x) * p  # noqa: E731
    def finger(f: Finger, g: Finger) -> Finger:
        return dataclasses.replace(f, across=lerp(f.across, g.across), reach=lerp(f.reach, g.reach))
    return dataclasses.replace(a, palm_depth=lerp(a.palm_depth, b.palm_depth), palm_height=lerp(a.palm_height, b.palm_height),
                               fingers=[finger(f, g) for f, g in zip(a.fingers, b.fingers, strict=True)], thumb=finger(a.thumb, b.thumb))


def hands_over_time(rng: np.random.Generator, hands: list[Hand], duration: float) -> tuple[list[float], list[list[Hand]]]:
    """Grips every 300 to 700 ms: the times and the hands at each."""
    times, grips = [0.0], [hands]
    while times[-1] < duration:
        times.append(times[-1] + float(rng.uniform(300, 700)))
        grips.append([_regrip(rng, h) for h in grips[-1]])
    return times, grips


def hands_at(times: list[float], grips: list[list[Hand]], t: float) -> list[Hand]:
    k = max(0, min(len(times) - 2, int(np.searchsorted(times, t, side="right")) - 1))
    p = _ease((t - times[k]) / (times[k + 1] - times[k]))
    return [_blend(a, b, p) for a, b in zip(grips[k], grips[k + 1], strict=True)]


class CameraLook:
    """What a phone or a webcam does to the picture, drawn once per video, changing slowly over it: a hand
    holding the camera (sway and tremor), exposure and white balance hunting, focus hunting, sensor noise
    (more in the dark, coloured blotches), sharpening halos, vignetting."""

    def __init__(self, rng: np.random.Generator):
        u = lambda a, b: float(rng.uniform(a, b))  # noqa: E731
        self.rng = rng
        held = rng.random() < 0.6  # a phone in a hand, or on a stand
        self.sway = [(u(0.2, 1.5), u(0, 2 * math.pi), u(0, 0.012 if held else 0.002)) for _ in range(4)]
        self.tremor = (u(6, 11), u(0, 0.002) if held else 0.0)
        self.tilt = u(0, 0.6 if held else 0.1)
        self.exposure = (u(0.0, 0.12), u(0.05, 0.4), u(0, 2 * math.pi))
        self.white = np.array([u(0.92, 1.08), 1.0, u(0.92, 1.08)])
        self.white_drift = (u(0, 0.04), u(0.03, 0.2), u(0, 2 * math.pi))
        self.focus = u(0.0, 0.9)
        self.hunts = [(u(0, 30000), u(150, 500), u(1.0, 2.5)) for _ in range(int(rng.integers(0, 4)))]
        self.shot, self.read = u(0.0005, 0.006), u(1e-5, 2e-4)
        self.chroma = u(0.0, 0.025)
        self.sharpen = u(0.3, 1.2) if rng.random() < 0.5 else 0.0
        self.vignette = u(0.0, 0.35)
        self.gamma = u(0.9, 1.1)

    def __call__(self, image: np.ndarray, t: float, size: tuple[int, int] | None = None) -> np.ndarray:
        """The frame ``image`` at ``t`` ms as filmed, ``size`` (width, height) if given: a frame rendered
        smaller is scaled up in the same resampling as the hand's sway."""
        from PIL import Image, ImageChops, ImageFilter

        src = Image.fromarray(image)
        w, h = size or src.size
        kx, ky = src.width / w, src.height / h
        sec = t / 1000
        # The hand holding the camera: slow sway, a quick tremor, a slight roll; zoomed a little so no edge shows
        dx = sum(a * math.sin(2 * math.pi * f * sec + ph) for f, ph, a in self.sway[:2]) * w
        dy = sum(a * math.sin(2 * math.pi * f * sec + ph) for f, ph, a in self.sway[2:]) * w
        dx += self.tremor[1] * w * math.sin(2 * math.pi * self.tremor[0] * sec)
        roll = math.radians(self.tilt * math.sin(2 * math.pi * self.sway[0][0] * 0.7 * sec + 1))
        zoom = 1.0 + 2 * max(abs(dx) / w, abs(dy) / h) + abs(roll) * 0.6 + 0.01
        c, s_ = math.cos(roll) / zoom, math.sin(roll) / zoom
        cx, cy = w / 2, h / 2
        img = src.transform((w, h), Image.AFFINE, (c * kx, s_ * kx, (cx - c * cx - s_ * cy + dx) * kx,
                                                   -s_ * ky, c * ky, (cy + s_ * cx - c * cy + dy) * ky), resample=Image.BICUBIC)
        # Focus hunting now and then
        blur = self.focus + sum(amp * math.sin(math.pi * min(1.0, max(0.0, (t - at) / span))) for at, span, amp in self.hunts)
        if blur > 0.15:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        if self.sharpen:
            img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(100 * self.sharpen), threshold=2))
        # Exposure and white balance hunting, gamma: a table per channel; then vignetting
        amp, f, ph = self.exposure
        gain = 1 + amp * math.sin(2 * math.pi * f * sec + ph)
        amp, f, ph = self.white_drift
        white = self.white * (1 + amp * math.sin(2 * math.pi * f * sec + ph) * np.array([1, 0, -1]))
        v = np.arange(256) / 255
        lut = (np.clip(v[None] * (gain * white)[:, None], 0, 1) ** self.gamma * 255 + 0.5).astype(np.uint8)
        img = ImageChops.multiply(img.point(lut.ravel().tolist()), self._vignette(w, h))
        # Sensor noise: shot noise grows with the light, read noise in the dark (a table of sigmas); colour blotches
        level = np.asarray(img)
        sigma = (255 * np.sqrt(self.shot * v + self.read)).astype(np.float32)
        x = self.rng.standard_normal(level.shape, dtype=np.float32)
        x *= sigma[level]
        x += level
        if self.chroma:
            small = self.rng.normal(0, 255 * self.chroma, size=(2, h // 4 + 1, w // 4 + 1)).astype(np.float32)
            red, blue = (np.asarray(Image.fromarray(c_, mode="F").resize((w, h), Image.BILINEAR)) for c_ in small)
            x[..., 0] += red
            x[..., 1] -= (red + blue) / 2
            x[..., 2] += blue
        x += 0.5
        return np.clip(x, 0, 255, out=x).astype(np.uint8)

    def _vignette(self, w: int, h: int):
        """Darker towards the corners, one image per size."""
        from PIL import Image

        if getattr(self, "_vignetted", (None,))[0] != (w, h):
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
            shade = (255 * (1 - self.vignette * r2 / 2) + 0.5).clip(0, 255).astype(np.uint8)
            self._vignetted = ((w, h), Image.fromarray(np.repeat(shade[..., None], 3, axis=2)))
        return self._vignetted[1]


def truth(solve: Solve, name: str, fps: int, size: tuple[int, int], ext: str = "webm", mirrored: bool = False) -> dict:
    return {
        "video": f"{name}.{ext}",
        "scramble": solve.scramble,
        "solution": solve.written,
        "moves": [{"move": s.token, "start": round(s.start), "end": round(s.end), "stage": s.stage} for s in solve.steps],
        "stages": {k: round(v) for k, v in solve.stages.items()},
        "rotations": sum(s.token[0] in "xyz" for s in solve.steps),
        "tps": round(solve.tps, 2),
        "tilt_deg": round(solve.tilt, 1),
        "fps": fps,
        "size": list(size),
        **({"mirrored": True} if mirrored else {}),
    }


def _cuber(rng: np.random.Generator, sc: Scenario, real_hands: bool, photos: bool) -> Scenario:
    """A solve's scene, from an image's: both hands on the cube (an image may have none), and with photos
    most often a photo behind (rooms, people and their faces) rather than a plain wall."""
    sides = {h.side for h in sc.hands}
    skin = np.array(sc.hands[0].skin if sc.hands else sc.body["skin"] if sc.body else SKIN_TONES[rng.integers(len(SKIN_TONES))])
    added = [(_real_hand if real_hands else _hand)(rng, side, skin) for side in (-1, 1) if side not in sides]
    background = sc.background
    if photos and background["kind"] != "image" and background["image"] >= 0 and rng.random() < 0.85:
        background = {**background, "kind": "image"}
    hands = sorted([*sc.hands, *added], key=lambda h: h.side)
    if real_hands and hand_mesh.ASSET.exists():
        # The sculpted hand (hand_mesh), built from Blender Studio's Human Base Meshes
        hands = [dataclasses.replace(h, model="mesh") for h in hands]
    return dataclasses.replace(sc, hands=hands, background=background)


# Encoders as the app's recorder (Chrome's MediaRecorder: VP8, real time) or VP9
CODECS = {
    "vp8": ["-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8"],
    "vp9": ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1"],
    # A phone's video, an upload squeezed again
    "h264": ["-c:v", "libx264", "-preset", "veryfast"],
}


def render_video(index: int, seed: int, out: Path, *, fps: int, resolution: int, samples: int, device: str, bitrate: str,
                 backgrounds: list[Path] | None = None, distractors: bool = False, real_hands: bool = False, variety: bool = False,
                 cut: float = 0.0, tps: tuple[float, float] | None = None, look: bool = False, render_scale: float = 1.0,
                 codec: str = "vp9", finishers: list | None = None, wild: bool = False, portrait: bool = False,
                 mirrored: bool = False, drops: float = 0.0, overhead: bool = False, peeks: float = 0.0) -> dict:
    """One video: its scene, its solve, every frame rendered, then encoded. ``cut``: corner cutting (timed),
    ``tps``: the range of the solve's pace, ``look``: what a real camera does to the picture (CameraLook),
    ``render_scale``: frames rendered that much smaller, scaled up to ``resolution``. ``wild``: the scene as
    sample(wild=True), extra whole-cube rotations, the cube agitated, near or far; ``portrait``: a phone held
    upright; ``mirrored``: a selfie camera's mirrored picture; ``drops``: share of frames repeated (frames dropped);
    ``overhead``: filmed from above (a phone over the desk), the last layer seen mostly by its top, uniform in PLL;
    ``peeks``: looks at another side, the cube turned and turned back, before that share of the moves (peeked).

    The frames go to another process as they are rendered (the camera's look, the encoder): with
    ``finishers``, that process is added to it and left running, and writes the truth when the video is done."""
    from .scene import CubeScene

    rng = np.random.default_rng([seed, index, 7])
    tokens, scramble = plan(rng, wide=bool(rng.random() < 0.7))
    if wild:
        turner = np.random.default_rng([seed, index, 19])
        tokens = reheld(turner, tokens, float(turner.uniform(0.03, 0.1)))
    if peeks:
        tokens = peeked(np.random.default_rng([seed, index, 53]), tokens, peeks)
    solve = timed(rng, tokens, scramble, tps=float(rng.uniform(*tps)) if tps else None, cut=cut)
    if wild:
        solve = agitated(np.random.default_rng([seed, index, 29]), solve)
    base = sample(seed, index, framing="full", backgrounds=len(backgrounds or []), distractors=distractors, real_hands=real_hands, variety=variety,
                  wild=wild)
    # A whole frame of 16:9 (or upright), the cube well inside it, held by the cuber in front of the camera
    far, near = (0.05, 0.45) if wild else (0.08, 0.25)
    base = dataclasses.replace(base, aspect=9 / 16 if portrait else 16 / 9, cube_hidden=False, turn=None,
                               frame_fraction=float(rng.uniform(far, near)), offset=[float(x) for x in rng.uniform(-0.1, 0.1, size=2)],
                               shutter=float(rng.uniform(0.3, 0.8)))
    base = _cuber(np.random.default_rng([seed, index, 17]), base, real_hands, bool(backgrounds))
    if overhead:
        base = dataclasses.replace(base, elevation=float(np.random.default_rng([seed, index, 47]).uniform(60, 85)))
    name = f"synth-{seed}-{index:03d}"
    frames = out / f"{name}-frames"
    frames.mkdir(parents=True, exist_ok=True)
    times = [k * 1000.0 / fps for k in range(int(solve.duration * fps / 1000) + 1)]
    grip_times, grips = hands_over_time(rng, base.hands, solve.duration)
    # ``resolution`` is the longer side (an upright phone's height)
    size = (resolution, max(2, round(resolution / base.aspect / 2) * 2)) if base.aspect >= 1 else (max(2, round(resolution * base.aspect / 2) * 2), resolution)
    record = truth(solve, name, fps, size, "mp4" if codec == "h264" else "webm", mirrored)
    if overhead:
        record["overhead"] = True
    spec = {"frames": str(frames), "video": str(out / record["video"]), "truth": str(out / f"{name}.truth.json"), "record": record,
            "fps": fps, "size": size, "codec": codec, "bitrate": bitrate, "look": [seed, index, 13] if look else None,
            "mirrored": mirrored, "drops": [seed, index, 37, drops] if drops else None}
    finisher = subprocess.Popen([sys.executable, "-m", "cubevision.render.solve_video", "--finish", json.dumps(spec)],
                                stdin=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])})
    if real_hands and hand_mesh.ASSET.exists():
        # The sculpted hands' grips, every frame's, solved ahead on the CPUs left
        hand_mesh.prime([(h, base.index * 2 + i) for t in times for i, h in enumerate(hands_at(grip_times, grips, t))],
                        max(1, (os.cpu_count() or 2) // 2 - 1))
    scene = CubeScene(max(2, round(resolution * render_scale)), samples, device, backgrounds)
    # The quick denoiser's prefilter: 40 ms a frame less at 720p, under the camera's own noise
    scene.scene.cycles.denoising_prefilter = "FAST"
    for k, pose in enumerate(poses(solve, times)):
        # Motion blur: how far the cube and the layer move during one frame
        nxt = poses(solve, [times[k] + 1000.0 / fps])[0]
        spin_axis, spin = _between(pose.orientation, nxt.orientation)
        speed = abs(nxt.turn.angle - pose.turn.angle) if pose.turn and nxt.turn and pose.turn.move == nxt.turn.move else 0.0
        sc = dataclasses.replace(base, stickers=pose.stickers, turn=pose.turn, orientation=pose.orientation,
                                 spin=[*spin_axis, spin], turn_speed=speed, hands=hands_at(grip_times, grips, times[k]),
                                 turn_next=pose.turn_next, tricks=tricks_at(solve, times[k]) if real_hands else [])
        path = frames / f"{k:05d}.jpg"
        scene.render(sc, path, measure=False)
        finisher.stdin.write(f"{path}\t{times[k]}\n")
        finisher.stdin.flush()
    finisher.stdin.close()
    if finishers is None:
        _wait(finisher)
    else:
        finishers.append(finisher)
    return record


def _wait(finisher: subprocess.Popen) -> None:
    if finisher.wait():
        raise RuntimeError(f"finishing a video failed ({finisher.returncode})")


def finish(spec: dict, lines) -> None:
    """Each frame rendered (``lines``: its path and time), as filmed (CameraLook), piped to the encoder;
    then the truth, the video being whole."""
    from PIL import Image

    w, h = spec["size"]
    camera = CameraLook(np.random.default_rng(spec["look"])) if spec["look"] else None
    drops = np.random.default_rng(spec["drops"][:3]) if spec.get("drops") else None
    last = None
    encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                "-framerate", str(spec["fps"]), "-i", "-", *CODECS[spec["codec"]], "-b:v", spec["bitrate"],
                                "-pix_fmt", "yuv420p", spec["video"]], stdin=subprocess.PIPE)
    for line in lines:
        path, t = line.rstrip("\n").split("\t")
        image = Image.open(path).convert("RGB")
        Path(path).unlink()
        if camera:
            frame = camera(np.asarray(image), float(t), (w, h))
        else:
            frame = np.asarray(image if image.size == (w, h) else image.resize((w, h), Image.BICUBIC))
        if spec.get("mirrored"):
            frame = frame[:, ::-1]
        # A phone that drops a frame now and then repeats the last one
        if drops is not None and last is not None and drops.random() < spec["drops"][3]:
            frame = last
        last = frame
        encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError(f"ffmpeg failed on {spec['video']}")
    shutil.rmtree(spec["frames"], ignore_errors=True)
    Path(spec["truth"]).write_text(json.dumps(spec["record"], indent=1))


def _between(a: list[float], b: list[float]) -> tuple[list[float], float]:
    """Axis (cuber's frame) and angle (degrees) of the rotation from orientation a to b."""
    inv = [a[0], -a[1], -a[2], -a[3]]
    d = _quaternion_multiply(b, inv)
    if d[0] < 0:
        d = [-v for v in d]
    angle = 2 * math.degrees(math.acos(min(1.0, d[0])))
    n = math.sqrt(d[1] ** 2 + d[2] ** 2 + d[3] ** 2)
    return ([d[1] / n, d[2] / n, d[3] / n] if n > 1e-9 else [0.0, 0.0, 1.0]), angle


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--finish"]:
        finish(json.loads(argv[1]), sys.stdin)
        return 0
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("evals/synthetic"))
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--bitrate", default="2500k", help="VP9 bitrate, about a browser's MediaRecorder at 720p")
    p.add_argument("--fps", type=int, help="every video at this frame rate (default: 30, a quarter at 60)")
    p.add_argument("--resolution", type=int, help="every video at this width (default: 1280, 30 %% at 640)")
    p.add_argument("--backgrounds", type=Path, help="folder of photos used as backdrops (jpg/png)")
    p.add_argument("--distractors", action="store_true", help="look-alikes and things of a room around the cube")
    p.add_argument("--real-hands", action="store_true", help="hands as filmed: round phalanges, knuckles, nails")
    p.add_argument("--variety", action="store_true", help="faded and worn cubes, backlight, dim rooms, hard spot lights")
    p.add_argument("--corner-cutting", type=float, default=0.0, help="share of the turns following another that start before it ends")
    p.add_argument("--tps", type=float, nargs=2, help="range of the solves' pace, turns per second (default 3 to 8)")
    p.add_argument("--camera-look", action="store_true", help="a real camera: hand sway, exposure and focus hunting, sensor noise")
    p.add_argument("--bitrates", nargs="+", help="bitrates drawn from per video (default --bitrate)")
    p.add_argument("--codecs", nargs="+", choices=sorted(CODECS), default=["vp9"],
                   help="encoders drawn from per video (vp8: the app's recorder in Chrome)")
    p.add_argument("--render-scale", type=float, default=1.0,
                   help="frames rendered that much smaller, scaled up (0.75: half the time, the cube's crop barely changes)")
    p.add_argument("--wild", action="store_true",
                   help="footage as it comes: the wild scenes, extra x y z rotations, the cube agitated, near or far, 480p to 1080p, "
                        "upright phones, mirrored selfie cameras, dropped frames, H.264")
    p.add_argument("--overhead", type=float, default=0.0,
                   help="share of the videos filmed from above (the last layer seen by its top, uniform in PLL)")
    p.add_argument("--peeks", type=float, default=0.0,
                   help="share of the moves before which the cuber turns the cube to look at another side, then turns it back")
    p.add_argument("--workers", type=int, default=1,
                   help="Blender processes sharing the GPU, one video each (2: 1.35 times the frames, Cycles' scene updates being on the CPU)")
    p.add_argument("--shard", type=int, nargs=2, help=argparse.SUPPRESS)  # a worker's share: the videos k of N
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    backgrounds = sorted(q for q in args.backgrounds.rglob("*") if q.suffix.lower() in (".jpg", ".jpeg", ".png")) if args.backgrounds else []
    indexes = range(args.start, args.start + args.count)
    if args.workers > 1:
        # The same command per worker, each its share of the videos; the list once all are done
        workers = [subprocess.Popen([sys.executable, "-m", "cubevision.render.solve_video", *argv, "--workers", "1", "--shard", str(w), str(args.workers)])
                   for w in range(args.workers)]
        if any(w.wait() for w in workers):
            raise RuntimeError("a worker failed")
    cases, finishers = [], []
    for index in indexes:
        name = f"synth-{args.seed}-{index:03d}"
        truth_path = args.out / f"{name}.truth.json"
        if args.shard and (index - args.start) % args.shard[1] != args.shard[0]:
            continue
        if truth_path.exists():
            record = json.loads(truth_path.read_text())
        else:
            # Most at 30 fps in 720p, some at 60 fps or 360p (a phone, a webcam)
            rng = np.random.default_rng([args.seed, index, 11])
            fps = args.fps or int(rng.choice([30, 60], p=[0.75, 0.25] if not args.wild else [0.6, 0.4]))
            resolution = args.resolution or int(rng.choice([1280, 640], p=[0.7, 0.3]) if not args.wild else rng.choice([854, 1280, 1920], p=[0.3, 0.45, 0.25]))
            t0 = time.time()
            bitrate = str(rng.choice(args.bitrates)) if args.bitrates else args.bitrate
            codec = str(rng.choice(args.codecs))
            # Drawn after the rest: the plain videos stay as they were
            wild = np.random.default_rng([args.seed, index, 43])
            portrait, mirrored = (bool(wild.random() < 0.2), bool(wild.random() < 0.2)) if args.wild else (False, False)
            drops = float(wild.uniform(0, 0.05)) if args.wild and wild.random() < 0.4 else 0.0
            overhead = bool(wild.random() < args.overhead)
            # The last videos' frames are still being finished while this one renders: two at most
            while len(finishers) >= 2:
                _wait(finishers.pop(0))
            record = render_video(index, args.seed, args.out, fps=fps, resolution=resolution, samples=args.samples,
                                  device=args.device, bitrate=bitrate, backgrounds=backgrounds, distractors=args.distractors,
                                  real_hands=args.real_hands, variety=args.variety, cut=args.corner_cutting,
                                  tps=tuple(args.tps) if args.tps else None, look=args.camera_look,
                                  render_scale=args.render_scale, codec=codec, finishers=finishers, wild=args.wild,
                                  portrait=portrait, mirrored=mirrored, drops=drops, overhead=overhead, peeks=args.peeks)
            print(f"[solve_video] {name}: {len(record['moves'])} tokens, {record['moves'][-1]['end'] / 1000:.1f} s, "
                  f"{fps} fps, {record['size'][0]}x{record['size'][1]}, {codec} {bitrate}, rendered in {time.time() - t0:.0f} s", flush=True)
        cases.append({"video": record["video"], "scramble": record["scramble"], "truth": f"{name}.truth.json"})
    for finisher in finishers:
        _wait(finisher)
    if args.shard:
        return 0
    (args.out / "synthetic.json").write_text(json.dumps(cases, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
