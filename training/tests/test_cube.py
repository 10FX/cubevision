"""The Python cube must turn exactly like the app's StickerCube."""
import json
from pathlib import Path

import numpy as np
import pytest

from cubevision.cube import FACE_MOVES, ROT, Cube, invert_alg, parse_move

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "stickercube.json").read_text())


@pytest.mark.parametrize("state", FIXTURE["states"], ids=lambda s: s["alg"])
def test_matches_stickercube(state):
    assert Cube().apply(state["alg"]).stickers.tolist() == state["stickers"]


def test_rotation_table_matches_search():
    assert ROT == FIXTURE["rotations"]


def test_inverse_undoes_a_scramble():
    scramble = "D2 F2 U' B2 U L2 U2 F2 D' R2 U' R' D2 U' F' R2 D B' U2 R'"
    assert Cube().apply(scramble).apply(invert_alg(scramble)).is_solved()
    assert not Cube().apply(scramble).is_solved()


@pytest.mark.parametrize("move", FACE_MOVES)
def test_four_quarter_turns_are_identity(move):
    m = parse_move(move)
    quarter = parse_move(move[0])
    assert Cube().apply([quarter] * 4).is_solved()
    assert m is not None


@pytest.mark.parametrize("move,angle", [("R", -90), ("R'", 90), ("R2", -180), ("L", 90), ("U", -90), ("D", 90), ("F", -90), ("B", 90), ("M", 90), ("x", -90)])
def test_animation_angle_matches_turn(move, angle):
    """Rotating the layer's sticker positions by the angle gives the permutation of the move."""
    m = parse_move(move)
    assert m.angle() == angle
    from cubevision.cube import POSITIONS, cubie_of

    a = np.radians(m.angle())
    c, s = np.cos(a), np.sin(a)
    axis = m.axis
    turned = Cube().apply(move).stickers
    for i, p in enumerate(POSITIONS):
        if cubie_of(p)[axis] not in m.layers():
            continue
        v = np.array(p, dtype=float)
        i1, i2 = [k for k in range(3) if k != axis]
        # Right-handed rotation about the axis (x: y→z, y: z→x, z: x→y)
        if axis == 1:
            i1, i2 = 2, 0
        w = v.copy()
        w[i1] = c * v[i1] - s * v[i2]
        w[i2] = s * v[i1] + c * v[i2]
        target = tuple(int(round(x)) for x in w)
        j = POSITIONS.index(target)
        # The sticker that was at i is now at j
        assert turned[j] == i
