"""
Sticker model of the 3x3x3 cube, identical to src/lib/cube/stickerCube.ts.

Coordinates are doubled so that every value is an integer: x towards R, y
towards U, z towards F; cubie centres at -2, 0, 2 and stickers on the planes
±3. Faces are stored U R F D L B, each row by row as seen from the outside
(U with B at the top, D with F at the top, side faces with U at the top).
``stickers[position]`` is the sticker that sits at ``position`` (its start
position), so the colour at a position is ``stickers[position] // 9``.

Parity with the TypeScript model is checked by ml2/tests/test_cube.py against
a fixture generated from the app.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

import numpy as np

N = 3
FACES = ("U", "R", "F", "D", "L", "B")
FACE = {name: i for i, name in enumerate(FACES)}

# Axis (0 x, 1 y, 2 z), side (+1 R U F, -1 L D B) and the layers a family turns
_FAMILY = {
    "R": (0, 1, "move"), "L": (0, -1, "move"),
    "U": (1, 1, "move"), "D": (1, -1, "move"),
    "F": (2, 1, "move"), "B": (2, -1, "move"),
    "M": (0, -1, "inner"), "E": (1, -1, "inner"), "S": (2, 1, "inner"),
    "x": (0, 1, "all"), "y": (1, 1, "all"), "z": (2, 1, "all"),
}

_TOKEN = re.compile(r"^(?:(\d+)(?:-(\d+))?)?([UDFBRL]w?|[udfbrl]|[MESxyz])(\d*)('?)$")


def _sticker_position(face: int, row: int, col: int) -> tuple[int, int, int]:
    g = lambda i: -(N - 1) + 2 * i  # noqa: E731
    if face == 0:
        return (g(col), N, g(row))
    if face == 1:
        return (N, -g(row), -g(col))
    if face == 2:
        return (g(col), -g(row), N)
    if face == 3:
        return (g(col), -N, -g(row))
    if face == 4:
        return (-N, -g(row), g(col))
    return (-g(col), -g(row), -N)


def cubie_of(p: tuple[int, int, int]) -> tuple[int, int, int]:
    """Centre of the cubie a sticker position belongs to."""
    return tuple(int(np.sign(c)) * (N - 1) if abs(c) == N else c for c in p)  # type: ignore[return-value]


POSITIONS: tuple[tuple[int, int, int], ...] = tuple(
    _sticker_position(f, r, c) for f in range(6) for r in range(N) for c in range(N)
)
_INDEX = {p: i for i, p in enumerate(POSITIONS)}

# A quarter turn (positive, right-handed) around an axis
_ROTATE = {
    0: lambda v: (v[0], -v[2], v[1]),
    1: lambda v: (v[2], v[1], -v[0]),
    2: lambda v: (-v[1], v[0], v[2]),
}


@dataclass(frozen=True)
class Move:
    family: str
    #: Layers turned, counted from the family's face (1 = outer). 0 for M/E/S and rotations.
    first: int
    last: int
    #: Quarter turns, clockwise seen from the face: 1, 2 or 3 (= prime)
    turns: int

    @property
    def axis(self) -> int:
        return _FAMILY[self.family][0]

    def layers(self) -> set[int]:
        """Cubie coordinates along the axis that the move turns."""
        _, side, kind = _FAMILY[self.family]
        first, last = (self.first, self.last) if kind == "move" else (2, N - 1) if kind == "inner" else (1, N)
        return {side * (N + 1 - 2 * k) for k in range(first, last + 1)}

    def angle(self) -> float:
        """Signed angle in degrees (right-handed about +axis) of the shortest way to make the move."""
        _, side, _ = _FAMILY[self.family]
        signed = {1: 1, 2: 2, 3: -1}[self.turns]
        return -side * signed * 90.0


def parse_move(token: str) -> Move | None:
    """One move token ("R", "U2'", "Rw", "r", "M'", "y2"), or None."""
    text = re.sub(r"[’‘`´]", "'", token)
    text = re.sub(r"^(.+?)'(\d+)$", r"\1\2'", text)
    m = _TOKEN.match(text)
    if not m:
        return None
    a, b, letters, amount_text, prime = m.groups()
    amount = int(amount_text) if amount_text else 1
    turns = ((-amount if prime else amount) % 4 + 4) % 4
    if turns == 0:
        return None
    first = int(a) if a else None
    second = int(b) if b else None
    if letters in ("M", "E", "S", "x", "y", "z"):
        return None if a else Move(letters, 0, 0, turns)
    family = letters[0].upper()
    if letters.endswith("w") or letters.islower():
        if second is not None:
            return Move(family, first, second, turns)  # type: ignore[arg-type]
        return Move(family, 1, first or 2, turns)
    if second is not None:
        return Move(family, first, second, turns)  # type: ignore[arg-type]
    if first is not None:
        return Move(family, first, first, turns)
    return Move(family, 1, 1, turns)


def parse_alg(alg: str) -> list[Move]:
    moves = []
    for token in alg.replace("(", " ").replace(")", " ").split():
        move = parse_move(token)
        if move is None:
            raise ValueError(f"not a move: {token!r}")
        moves.append(move)
    return moves


@cache
def _permutation(move: Move) -> np.ndarray:
    axis = move.axis
    selected = move.layers()
    _, side, _ = _FAMILY[move.family]
    quarter = (4 - move.turns) % 4 if side > 0 else move.turns
    perm = np.empty(len(POSITIONS), dtype=np.int64)
    for i, p in enumerate(POSITIONS):
        if cubie_of(p)[axis] not in selected:
            perm[i] = i
            continue
        v = p
        for _ in range(quarter):
            v = _ROTATE[axis](v)
        perm[i] = _INDEX[v]
    return perm


class Cube:
    """Immutable sticker cube (see the module docstring)."""

    __slots__ = ("stickers",)

    def __init__(self, stickers: np.ndarray | None = None):
        self.stickers = np.arange(54, dtype=np.int64) if stickers is None else np.asarray(stickers, dtype=np.int64)

    def apply(self, alg: str | list[Move] | Move) -> Cube:
        moves = [alg] if isinstance(alg, Move) else parse_alg(alg) if isinstance(alg, str) else alg
        current = self.stickers
        for move in moves:
            perm = _permutation(move)
            nxt = np.empty_like(current)
            nxt[perm] = current
            current = nxt
        return Cube(current)

    def colors(self) -> np.ndarray:
        """Colour (start face, 0..5) at each of the 54 positions."""
        return self.stickers // 9

    def is_solved(self) -> bool:
        c = self.colors().reshape(6, 9)
        return bool((c == c[:, 4:5]).all())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cube) and bool((self.stickers == other.stickers).all())

    def __hash__(self) -> int:
        return hash(self.stickers.tobytes())


def invert_alg(alg: str) -> str:
    """A B C → C' B' A' (written with the face letters; enough for scrambles and holds)."""
    out = []
    for token in reversed(alg.split()):
        move = parse_move(token)
        if move is None:
            raise ValueError(token)
        out.append(_format(move, (4 - move.turns) % 4))
    return " ".join(out)


def _format(move: Move, turns: int) -> str:
    suffix = {1: "", 2: "2", 3: "'"}[turns]
    if move.family in ("M", "E", "S", "x", "y", "z"):
        return move.family + suffix
    if move.first == 1 and move.last == 1:
        return move.family + suffix
    if move.first == 1:
        return f"{'' if move.last == 2 else move.last}{move.family}w{suffix}"
    if move.first == move.last:
        return f"{move.first}{move.family}{suffix}"
    return f"{move.first}-{move.last}{move.family}{suffix}"


#: A face read in rotation r: cell i of the reading is sticker ROT[r][i] of the face
#: (same table as src/lib/vision/search.ts)
def _rot_table() -> list[list[int]]:
    table = []
    for r in range(4):
        cells = []
        for i in range(9):
            row, col = divmod(i, 3)
            for _ in range(r):
                row, col = 2 - col, row
            cells.append(row * 3 + col)
        table.append(cells)
    return table


ROT: list[list[int]] = _rot_table()

FACE_MOVES = tuple(f + s for f in FACES for s in ("", "2", "'"))
