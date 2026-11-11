//! The stickers of a 3×3 cube (`stickerCube.ts`, for n = 3)
//!
//! Axes: x towards R, y towards U, z towards F. Coordinates are doubled so that every value is an
//! integer: cubie centres sit on -2, 0, 2 and stickers lie on the planes ±3.
//!
//! Faces are stored U R F D L B, each row by row as seen from the outside: U with B at the top, D with
//! F at the top, the side faces with U at the top. The colour of a sticker is the face it started on.

use std::sync::OnceLock;

use crate::notation::{Family, Move, parse_alg};

/// Faces as stored: U R F D L B
pub const FACES: [char; 6] = ['U', 'R', 'F', 'D', 'L', 'B'];

const N: i32 = 3;

type Vec3 = [i32; 3];

/// Position of sticker (row, col) of a face
fn sticker_position(face: usize, row: usize, col: usize) -> Vec3 {
    let g = |i: usize| -(N - 1) + 2 * i as i32;
    match face {
        0 => [g(col), N, g(row)],
        1 => [N, -g(row), -g(col)],
        2 => [g(col), -g(row), N],
        3 => [g(col), -N, -g(row)],
        4 => [-N, -g(row), g(col)],
        _ => [-g(col), -g(row), -N],
    }
}

/// Centre of the cubie a sticker belongs to
fn cubie_of(v: Vec3) -> Vec3 {
    v.map(|c| if c.abs() == N { c.signum() * (N - 1) } else { c })
}

/// A quarter turn (positive, right-handed) around an axis
fn rotate(axis: usize, [x, y, z]: Vec3) -> Vec3 {
    match axis {
        0 => [x, -z, y],
        1 => [z, y, -x],
        _ => [-y, x, z],
    }
}

struct Geometry {
    positions: [Vec3; 54],
    /// Sticker index at each position, by coordinates + 3 (7 × 7 × 7)
    index: [u8; 343],
}

fn slot([x, y, z]: Vec3) -> usize {
    (((x + N) * 7 + (y + N)) * 7 + (z + N)) as usize
}

fn geometry() -> &'static Geometry {
    static GEO: OnceLock<Geometry> = OnceLock::new();
    GEO.get_or_init(|| {
        let mut positions = [[0; 3]; 54];
        let mut index = [u8::MAX; 343];
        for (i, p) in positions.iter_mut().enumerate() {
            *p = sticker_position(i / 9, i / 3 % 3, i % 3);
            index[slot(*p)] = i as u8;
        }
        Geometry { positions, index }
    })
}

/// Axis, side (+1: R U F, -1: L D B) and the layers turned: the move's own (None), the inner one, or all
fn family(f: Family) -> (usize, i32, Option<(u32, u32)>) {
    use Family::*;
    match f {
        R => (0, 1, None),
        L => (0, -1, None),
        U => (1, 1, None),
        D => (1, -1, None),
        F => (2, 1, None),
        B => (2, -1, None),
        M => (0, -1, Some((2, 2))),
        E => (1, -1, Some((2, 2))),
        S => (2, 1, Some((2, 2))),
        X => (0, 1, Some((1, 3))),
        Y => (1, 1, Some((1, 3))),
        Z => (2, 1, Some((1, 3))),
    }
}

/// Where each sticker goes: next[perm[i]] = current[i]
fn permutation(m: &Move) -> [u8; 54] {
    let geo = geometry();
    let (axis, side, layers) = family(m.family);
    let (from, to) = layers.unwrap_or((m.from, m.to));
    // Layer k (counted from the face) holds the cubies at side * (n + 1 - 2k); layers past the cube hold none
    let selected = |c: i32| (from.max(1)..=to.min(N as u32)).any(|k| side * (N + 1 - 2 * k as i32) == c);
    // Clockwise seen from the face: negative rotation on the + side
    let quarter = if side > 0 { (4 - m.turns) % 4 } else { m.turns };
    let mut perm = [0u8; 54];
    for (i, &p) in geo.positions.iter().enumerate() {
        perm[i] = if selected(cubie_of(p)[axis]) {
            let mut v = p;
            for _ in 0..quarter {
                v = rotate(axis, v);
            }
            geo.index[slot(v)]
        } else {
            i as u8
        };
    }
    perm
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Cube {
    /// stickers[position] = the sticker that sits there (its start position)
    stickers: [u8; 54],
}

impl Cube {
    pub fn solved() -> Cube {
        Cube { stickers: std::array::from_fn(|i| i as u8) }
    }

    /// The cube with these stickers at each place (start positions 0..54)
    pub fn from_stickers(stickers: [u8; 54]) -> Cube {
        Cube { stickers }
    }

    /// A new cube with the moves applied (tokens that are not moves are ignored)
    pub fn apply(&self, alg: &str) -> Cube {
        self.apply_moves(&parse_alg(alg).moves)
    }

    pub fn apply_moves(&self, moves: &[Move]) -> Cube {
        let mut current = self.stickers;
        for m in moves {
            let perm = permutation(m);
            let mut next = [0u8; 54];
            for i in 0..54 {
                next[perm[i] as usize] = current[i];
            }
            current = next;
        }
        Cube { stickers: current }
    }

    pub fn stickers(&self) -> &[u8; 54] {
        &self.stickers
    }

    /// Colour (start face, U R F D L B = 0..6) of the sticker at (face, row, col)
    pub fn color(&self, face: usize, row: usize, col: usize) -> u8 {
        self.stickers[(face * 3 + row) * 3 + col] / 9
    }

    /// Colour of every sticker, U R F D L B, nine each
    pub fn colors(&self) -> [u8; 54] {
        self.stickers.map(|s| s / 9)
    }

    /// Colours of every sticker on the cubie that holds (face, row, col), in sticker order
    pub fn cubie_colors(&self, face: usize, row: usize, col: usize) -> Vec<u8> {
        let geo = geometry();
        let cubie = cubie_of(sticker_position(face, row, col));
        (0..54).filter(|&i| cubie_of(geo.positions[i]) == cubie).map(|i| self.stickers[i] / 9).collect()
    }
}
