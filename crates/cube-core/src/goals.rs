//! The 24 ways to hold a cube, and how far a CFOP solve went (`goals.ts`)

use crate::cube::Cube;

/// The 24 ways to hold a cube: a rotation, then y turns
pub const ORIENTATIONS: [&str; 24] = [
    "", "x", "x2", "x'", "z", "z'",
    "y", "x y", "x2 y", "x' y", "z y", "z' y",
    "y2", "x y2", "x2 y2", "x' y2", "z y2", "z' y2",
    "y'", "x y'", "x2 y'", "x' y'", "z y'", "z' y'",
];

const U: usize = 0;
const D: usize = 3;
/// F R B L
const SIDES: [usize; 4] = [2, 1, 5, 4];

/// The cells of a face chosen by `cells` are its colour: the start face's, or with `neutral` its centre's
fn face_is(cube: &Cube, face: usize, cells: impl Fn(usize, usize) -> bool, neutral: bool) -> bool {
    let colour = if neutral { cube.color(face, 1, 1) } else { face as u8 };
    (0..3).all(|row| (0..3).all(|col| !cells(row, col) || cube.color(face, row, col) == colour))
}

fn is_edge(row: usize, col: usize) -> bool {
    (row == 0 || row == 2) != (col == 0 || col == 2)
}

fn is_corner(row: usize, col: usize) -> bool {
    (row == 0 || row == 2) && (col == 0 || col == 2)
}

/// The D cross with its side stickers
pub fn cross(cube: &Cube, neutral: bool) -> bool {
    face_is(cube, D, |r, c| is_edge(r, c) || !is_corner(r, c), neutral)
        && SIDES.iter().all(|&f| face_is(cube, f, |row, col| row == 2 && col > 0 && col < 2, neutral))
}

/// Slots between consecutive side faces, solved (D corner + both side columns)
pub fn solved_slots(cube: &Cube, neutral: bool) -> u32 {
    (0..4)
        .filter(|&i| {
            let (left, right) = (SIDES[i], SIDES[(i + 1) % 4]);
            face_is(cube, left, |row, col| row > 0 && col == 2, neutral) && face_is(cube, right, |row, col| row > 0 && col == 0, neutral)
        })
        .count() as u32
}

/// How far a CFOP solve went, held in its best orientation: 0 nothing, 1 cross, 2–5 cross + 1–4 F2L
/// pairs, 6 last layer oriented, 7 solved. The cross may be of any colour.
pub fn cfop_progress(cube: &Cube) -> u32 {
    let mut best = 0;
    for orientation in ORIENTATIONS {
        let held = cube.apply(orientation);
        if !cross(&held, true) {
            continue;
        }
        let slots = solved_slots(&held, true);
        let mut progress = 1 + slots;
        if slots == 4 && face_is(&held, U, |_, _| true, true) {
            progress = if (0..6).all(|f| face_is(&held, f, |_, _| true, true)) { 7 } else { 6 };
        }
        best = best.max(progress);
        if best == 7 {
            break;
        }
    }
    best
}
