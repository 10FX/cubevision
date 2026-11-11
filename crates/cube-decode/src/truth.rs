//! Moves brought to one form (ChronoCube's truth.ts): quarter turns of the outer faces, named by the cube's own faces
//! (the scramble's), in a canonical order. An M2 found as R2 L2 x2, a Rw as L x, or L R as R L, are then the same turns.

use cube_core::{Cube, FACES, Family, parse_move};

/// A quarter turn of a face of the cube (clockwise: +1), and when it was done
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quarter {
    pub face: char,
    pub dir: i8,
    pub t: f64,
}

/// Faces in the order turns on one axis are written
const ORDER: [char; 6] = ['R', 'L', 'U', 'D', 'F', 'B'];

fn axis(face: char) -> u8 {
    match face {
        'R' | 'L' => 0,
        'U' | 'D' => 1,
        _ => 2,
    }
}

/// Outer face turns and rotations, one quarter at a time, that a token amounts to (in its own frame)
fn expand(token: &str) -> Vec<&'static str> {
    use Family::*;
    let Some(m) = parse_move(token) else { return vec![] };
    let once: &[&'static str] = match m.family {
        // A slice: the two faces of its axis, the whole cube the other way
        M => &["R", "L'", "x'"],
        E => &["U", "D'", "y'"],
        S => &["F'", "B", "z"],
        X => &["x"],
        Y => &["y"],
        Z => &["z"],
        // A wide move turns the opposite face the same way, with the whole cube
        family if m.from == 1 && m.to == 2 => match family {
            R => &["L", "x"],
            L => &["R", "x'"],
            U => &["D", "y"],
            D => &["U", "y'"],
            F => &["B", "z"],
            _ => &["F", "z'"],
        },
        U => &["U"],
        D => &["D"],
        F => &["F"],
        B => &["B"],
        R => &["R"],
        L => &["L"],
    };
    (0..m.turns).flat_map(|_| once.iter().copied()).collect()
}

/// Moves written in a frame (the cuber's, rotations included; or the search's, held as `hold`), as quarter turns of
/// the cube's own faces, in canonical order
pub fn quarters(moves: &[(&str, f64)], hold: &str) -> Vec<Quarter> {
    let mut frame = Cube::solved().apply(hold);
    let mut turns: Vec<(char, u32, f64)> = Vec::new();
    for &(token, t) in moves {
        for part in expand(token) {
            let family = part.chars().next().unwrap();
            if matches!(family, 'x' | 'y' | 'z') {
                frame = frame.apply(part);
                continue;
            }
            // The face of the cube that sits where the frame's face is: the colour of the centre there
            let at = FACES.iter().position(|&f| f == family).unwrap();
            let face = FACES[frame.color(at, 1, 1) as usize];
            turns.push((face, if part.ends_with('\'') { 3 } else { 1 }, t));
        }
    }
    // Turns in a row on one axis commute: their net turn per face, faces in a fixed order
    let mut out = Vec::new();
    let mut i = 0;
    while i < turns.len() {
        let mut j = i;
        while j < turns.len() && axis(turns[j].0) == axis(turns[i].0) {
            j += 1;
        }
        for face in ORDER {
            let group: Vec<_> = turns[i..j].iter().filter(|q| q.0 == face).collect();
            let net = group.iter().map(|q| q.1).sum::<u32>() % 4;
            let t = group.iter().fold(f64::NEG_INFINITY, |a, q| a.max(q.2));
            if net == 1 || net == 2 {
                out.extend((0..net).map(|_| Quarter { face, dir: 1, t }));
            }
            if net == 3 {
                out.push(Quarter { face, dir: -1, t });
            }
        }
        i = j;
    }
    out
}
