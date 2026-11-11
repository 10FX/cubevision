//! A video reconstruction scored against the truth of a synthetic solve video (ChronoCube's truth.ts): moves exact,
//! timing, CFOP stages and the cuber's rotations.
//!
//! The moves found and the moves played are first brought to one form: quarter turns of the outer faces, named by the
//! cube's own faces (the scramble's), in a canonical order. An M2 found as R2 L2 x2, a Rw as L x, or L R as R L, are
//! then the same turns, and the distance between the two lists (Levenshtein, in quarter turns) counts only what was
//! really missed, added or mistaken.

use cube_core::{Cube, FACES, Family, js_space, parse_move};

use crate::decode::js_round;
use crate::pipeline::VisionResult;
use crate::reconstruct::Phase;

/// The stages a synthetic video tells the end of
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Cross,
    F2l,
    Oll,
    Pll,
}

impl Stage {
    pub const ALL: [Stage; 4] = [Stage::Cross, Stage::F2l, Stage::Oll, Stage::Pll];

    /// The phase of a reconstruction that ends the stage (the last pair for F2L)
    fn phase(self) -> Phase {
        match self {
            Stage::Cross => Phase::Cross,
            Stage::F2l => Phase::F2l4,
            Stage::Oll => Phase::Oll,
            Stage::Pll => Phase::Pll,
        }
    }
}

/// truth.json of a synthetic video
#[derive(Clone, Debug, PartialEq)]
pub struct Truth {
    pub scramble: String,
    /// As the cuber turns it: rotations, wide moves and slices included
    pub solution: String,
    /// Each move played: the move, its start and end (ms of the video), its stage
    pub moves: Vec<(String, f64, f64, String)>,
    /// End of each stage (cross, F2L, OLL, PLL), ms of the video
    pub stages: [f64; 4],
}

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

/// Levenshtein distance between two lists of quarter turns, and the pairs it matched as equal
pub fn align(a: &[Quarter], b: &[Quarter]) -> (usize, Vec<(Quarter, Quarter)>) {
    let same = |p: &Quarter, q: &Quarter| p.face == q.face && p.dir == q.dir;
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + !same(&a[i - 1], &b[j - 1]) as usize);
        }
    }
    let mut matched = Vec::new();
    let (mut i, mut j) = (a.len(), b.len());
    while i > 0 && j > 0 {
        if same(&a[i - 1], &b[j - 1]) && d[i][j] == d[i - 1][j - 1] {
            matched.push((a[i - 1], b[j - 1]));
            i -= 1;
            j -= 1;
        } else if d[i][j] == d[i - 1][j - 1] + 1 {
            i -= 1;
            j -= 1;
        } else if d[i][j] == d[i - 1][j] + 1 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    matched.reverse();
    (d[a.len()][b.len()], matched)
}

/// Rotations, wide moves and slices in a solution: each turns the cube in the hands
pub fn rotations_in(solution: &str) -> usize {
    use Family::*;
    solution.split(js_space).filter_map(parse_move).filter(|m| matches!(m.family, X | Y | Z | M | E | S) || m.to > 1).count()
}

#[derive(Clone, Debug, PartialEq)]
pub struct TruthScore {
    /// Quarter turns played
    pub quarters: usize,
    pub distance: usize,
    /// Share of the quarter turns found exactly (1 - distance / quarters)
    pub exact: f64,
    /// Median gap between a move found and the end of the move played, ms
    pub time_median: Option<f64>,
    /// End of each stage found (cross, F2L, OLL, PLL), minus the truth's, ms (None: stage not found)
    pub stages: [Option<f64>; 4],
    /// Rotations played, and found (None: the reader gave no orientation)
    pub rotations: (usize, Option<usize>),
}

fn median(mut xs: Vec<f64>) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    Some(if n % 2 == 1 { xs[(n - 1) / 2] } else { (xs[n / 2 - 1] + xs[n / 2]) / 2.0 })
}

pub fn score_against_truth(result: &VisionResult, truth: &Truth) -> TruthScore {
    let played: Vec<(&str, f64)> = truth.moves.iter().map(|m| (m.0.as_str(), m.2)).collect();
    let played = quarters(&played, "");
    let found: Vec<(&str, f64)> = result.timed.iter().map(|m| (m.token.as_str(), m.t)).collect();
    let found = quarters(&found, &result.hold);
    let (distance, matched) = align(&played, &found);
    let stages = std::array::from_fn(|s| {
        let split = result.reconstruction.splits.iter().find(|split| split.phase == Stage::ALL[s].phase())?;
        Some(js_round(result.reconstruction.start + split.end - truth.stages[s]))
    });
    TruthScore {
        quarters: played.len(),
        distance,
        exact: if played.is_empty() { 1.0 } else { (1.0 - distance as f64 / played.len() as f64).max(0.0) },
        time_median: median(matched.iter().map(|(p, q)| (q.t - p.t).abs()).collect()),
        stages,
        rotations: (rotations_in(&truth.solution), result.orientation.as_ref().map(|o| o.rotations as usize)),
    }
}
