//! Cube states and how well a reading fits one (`search.ts`, without the search itself)

use std::sync::OnceLock;

use cube_core::Cube;

use crate::reading::{Reading, State, TimedMove};

/// The 18 face turns, U R F D L B, each as X, X2, X'
pub const MOVES: [&str; 18] = ["U", "U2", "U'", "R", "R2", "R'", "F", "F2", "F'", "D", "D2", "D'", "L", "L2", "L'", "B", "B2", "B'"];

/// Opposite face of each face (U R F D L B)
pub const OPPOSITE: [usize; 6] = [3, 4, 5, 0, 1, 2];

pub fn state_of(cube: &Cube) -> State {
    cube.colors()
}

pub fn solved_state() -> State {
    state_of(&Cube::solved())
}

/// For each move, where every sticker comes from
fn sources() -> &'static [[u8; 54]; 18] {
    static SOURCES: OnceLock<[[u8; 54]; 18]> = OnceLock::new();
    SOURCES.get_or_init(|| MOVES.map(|m| *Cube::solved().apply(m).stickers()))
}

/// The state after one of the 18 MOVES
pub fn apply_move(state: &State, mv: usize) -> State {
    let src = &sources()[mv];
    std::array::from_fn(|i| state[src[i] as usize])
}

/// A face read in rotation r: cell i of the reading is sticker ROT[r][i] of the face
pub const ROT: [[usize; 9]; 4] =
    [[0, 1, 2, 3, 4, 5, 6, 7, 8], [6, 3, 0, 7, 4, 1, 8, 5, 2], [8, 7, 6, 5, 4, 3, 2, 1, 0], [2, 5, 8, 1, 4, 7, 0, 3, 6]];

/// A face read in a frame may be wrong altogether (a lattice across a hand, stickers swapped by motion
/// blur, a face seen edge-on): with this probability it says nothing about the cube, and a wrong reading
/// costs no more than random colours instead of ruling the true state out
pub const GARBAGE: f64 = 0.5;

/// A reading prepared for scoring many states
#[derive(Clone, Debug)]
pub struct Scorer {
    pub centers: Vec<u8>,
    /// Face f, cell i, colour c at (f * 9 + i) * 6 + c; NaN: not seen. f32, as ChronoCube's Float32Array
    pub logs: Vec<f32>,
    /// For each face, the score of a misreading
    pub garbage: Vec<f64>,
    pub misread: f64,
}

pub fn scorer(reading: &Reading, misread: f64) -> Scorer {
    let mut logs = vec![f32::NAN; reading.faces.len() * 54];
    let garbage = reading
        .faces
        .iter()
        .enumerate()
        .map(|(f, face)| {
            let mut seen = 0;
            for (i, cell) in face.stickers.iter().enumerate() {
                let Some(p) = cell else { continue };
                if i == 4 {
                    continue;
                }
                seen += 1;
                for c in 0..6 {
                    logs[(f * 9 + i) * 6 + c] = (p[c] as f64).ln() as f32;
                }
            }
            misread.ln() + seen as f64 * (1.0f64 / 6.0).ln()
        })
        .collect();
    Scorer { centers: reading.faces.iter().map(|f| f.center).collect(), logs, garbage, misread }
}

/// Log-likelihood of a prepared reading for a state; with `rotations`, the rotation of each face that explains it
pub fn score(state: &State, s: &Scorer, mut rotations: Option<&mut Vec<usize>>) -> f64 {
    let keep = (1.0 - s.misread).ln();
    let mut total = 0.0;
    for (f, &center) in s.centers.iter().enumerate() {
        let base = center as usize * 9;
        let (mut best, mut best_r) = (f64::NEG_INFINITY, 0);
        for (r, rot) in ROT.iter().enumerate() {
            let mut sum = 0.0;
            for i in 0..9 {
                let l = s.logs[(f * 9 + i) * 6 + state[base + rot[i]] as usize];
                if !l.is_nan() {
                    sum += l as f64;
                }
            }
            if sum > best {
                best = sum;
                best_r = r;
            }
        }
        total += (best + keep).max(s.garbage[f]);
        if let Some(r) = rotations.as_deref_mut() {
            r.push(best_r);
        }
    }
    total
}

/// As `score`, with what the reader tells of each face's rotation (`turns`: for each face of the reading, the
/// log-probability of the 4 rotations; None: it does not tell). Two states that show the same faces up to a
/// rotation, as a D turn and a U' turn do from the solved cube, are then told apart.
pub fn score_turned(state: &State, s: &Scorer, turns: &[Option<[f64; 4]>]) -> f64 {
    let keep = (1.0 - s.misread).ln();
    let mut total = 0.0;
    for (f, &center) in s.centers.iter().enumerate() {
        let base = center as usize * 9;
        let mut best = f64::NEG_INFINITY;
        for (r, rot) in ROT.iter().enumerate() {
            let mut sum = turns.get(f).copied().flatten().map_or(0.0, |t| t[r]);
            for i in 0..9 {
                let l = s.logs[(f * 9 + i) * 6 + state[base + rot[i]] as usize];
                if !l.is_nan() {
                    sum += l as f64;
                }
            }
            best = best.max(sum);
        }
        total += (best + keep).max(s.garbage[f]);
    }
    total
}

/// Log-likelihood of a reading for a state, and the rotation of each face that explains it
pub fn reading_score(state: &State, reading: &Reading) -> (f64, Vec<usize>) {
    let mut rotations = Vec::new();
    let s = score(state, &scorer(reading, GARBAGE), Some(&mut rotations));
    (s, rotations)
}

/// A 52-bit hash of a state (two FNV-1a passes), to merge equal states
pub fn hash_of(s: &State) -> u64 {
    let (mut a, mut b) = (0x811c9dc5u32, 0x01000193u32);
    for (i, &x) in s.iter().enumerate() {
        a = (a ^ x as u32).wrapping_mul(0x01000193);
        b = (b ^ x as u32 ^ i as u32).wrapping_mul(0x5bd1e995);
    }
    ((a as u64) << 20) + (b & 0xfffff) as u64
}

/// The most likely colour of a cell: the first of the largest
fn argmax(p: &[f32; 6]) -> usize {
    let mut best = 0;
    for c in 1..6 {
        if p[c] > p[best] {
            best = c;
        }
    }
    best
}

/// Two readings that see the same faces with mostly the same colours: the cube seen again as it was (its
/// errors with it), not a new view; after a turn or a new side, they differ
pub fn alike(a: &Reading, b: &Reading, share: f64) -> bool {
    let (mut same, mut compared) = (0usize, 0usize);
    for fb in &b.faces {
        let Some(fa) = a.faces.iter().find(|f| f.center == fb.center) else { continue };
        for (i, (p, q)) in fb.stickers.iter().zip(&fa.stickers).enumerate() {
            let (Some(p), Some(q)) = (p, q) else { continue };
            if i == 4 {
                continue;
            }
            compared += 1;
            if argmax(p) == argmax(q) {
                same += 1;
            }
        }
    }
    compared >= 4 && same as f64 >= share * compared as f64
}

/// Moves found, and how well they explain the readings
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub moves: Vec<TimedMove>,
    /// States after each reading, and the rotation each face was read in
    pub path: Vec<(State, Vec<usize>)>,
    /// Share of the stickers read that agree with the reconstruction (0–1)
    pub agreement: f64,
    pub solved: bool,
}
