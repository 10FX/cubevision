//! Readings made ready to score states

use crate::reading::Reading;
use crate::state::ROT;

pub struct PreparedFace {
    /// Cells seen
    pub cells: usize,
    /// Log-probability of each colour of each cell seen (6 each)
    pub logs: Vec<f32>,
    /// The sticker of the state each cell is in each rotation: rotation r at r * cells
    pub stickers: Vec<u8>,
    /// Log-probability of each rotation
    pub turns: [f32; 4],
    /// The best the face could score
    pub best: f64,
    /// log(misread) + cells seen × log(1/6): a misreading
    pub garbage: f64,
}

pub struct Prepared {
    pub from: f64,
    pub to: f64,
    pub faces: Vec<PreparedFace>,
}

/// A reading made ready to score states; None when it shows no cell
pub fn prepare(reading: &Reading, misread: f64) -> Option<Prepared> {
    let faces: Vec<PreparedFace> = reading
        .faces
        .iter()
        .filter_map(|face| {
            let seen: Vec<usize> = (0..9).filter(|&i| i != 4 && face.stickers[i].is_some()).collect();
            if seen.is_empty() {
                return None;
            }
            let cells = seen.len();
            let mut logs = vec![0f32; cells * 6];
            let mut stickers = vec![0u8; 4 * cells];
            // Rotations equally likely, unless told: the best rotation is then worth nothing more
            let turns: [f32; 4] = std::array::from_fn(|r| face.rotation.map_or(0.0, |rot| (rot[r] as f64).max(1e-6).ln() as f32));
            let mut best = turns.iter().fold(f64::NEG_INFINITY, |a, &t| a.max(t as f64));
            for (j, &i) in seen.iter().enumerate() {
                let p = face.stickers[i].unwrap();
                let sum: f64 = p.iter().fold(0.0, |s, &x| s + (x as f64).max(1e-6));
                let mut top = f64::NEG_INFINITY;
                for c in 0..6 {
                    // The log as a double for the best, as a float in the table (Float32Array)
                    let l = ((p[c] as f64).max(1e-6) / sum).ln();
                    logs[j * 6 + c] = l as f32;
                    top = top.max(l);
                }
                best += top;
                for (r, rot) in ROT.iter().enumerate() {
                    stickers[r * cells + j] = face.center * 9 + rot[i] as u8;
                }
            }
            Some(PreparedFace { cells, logs, stickers, turns, best, garbage: misread.ln() + cells as f64 * (1.0f64 / 6.0).ln() })
        })
        .collect();
    (!faces.is_empty()).then_some(Prepared { from: reading.from, to: reading.to, faces })
}

pub fn log_add_exp(a: f64, b: f64) -> f64 {
    if a > b { a + (b - a).exp().ln_1p() } else { b + (a - b).exp().ln_1p() }
}

/// Log-likelihood of a reading for the state whose sticker i is `sticker(i)`: per face the best of its 4 rotations, or
/// a misreading. Given up (None) once `weight` times it cannot top `need`, even with the faces left read at best
/// (`rest[f]`: the bound of faces f and after).
#[inline(always)]
pub fn likelihood_above(sticker: impl Fn(usize) -> u8, p: &Prepared, keep: f64, weight: f64, need: f64, rest: &[f64]) -> Option<f64> {
    let mut total = 0.0;
    for (f, face) in p.faces.iter().enumerate() {
        if weight * (total + rest[f]) <= need {
            return None;
        }
        // The face's 9 colours once: its 4 rotations read the same stickers in other orders
        let base = face.stickers[0] as usize / 9 * 9;
        let colors: [u8; 9] = std::array::from_fn(|x| sticker(base + x));
        // The 4 sums side by side (each in the order of its cells, as one at a time): the additions of one do not wait
        // on another's
        let n = face.cells;
        let (logs, st) = (&face.logs[..n * 6], &face.stickers[..n * 4]);
        let mut sums = face.turns.map(|t| t as f64);
        for j in 0..n {
            let row = &logs[j * 6..j * 6 + 6];
            for (r, sum) in sums.iter_mut().enumerate() {
                *sum += row[colors[st[r * n + j] as usize - base] as usize] as f64;
            }
        }
        let best = sums.iter().fold(f64::NEG_INFINITY, |best, &s| if s > best { s } else { best });
        total += log_add_exp(best + keep, face.garbage);
    }
    Some(total)
}

/// The bound of each face and of the faces after it, with a margin for the float rounding of the logs: `rest[f]`
pub fn rest_bounds(p: &Prepared, keep: f64) -> Vec<f64> {
    let mut rest = vec![0.0; p.faces.len() + 1];
    for f in (0..p.faces.len()).rev() {
        let face = &p.faces[f];
        rest[f] = rest[f + 1] + log_add_exp(face.best + keep, face.garbage) + 1e-3;
    }
    rest
}

/// The best a state could score on a reading: every face read exactly as seen
pub fn bound(p: &Prepared, keep: f64) -> f64 {
    p.faces.iter().fold(0.0, |total, face| total + log_add_exp(face.best + keep, face.garbage))
}
