//! What the reader gives for one image, and the readings the decoder takes from it (`model.ts`)

use cube_decode::FaceReading;

const HIDDEN: usize = 6;
const CLASSES: usize = 7;
const CELLS: usize = 9;
/// A face below this probability of being readable is left out
pub const PRESENT: f32 = 0.5;
/// Nor a face whose stickers are read, on average, below this probability: a third of such faces are mostly
/// wrong, and they mislead the search more than they guide it
pub const MIN_CONFIDENCE: f64 = 0.6;
/// A frame above this probability of a layer mid-turn is not read
pub const TURNING: f32 = 0.5;
/// Floor on a colour's probability: one confident mistake must never rule the true state out
const FLOOR: f64 = 0.02;
/// Readers whose stickers come in the cube's order: how often they do for a face read with others (the centres
/// seen tell how the cube is turned). A face read alone is turned anyhow: its rotation is left free.
const IN_ORDER: f64 = 0.97;

/// Raw outputs for one image
#[derive(Clone, Debug, PartialEq)]
pub struct ReaderOutput {
    /// Probability that each face (U R F D L B, by its centre's colour) can be read
    pub present: [f32; 6],
    /// For each face, each of its nine stickers: the probability of the six colours, then of being hidden (6 × 9 × 7)
    pub stickers: Vec<f32>,
    /// Probability of a layer mid-turn
    pub turning: f32,
    /// The cube's orientation in the camera's frame, row-major 3×3 (models without it: None)
    pub orientation: Option<[f32; 9]>,
}

impl ReaderOutput {
    /// The orientation as the decoder takes it; a matrix with a value that is no number is none
    pub fn orientation64(&self) -> Option<[f64; 9]> {
        self.orientation.filter(|m| m.iter().all(|v| v.is_finite())).map(|m| m.map(f64::from))
    }
}

/// What reads the crops of the cube: the trained reader, or a stand-in in tests
pub trait Reader {
    /// Side of the square it reads
    fn size(&self) -> usize;
    /// Its stickers come in the cube's order
    fn ordered(&self) -> bool;
    /// Crops read: RGBA squares of `size()`
    fn read(&mut self, crops: &[&[u8]]) -> anyhow::Result<Vec<ReaderOutput>>;
}

/// Model outputs → readings for the search (faces read, stickers hidden by a finger left None). `ordered`: the
/// reader gives the stickers in the cube's order; `still`: the crop did not change since the frame before, so no
/// layer is on its way and the reader's "turning" is not heeded.
///
/// Values no probability can have (not a number, infinite, negative) are never handed on: the sticker counts as
/// hidden, the face as not read.
pub fn readings_from_output(out: &ReaderOutput, ordered: bool, still: bool) -> Vec<FaceReading> {
    if (out.turning > TURNING && !still) || out.stickers.len() < 6 * CELLS * CLASSES {
        return Vec::new();
    }
    let mut faces = Vec::new();
    for face in 0..6 {
        if out.present[face] < PRESENT || !out.present[face].is_finite() {
            continue;
        }
        let mut cells = [None; CELLS];
        for (cell, slot) in cells.iter_mut().enumerate() {
            let base = (face * CELLS + cell) * CLASSES;
            let hidden = out.stickers[base + HIDDEN] as f64;
            if hidden > 0.5 {
                continue;
            }
            // Colours given that the sticker is seen, floored
            let seen = if 1.0 - hidden == 0.0 { 1.0 } else { 1.0 - hidden };
            *slot = Some(std::array::from_fn(|c| (FLOOR + (1.0 - 6.0 * FLOOR) * (out.stickers[base + c] as f64 / seen)) as f32));
        }
        let reading = FaceReading { center: face as u8, stickers: cells, rotation: None }.sanitized();
        if confidence(&reading.stickers) < MIN_CONFIDENCE {
            continue;
        }
        faces.push(reading);
    }
    if ordered && faces.len() > 1 {
        let turned = ((1.0 - IN_ORDER) / 3.0) as f32;
        for face in &mut faces {
            face.rotation = Some([IN_ORDER as f32, turned, turned, turned]);
        }
    }
    faces
}

/// How surely a face's stickers are read: the mean, over its stickers seen (centre aside), of their likeliest
/// colour's probability
pub fn confidence(stickers: &[Option<[f32; 6]>; 9]) -> f64 {
    let (mut sum, mut n) = (0.0, 0);
    for (i, p) in stickers.iter().enumerate() {
        if let (Some(p), true) = (p, i != 4) {
            sum += p.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            n += 1;
        }
    }
    if n > 0 { sum / n as f64 } else { 0.0 }
}
