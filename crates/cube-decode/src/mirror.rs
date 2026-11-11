//! Videos from a selfie camera are often mirrored (ChronoCube's vision/mirror.ts): the cube in them is one no hand
//! could hold (its colours the wrong way round each corner), and the reader, trained on real cubes, misreads it. The
//! scrambled state is known (held white on top, green in front, or as told), so the first frames are read twice, as
//! they are and flipped: the way whose readings fit the scrambled cube better is how the whole video is read.

use cube_core::{Cube, invert_alg};

use crate::motion::Frame;
use crate::reading::{FaceReading, Reading, State};
use crate::state::{GARBAGE, score, scorer, state_of};

/// A frame counts when one way or the other shows the scrambled cube: this much evidence (log-likelihood over a
/// misreading; a face read right is worth about 13). Frames once the solve has started fit neither way, and say nothing.
const EVIDENCE: f64 = 8.0;
/// Frames that count before the two ways may be told apart, and after which the plain one is kept unless the other leads
const MIN_FRAMES: u32 = 4;
const MAX_FRAMES: u32 = 30;
/// Frames of the cube read twice after which the video is taken as it is, unless flipped leads (reading twice costs)
const MAX_TRIED: u32 = 300;
/// Log-likelihood by which one way must beat the other to be taken
const MARGIN: f64 = 25.0;

/// A frame's RGBA bytes flipped left to right
pub fn flip_frame(f: &Frame) -> Vec<u8> {
    let mut out = vec![0; f.data.len()];
    let row = f.width * 4;
    if row == 0 {
        return out;
    }
    for (src, dst) in f.data.chunks_exact(row).zip(out.chunks_exact_mut(row)).take(f.height) {
        for (s, d) in src.chunks_exact(4).rev().zip(dst.chunks_exact_mut(4)) {
            d.copy_from_slice(s);
        }
    }
    out
}

pub struct MirrorCheck {
    start: State,
    /// Per way (as it is, flipped): what the readings tell so far
    scores: [f64; 2],
    frames: u32,
    tried: u32,
    /// Whether the video is mirrored: None until known
    pub mirrored: Option<bool>,
}

impl MirrorCheck {
    /// The scramble, and how the cube was held for it when known (white on top, green in front otherwise)
    pub fn new(scramble: &str, hold: Option<&str>) -> Self {
        let h = hold.map_or("", str::trim);
        let start = state_of(&Cube::solved().apply(&format!("{h} {scramble} {}", invert_alg(h))));
        MirrorCheck { start, scores: [0.0; 2], frames: 0, tried: 0, mirrored: None }
    }

    /// How much more the flipped readings tell than the plain ones so far (log-likelihood)
    pub fn lead(&self) -> f64 {
        self.scores[1] - self.scores[0]
    }

    /// One frame's faces as read and flipped (before the solve starts: the cube still scrambled); the answer once known
    pub fn add(&mut self, normal: &[FaceReading], flipped: &[FaceReading]) -> Option<bool> {
        if self.mirrored.is_some() {
            return self.mirrored;
        }
        self.tried += 1;
        if self.tried >= MAX_TRIED {
            self.mirrored = Some(self.lead() > MARGIN);
            return self.mirrored;
        }
        if normal.is_empty() && flipped.is_empty() {
            return None;
        }
        // What each reading tells beyond a misreading (a way that reads less is not the likelier for it)
        let gains = [normal, flipped].map(|faces| {
            if faces.is_empty() {
                return 0.0;
            }
            // Sanitized first, as the decoder's readings (a NaN: not seen): one NaN would hide the answer for good
            let faces = faces.iter().cloned().map(FaceReading::sanitized).collect();
            let prepared = scorer(&Reading { from: 0.0, to: 0.0, faces, moved: None }, GARBAGE);
            score(&self.start, &prepared, None) - prepared.garbage.iter().fold(0.0, |a, b| a + b)
        });
        if gains[0].max(gains[1]) < EVIDENCE {
            return None;
        }
        self.scores[0] += gains[0];
        self.scores[1] += gains[1];
        self.frames += 1;
        let lead = self.lead();
        if self.frames >= MIN_FRAMES && lead.abs() > MARGIN {
            self.mirrored = Some(lead > 0.0);
        } else if self.frames >= MAX_FRAMES {
            self.mirrored = Some(false);
        }
        self.mirrored
    }
}
