//! A solve found in a video, played back: the cube at any moment of it

use cube_core::Cube;
use cube_decode::pipeline::VisionResult;

/// Time shown before the first turn and after the last, ms
const MARGIN: f64 = 600.0;

pub struct Replay {
    /// The scrambled cube, held as it was scrambled
    start: Cube,
    /// What is written for each turn: the turn itself, or a rotation and it, a wide move, a slice
    pub written: Vec<Vec<String>>,
    /// When each turn ends, ms from the start of the video
    pub times: Vec<f64>,
}

impl Replay {
    pub fn new(scramble: &str, result: &VisionResult) -> Self {
        Replay {
            start: Cube::solved().apply(&result.hold).apply(scramble),
            written: result.notation.clone(),
            times: result.timed.iter().map(|m| m.t).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// The cube once the first `moves` turns are made
    pub fn cube_at(&self, moves: usize) -> Cube {
        self.written.iter().take(moves).flatten().fold(self.start.clone(), |cube, token| cube.apply(token))
    }

    /// The turns made by this time of the video
    pub fn moves_at(&self, clock: f64) -> usize {
        self.times.iter().take_while(|&&t| t <= clock).count()
    }

    /// The times the replay runs between: a little before the first turn, a little after the last
    pub fn span(&self) -> (f64, f64) {
        match (self.times.first(), self.times.last()) {
            (Some(first), Some(last)) => (first - MARGIN, last + MARGIN),
            _ => (0.0, 0.0),
        }
    }

    /// The time just after turn `moves` (0: the start)
    pub fn time_of(&self, moves: usize) -> f64 {
        if moves == 0 { self.span().0 } else { self.times[moves.min(self.len()) - 1] }
    }
}
