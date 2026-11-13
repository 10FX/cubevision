//! A solve found in a video, played back: the cube at any moment of it

use cube_core::Cube;
use cube_decode::orientation::{OrientationTrack, Quat, decode_track, orientation_at, quat_conj, quat_mul, quat_of, rotation_of};
use cube_decode::pipeline::VisionResult;
use cube_decode::present::leading_rotation;

/// Time shown before the first turn and after the last, ms
const MARGIN: f64 = 600.0;

pub struct Replay {
    /// The scrambled cube, held as it was scrambled
    start: Cube,
    /// The face turns, as made in the hands that scrambled: what the 3D cube plays
    pub turns: Vec<String>,
    /// What is written for each turn: the turn itself, or a rotation and it, a wide move, a slice
    pub written: Vec<Vec<String>>,
    /// When each turn ends, ms from the start of the video
    pub times: Vec<f64>,
    /// How the cube is held as the solve is written: its cross down from the first turn
    held: Quat,
    /// The cube's orientation through the solve, when the video gave it, and the turn that sets it on `held`
    /// just before the first move
    track: Option<(OrientationTrack, Vec<Quat>, Quat)>,
}

/// A quaternion (w x y z) as the rotation matrix it stands for
fn matrix(&[w, x, y, z]: &Quat) -> [[f32; 3]; 3] {
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
    .map(|row| row.map(|v| v as f32))
}

impl Replay {
    pub fn new(scramble: &str, result: &VisionResult) -> Self {
        let times: Vec<f64> = result.timed.iter().map(|m| m.t).collect();
        // Shown as the solve is written: the cross down from the first turn (the solution's first rotations)
        let held = quat_of(&rotation_of(&leading_rotation(&result.reconstruction.solution).0));
        // The hold read in the video may be off by a whole turn of the cube: it is set at the first turn to the
        // hold written, and only the cube's turns since then are followed
        let track = result.orientation.as_ref().and_then(|o| o.track.clone()).zip(times.first()).map(|(track, first)| {
            let samples = decode_track(&track);
            let fix = quat_mul(&held, &quat_conj(&orientation_at(&track, &samples, first - 50.0)));
            (track, samples, fix)
        });
        Replay {
            start: Cube::solved().apply(&result.hold).apply(scramble),
            turns: result.timed.iter().map(|m| m.token.clone()).collect(),
            written: result.notation.clone(),
            times,
            held,
            track,
        }
    }

    /// Whether the cube follows the orientation read in the video
    pub fn follows(&self) -> bool {
        self.track.is_some()
    }

    /// How the whole cube is turned at this time of the video: as it was in the cuber's hands when the video
    /// gave its orientation, else as the solve is written
    pub fn orientation(&self, clock: f64) -> [[f32; 3]; 3] {
        match &self.track {
            Some((track, samples, fix)) => matrix(&quat_mul(fix, &orientation_at(track, samples, clock))),
            None => matrix(&self.held),
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
        self.turns.iter().take(moves).fold(self.start.clone(), |cube, token| cube.apply(token))
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
