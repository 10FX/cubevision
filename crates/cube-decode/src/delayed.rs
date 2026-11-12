//! The live tracker, written with a delay: a turn is shown once the likeliest state has held it for `LAG` frames
//! in a row, and what is shown never changes afterwards. The tracker alone rewrites its last turns as the frames
//! come (a face read up to a rotation tells some turns apart only a few frames later); a list on screen must not.

use crate::live::{LiveTracker, TrackerOptions};
use crate::reading::{FaceReading, State};

/// Frames a turn must stay in the likeliest state before it is written: a third of a second at 30 frames a second
pub const LAG: usize = 10;

pub struct DelayedTracker {
    tracker: LiveTracker,
    /// The turns written: settled for good (indices into MOVES, faces named by centre colour)
    confirmed: Vec<usize>,
    /// For each turn not written yet, the frames it has been there unchanged, with all those before it
    ages: Vec<usize>,
    /// Those turns as they were at the last frame
    last: Vec<usize>,
}

impl DelayedTracker {
    pub fn new(start: State) -> Self {
        DelayedTracker {
            tracker: LiveTracker::new(start, TrackerOptions::default()),
            confirmed: Vec::new(),
            ages: Vec::new(),
            last: Vec::new(),
        }
    }

    /// The cube is in this state now: nothing is written any more
    pub fn reset(&mut self, state: State) {
        self.tracker.reset(state);
        self.confirmed.clear();
        self.ages.clear();
        self.last.clear();
    }

    /// The turns written so far; only ever added to
    pub fn confirmed(&self) -> &[usize] {
        &self.confirmed
    }

    /// The turns seen since, not settled yet: they may still change
    pub fn pending(&self) -> &[usize] {
        self.tracker.moves()
    }

    /// The most likely state now, pending turns included
    pub fn state(&self) -> &State {
        self.tracker.state()
    }

    pub fn confidence(&self) -> f64 {
        self.tracker.confidence()
    }

    /// One frame's readings (empty: nothing could be read); the turns this frame settles
    pub fn update(&mut self, faces: &[FaceReading]) -> Vec<usize> {
        self.tracker.update(faces);
        let now = self.tracker.moves();
        // A turn ages while it and all those before it stay as they were
        let same = now.iter().zip(&self.last).take_while(|(a, b)| a == b).count();
        self.ages.truncate(same);
        self.ages.iter_mut().for_each(|age| *age += 1);
        self.ages.resize(now.len(), 0);
        self.last = now.to_vec();
        let settled = self.ages.iter().take_while(|&&age| age >= LAG).count();
        if settled == 0 {
            return Vec::new();
        }
        let new: Vec<usize> = self.last.drain(..settled).collect();
        self.ages.drain(..settled);
        self.tracker.commit(settled);
        self.confirmed.extend(&new);
        new
    }
}
