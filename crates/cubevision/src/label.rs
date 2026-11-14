//! A labelling session: the program asks for moves from a solved cube, the user makes each and presses Space, so
//! the cube's state is known at every frame without a model reading it. The frames of the camera, cut to the
//! cube, are then labelled images of real cubes to train the reader on.
//!
//! A sequence scrambles the cube then undoes the scramble: a cube solved at its end tells that every move was
//! made as asked. After each press the cube is only shown for a moment (`STILL`: nothing turns, the frames have
//! the state for sure), then the next move is asked for (its frames show the state before, the move on its way,
//! or the state after: told apart later).

use cube_decode::state::{MOVES, ROT, apply_move, solved_state};
use cube_decode::{FaceReading, State};
use serde_json::{Value, json};

/// Seconds after a press when the cube is only shown, nothing turned
pub const STILL: f64 = 1.5;

/// Why nothing is recorded
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pause {
    /// The session begins: the solved cube is to be shown
    Start,
    /// A move was taken back: the cube is to be put back as on screen
    Undone,
    /// The cube no longer matches the screen: it is to be solved
    Lost,
    /// The sequence's last move is made: is the cube solved?
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Paused(Pause),
    /// The cube is shown, nothing turns
    Still,
    /// The move asked for is being made
    Move,
}

/// What is known of a frame
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub sequence: usize,
    /// Moves of the sequence made before this frame
    pub step: usize,
    /// Frames of one phase share a segment: a move taken back discards whole segments
    pub segment: usize,
    /// "still": the cube is in `state`; "move": in `state`, turning `mv`, or just after it
    pub phase: &'static str,
    pub state: State,
    pub mv: Option<usize>,
    /// Seconds since the phase began
    pub age: f64,
}

/// A time as the log has it: to the millisecond
pub fn ms(t: f64) -> f64 {
    (t * 1000.0).round() / 1000.0
}

/// SplitMix64: the moves of a seed are the same everywhere
fn next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The move that undoes a move
fn inverse(mv: usize) -> usize {
    mv / 3 * 3 + [2, 1, 0][mv % 3]
}

/// A sequence's moves (indices into `MOVES`): `length` random ones, never the same face twice in a row nor three
/// turns on one axis, then the moves that undo them
pub fn plan(seed: u64, length: usize) -> Vec<usize> {
    let mut seed = seed;
    let mut moves: Vec<usize> = Vec::with_capacity(2 * length);
    let axis = |m: usize| (m / 3) % 3;
    while moves.len() < length {
        let mv = (next(&mut seed) % 18) as usize;
        let n = moves.len();
        if n >= 1 && moves[n - 1] / 3 == mv / 3 {
            continue;
        }
        if n >= 2 && axis(moves[n - 1]) == axis(mv) && axis(moves[n - 2]) == axis(mv) {
            continue;
        }
        moves.push(mv);
    }
    let back: Vec<usize> = moves.iter().rev().map(|&m| inverse(m)).collect();
    moves.extend(back);
    moves
}

pub struct Session {
    seed: u64,
    length: usize,
    sequence: usize,
    plan: Vec<usize>,
    step: usize,
    state: State,
    phase: Phase,
    /// When the phase began (seconds)
    since: f64,
    segment: usize,
    /// For each move made, the state before it and the segment its move was asked in
    made: Vec<(State, usize)>,
    events: Vec<Value>,
}

impl Session {
    /// `length`: the moves that scramble (as many undo them)
    pub fn new(seed: u64, length: usize) -> Session {
        let length = length.max(1);
        Session {
            seed,
            length,
            sequence: 0,
            plan: plan(seed, length),
            step: 0,
            state: solved_state(),
            phase: Phase::Paused(Pause::Start),
            since: 0.0,
            segment: 0,
            made: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// The cube as it should be now
    pub fn state(&self) -> State {
        self.state
    }

    pub fn sequence(&self) -> usize {
        self.sequence
    }

    /// Moves of the sequence made
    pub fn step(&self) -> usize {
        self.step
    }

    /// The sequence's moves
    pub fn plan(&self) -> &[usize] {
        &self.plan
    }

    /// The move to make next (none: the sequence is over)
    pub fn prompt(&self) -> Option<usize> {
        self.plan.get(self.step).copied()
    }

    /// Share of the still time gone by (1: over, or not a still time)
    pub fn still_progress(&self, t: f64) -> f64 {
        if self.phase == Phase::Still { ((t - self.since) / STILL).clamp(0.0, 1.0) } else { 1.0 }
    }

    /// Seconds since the phase began
    pub fn age(&self, t: f64) -> f64 {
        t - self.since
    }

    fn enter(&mut self, phase: Phase, t: f64) {
        self.phase = phase;
        self.since = t;
        self.segment += 1;
    }

    fn start_sequence(&mut self, t: f64) {
        self.sequence += 1;
        self.plan = plan(self.seed.wrapping_add(self.sequence as u64), self.length);
        self.step = 0;
        self.state = solved_state();
        self.made.clear();
        self.enter(Phase::Still, t);
    }

    /// Time goes by: the still time over, the move is asked for
    pub fn tick(&mut self, t: f64) {
        if self.phase == Phase::Still && t - self.since >= STILL {
            if self.step == self.plan.len() {
                self.enter(Phase::Paused(Pause::Done), t);
            } else {
                self.enter(Phase::Move, t);
            }
        }
    }

    /// Space: the move asked for is made, or what a pause waited for is done
    pub fn space(&mut self, t: f64) {
        match self.phase {
            Phase::Still => {}
            Phase::Move => {
                let Some(mv) = self.prompt() else { return };
                self.events.push(json!({"type": "press", "t": ms(t), "sequence": self.sequence, "step": self.step, "move": MOVES[mv]}));
                self.made.push((self.state, self.segment));
                self.state = apply_move(&self.state, mv);
                self.step += 1;
                self.enter(Phase::Still, t);
            }
            Phase::Paused(Pause::Start | Pause::Undone) => self.enter(Phase::Still, t),
            Phase::Paused(Pause::Done) => {
                self.events.push(json!({"type": "sequence", "t": ms(t), "index": self.sequence, "ok": true}));
                self.start_sequence(t);
            }
            Phase::Paused(Pause::Lost) => self.start_sequence(t),
        }
    }

    /// The last move is taken back (it was not the one asked for, or Space came too early): its frames are
    /// discarded, and the cube is to be put back as it was before it
    pub fn undo(&mut self, t: f64) {
        if matches!(self.phase, Phase::Paused(Pause::Start | Pause::Lost)) {
            return;
        }
        let Some((state, from)) = self.made.pop() else { return };
        let segments: Vec<usize> = (from..=self.segment).collect();
        self.events.push(json!({"type": "discard", "t": ms(t), "sequence": self.sequence, "step": self.step - 1, "segments": segments}));
        self.state = state;
        self.step -= 1;
        self.enter(Phase::Paused(Pause::Undone), t);
    }

    /// The cube no longer matches the screen: the sequence is wrong, and a new one starts once the cube is solved
    pub fn lost(&mut self, t: f64) {
        if matches!(self.phase, Phase::Paused(Pause::Start | Pause::Lost)) {
            return;
        }
        self.events.push(json!({"type": "sequence", "t": ms(t), "index": self.sequence, "ok": false}));
        self.enter(Phase::Paused(Pause::Lost), t);
    }

    /// What is known of a frame taken now (None: nothing is recorded)
    pub fn label(&self, t: f64) -> Option<Label> {
        let (phase, mv) = match self.phase {
            Phase::Paused(_) => return None,
            Phase::Still => ("still", None),
            Phase::Move => ("move", self.prompt()),
        };
        Some(Label { sequence: self.sequence, step: self.step, segment: self.segment, phase, state: self.state, mv, age: t - self.since })
    }

    /// The records made since the last call, for the session's log
    pub fn take_events(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.events)
    }
}

/// Share of the stickers read (their likeliest colour) that are as the state has them, each face turned as it
/// fits best; None: no sticker read
pub fn fit(state: &State, faces: &[FaceReading]) -> Option<f64> {
    let (mut right, mut all) = (0usize, 0usize);
    for face in faces {
        let read: Vec<(usize, u8)> = face
            .stickers
            .iter()
            .enumerate()
            .filter_map(|(cell, p)| {
                let p = p.as_ref()?;
                let colour = (0..6).max_by(|&a, &b| p[a].total_cmp(&p[b]))?;
                Some((cell, colour as u8))
            })
            .collect();
        let base = face.center as usize * 9;
        let best =
            ROT.iter().map(|rot| read.iter().filter(|&&(cell, colour)| state[base + rot[cell]] == colour).count()).max().unwrap_or(0);
        right += best;
        all += read.len();
    }
    (all > 0).then(|| right as f64 / all as f64)
}

/// A state as text: the 54 stickers' colours, faces U R F D L B (by their centre's colour), each row by row
pub fn state_text(state: &State) -> String {
    state.iter().map(|&c| char::from(b"WRGYOB"[c as usize % 6])).collect()
}
