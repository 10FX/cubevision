//! Following the cube live from the camera, before a solve (ChronoCube's vision/live.ts): which turns are made
//! (LiveTracker) and how far into the scramble they are (ScrambleFollower).
//!
//! The tracker keeps several hypotheses about the cube's state (faces named by their centre colour, so the cube may be
//! held and rotated freely), like the offline search: each frame extends every hypothesis by staying put or by one turn
//! (two after frames where nothing could be read), scores them by how well they explain what the camera reads, and
//! keeps the best. One frame cannot tell some turns apart (a face is read up to a rotation); the next ones can. An
//! optional bonus favours the states that follow the scramble shown: the cuber is turning it.

use std::collections::HashMap;

use cube_core::{Cube, FACES, invert_alg, js_space, parse_move};

use crate::reading::{FaceReading, Reading, State};
use crate::state::{MOVES, apply_move, hash_of, score, scorer, state_of};

/// The live tracker keeps the misreading rate it was tuned with (the video search's was raised to 0.5)
const LIVE_MISREAD: f64 = 0.15;

struct Hypothesis {
    state: State,
    score: f64,
    /// Turns from the start (indices into MOVES, faces named by centre colour)
    moves: Vec<usize>,
}

/// A turn added to a sequence, merged with a turn of the same face just before (U U' vanish, U U = U2)
pub fn append_move(moves: &[usize], mv: usize) -> Vec<usize> {
    let mut out = moves.to_vec();
    match moves.last() {
        Some(&last) if last / 3 == mv / 3 => {
            let turns = (last % 3 + 1 + mv % 3 + 1) % 4;
            out.pop();
            if turns != 0 {
                out.push(mv / 3 * 3 + turns - 1);
            }
        }
        _ => out.push(mv),
    }
    out
}

/// The turns that undo a sequence (in reverse order, each inverted)
pub fn undo_moves(moves: &[usize]) -> Vec<usize> {
    moves.iter().rev().map(|&m| m / 3 * 3 + [2, 1, 0][m % 3]).collect()
}

/// A state's log-likelihood bonus
pub type Bonus = Box<dyn Fn(&State) -> f64 + Send>;
/// States the cube may jump to, each with the turns from the start
pub type Anchors = Box<dyn Fn() -> Vec<(State, Vec<usize>)> + Send>;

pub struct TrackerOptions {
    /// Cost of a turn (log-likelihood)
    pub move_cost: f64,
    /// Hypotheses kept
    pub beam: usize,
    /// Cost of a jump to an anchor (log-likelihood): the frame must explain the state far better than the others
    pub jump_cost: f64,
    /// Log-likelihood bonus of a state, e.g. for following the scramble (once per state reached)
    pub bonus: Option<Bonus>,
    /// States the cube may jump to in one frame, whatever turns lead there, e.g. the next ones of a scramble turned
    /// faster than the camera could follow (the turns given lead from the start)
    pub anchors: Option<Anchors>,
}

impl Default for TrackerOptions {
    fn default() -> Self {
        TrackerOptions { move_cost: 4.0, beam: 24, jump_cost: 12.0, bonus: None, anchors: None }
    }
}

pub struct LiveTracker {
    beam: Vec<Hypothesis>,
    blind: u32,
    o: TrackerOptions,
}

impl LiveTracker {
    pub fn new(start: State, o: TrackerOptions) -> Self {
        let mut tracker = LiveTracker { beam: Vec::new(), blind: 0, o };
        tracker.reset(start);
        tracker
    }

    /// The cube is in this state now (the cuber says it is solved, a new scramble starts…)
    pub fn reset(&mut self, state: State) {
        self.beam = vec![Hypothesis { state, score: 0.0, moves: Vec::new() }];
        self.blind = 0;
    }

    /// The most likely state
    pub fn state(&self) -> &State {
        &self.beam[0].state
    }

    /// The turns that lead to it
    pub fn moves(&self) -> &[usize] {
        &self.beam[0].moves
    }

    /// Lead of the best state over the next one (log-likelihood): how sure the tracker is
    pub fn confidence(&self) -> f64 {
        if self.beam.len() > 1 { self.beam[0].score - self.beam[1].score } else { f64::INFINITY }
    }

    /// One frame's readings (empty: nothing could be read)
    pub fn update(&mut self, faces: &[FaceReading]) {
        if faces.is_empty() {
            self.blind += 1;
            return;
        }
        let prepared = scorer(&Reading { from: 0.0, to: 0.0, faces: faces.to_vec(), moved: None }, LIVE_MISREAD);
        let move_cost = self.o.move_cost;
        // Hypotheses in the order first offered (a Map's), the best score for each state
        let mut next: Vec<Hypothesis> = Vec::new();
        let mut index: HashMap<u64, usize> = HashMap::new();
        let bonus = &self.o.bonus;
        let mut offer = |from: f64, state: State, moves: Vec<usize>, cost: f64| {
            let key = hash_of(&state);
            let gain = match bonus {
                Some(bonus) if cost > 0.0 => bonus(&state),
                _ => 0.0,
            };
            let total = from - cost + gain + score(&state, &prepared, None);
            match index.get(&key) {
                Some(&i) => {
                    if next[i].score < total {
                        next[i] = Hypothesis { state, score: total, moves };
                    }
                }
                None => {
                    index.insert(key, next.len());
                    next.push(Hypothesis { state, score: total, moves });
                }
            }
        };
        // After frames where nothing was read, two turns may have been made
        let deep = self.blind >= 3;
        for (rank, h) in self.beam.iter().enumerate() {
            offer(h.score, h.state, h.moves.clone(), 0.0);
            for a in 0..18 {
                let sa = apply_move(&h.state, a);
                let with_a = append_move(&h.moves, a);
                offer(h.score, sa, with_a.clone(), move_cost);
                if !deep || rank >= 6 {
                    continue;
                }
                for b in 0..18 {
                    if a / 3 == b / 3 {
                        continue;
                    }
                    offer(h.score, apply_move(&sa, b), append_move(&with_a, b), 2.0 * move_cost);
                }
            }
        }
        // A jump ahead, from the likeliest state
        if let Some(anchors) = &self.o.anchors {
            let from = self.beam[0].score;
            for (state, moves) in anchors() {
                offer(from, state, moves, self.o.jump_cost);
            }
        }
        self.blind = 0;
        next.sort_by(|x, y| y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal));
        next.truncate(self.o.beam);
        // Keep the numbers small: only differences matter
        let top = next[0].score;
        for h in &mut next {
            h.score -= top;
        }
        self.beam = next;
    }
}

/// Where a tracked state stands in a scramble
#[derive(Clone, Debug, PartialEq)]
pub struct ScrambleProgress {
    /// Moves of the scramble done (a double turn half made counts half)
    pub done: f64,
    pub total: f64,
    /// How the cube may be held, given what was done so far (all 24 at the start)
    pub holds: Vec<String>,
}

/// How the cube is held to scramble: white on top, green in front (the WCA's), so the scramble's U turns the white face
pub const SCRAMBLE_HOLD: &str = "";

/// The state after a move of the scramble held white on top, green in front, and the turns to it
#[derive(Clone, Debug, PartialEq)]
pub struct ScrambleStep {
    pub state: State,
    pub done: f64,
    pub moves: Vec<usize>,
}

/// Where a tracked state stands in a scramble, for each way the scrambler may hold the cube (ORIENTATIONS for all 24)
pub struct ScrambleFollower {
    pub total: f64,
    pub tokens: Vec<String>,
    /// The state after each move of the scramble held white on top, green in front, and the turns to it
    pub steps: Vec<ScrambleStep>,
    index: HashMap<u64, Vec<(String, f64)>>,
    letter_of: Vec<(String, [char; 6])>,
}

impl ScrambleFollower {
    pub fn new(scramble: &str, holds: &[&str]) -> Self {
        let tokens: Vec<String> = scramble.split(js_space).filter(|t| !t.is_empty()).map(str::to_string).collect();
        let mut index: HashMap<u64, Vec<(String, f64)>> = HashMap::new();
        let mut letter_of = Vec::with_capacity(holds.len());
        for &hold in holds {
            let held = Cube::solved().apply(hold);
            let letters = std::array::from_fn(|colour| FACES[(0..6).find(|&f| held.color(f, 1, 1) as usize == colour).unwrap()]);
            if !letter_of.iter().any(|(h, _): &(String, [char; 6])| h == hold) {
                letter_of.push((hold.to_string(), letters));
            }
            let undo = invert_alg(hold);
            let mut add = |alg: &str, done: f64| {
                let key = hash_of(&state_of(&Cube::solved().apply(&format!("{hold} {alg} {undo}"))));
                index.entry(key).or_default().push((hold.to_string(), done));
            };
            add("", 0.0);
            let mut prefix = String::new();
            for (k, token) in tokens.iter().enumerate() {
                // Half way through a double turn
                if parse_move(token).is_some_and(|m| m.turns == 2) {
                    let first = token.chars().next().unwrap();
                    add(&format!("{prefix} {first}"), k as f64 + 0.5);
                    add(&format!("{prefix} {first}'"), k as f64 + 0.5);
                }
                prefix = format!("{prefix} {token}");
                add(&prefix, (k + 1) as f64);
            }
        }
        // Held as scrambled, the scramble's letters are the faces' centres
        let mut steps = vec![ScrambleStep { state: state_of(&Cube::solved()), done: 0.0, moves: Vec::new() }];
        let mut moves = Vec::new();
        for (k, token) in tokens.iter().enumerate() {
            let Some(m) = MOVES.iter().position(|&n| n == token) else { break };
            moves.push(m);
            steps.push(ScrambleStep { state: state_of(&Cube::solved().apply(&tokens[..=k].join(" "))), done: (k + 1) as f64, moves: moves.clone() });
        }
        ScrambleFollower { total: tokens.len() as f64, tokens, steps, index, letter_of }
    }

    /// The fewest turns (two at most) from a state off the scramble back onto it, at the furthest point from `after`
    /// on, with how far that is; None when none
    pub fn way_back(&self, state: &State, after: f64) -> Option<(Vec<usize>, f64)> {
        let mut best: Option<(Vec<usize>, f64)> = None;
        for a in 0..18 {
            self.consider(&mut best, &apply_move(state, a), vec![a], after);
        }
        if best.is_some() {
            return best;
        }
        for a in 0..18 {
            let sa = apply_move(state, a);
            for b in 0..18 {
                if a / 3 != b / 3 {
                    self.consider(&mut best, &apply_move(&sa, b), vec![a, b], after);
                }
            }
        }
        best
    }

    /// A way back kept when it is shorter than the best so far, or as short and further into the scramble
    fn consider(&self, best: &mut Option<(Vec<usize>, f64)>, s: &State, moves: Vec<usize>, after: f64) {
        if let Some(found) = self.locate(s)
            && found.done >= after
            && best.as_ref().is_none_or(|(b, done)| moves.len() < b.len() || (moves.len() == b.len() && found.done > *done))
        {
            *best = Some((moves, found.done));
        }
    }

    /// Progress of a state, or None when it is off the scramble
    pub fn locate(&self, state: &State) -> Option<ScrambleProgress> {
        let found = self.index.get(&hash_of(state))?;
        let done = found.iter().map(|f| f.1).fold(f64::NEG_INFINITY, f64::max);
        let mut holds: Vec<String> = Vec::new();
        for (hold, d) in found {
            if *d == done && !holds.contains(hold) {
                holds.push(hold.clone());
            }
        }
        Some(ScrambleProgress { done, total: self.total, holds })
    }

    /// A turn (faces named by centre colour) as written by someone holding the cube this way (one of the holds given)
    pub fn written(&self, mv: usize, hold: &str) -> String {
        let name = MOVES[mv];
        let (_, letters) = self.letter_of.iter().find(|(h, _)| h == hold).expect("a hold the follower was given");
        let face = FACES.iter().position(|&f| name.starts_with(f)).unwrap();
        format!("{}{}", letters[face], &name[1..])
    }
}
