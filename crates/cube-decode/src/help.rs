//! Moves the cuber vouches for, from the start of the solve (ChronoCube's help.ts, typed on the analysis page when
//! the model got lost): they are timed against the readings, and the decoder finds the rest after them.
//!
//! Each reading scores each state the moves go through by how much better that state explains it than the best state
//! a turn away: above zero while the cube is there, below once it has moved on. The readings are laid on the states
//! in order (a dynamic programme), as far as that sum keeps growing; the readings after them go to the decoder, from
//! the state the moves leave.

use cube_core::{js_space, parse_move};

use crate::decode::{DecodeResult, explain, js_round};
use crate::reading::{Reading, State, TimedMove};
use crate::state::{GARBAGE, MOVES, apply_move, score, scorer, solved_state};
use crate::truth::quarters;

/// Time given to a turn no reading saw, ms
const UNSEEN_MS: f64 = 150.0;

/// Same-face turns in a row become one: R R = R2, R R' = nothing (ChronoCube's collapse)
fn collapse(moves: &[String]) -> Vec<String> {
    let mut out: Vec<(char, u32)> = Vec::new();
    for m in moves {
        let face = m.chars().next().unwrap();
        let turns = if m.ends_with('2') { 2 } else if m.ends_with('\'') { 3 } else { 1 };
        match out.last_mut() {
            Some(last) if last.0 == face => {
                last.1 = (last.1 + turns) % 4;
                if last.1 == 0 {
                    out.pop();
                }
            }
            _ => out.push((face, turns)),
        }
    }
    out.into_iter().map(|(face, turns)| format!("{face}{}", if turns == 2 { "2" } else if turns == 3 { "'" } else { "" })).collect()
}

/// The moves typed (the cuber's notation, rotations, wide moves and slices included, from the hold that scrambled)
/// as turns of the decoder (MOVES indices); None: a token is not a move
pub fn help_moves(text: &str, hold: &str) -> Option<Vec<usize>> {
    let tokens: Vec<&str> = text.split(js_space).filter(|t| !t.is_empty()).collect();
    if !tokens.iter().all(|t| parse_move(t).is_some()) {
        return None;
    }
    let timed: Vec<(&str, f64)> = tokens.iter().map(|&t| (t, 0.0)).collect();
    let turns: Vec<String> = quarters(&timed, hold).iter().map(|q| format!("{}{}", q.face, if q.dir < 0 { "'" } else { "" })).collect();
    Some(collapse(&turns).iter().map(|m| MOVES.iter().position(|x| x == m).expect("a face turn")).collect())
}

struct Followed {
    moves: Vec<TimedMove>,
    /// The last reading laid on the moves (-1: none)
    end: isize,
    done: bool,
    tail: Option<(State, Vec<Reading>)>,
}

/// The moves given, timed against the readings, then the decoder's from where they leave the cube. Readings are
/// sanitized first (a NaN: not seen).
pub fn helped_solve(start: &State, readings: &[Reading], help: &[usize], decode: impl Fn(&State, &[Reading]) -> DecodeResult) -> DecodeResult {
    let readings: Vec<Reading> = readings.iter().map(|r| Reading { faces: r.faces.iter().cloned().map(|f| f.sanitized()).collect(), ..r.clone() }).collect();
    let given = follow_help(start, &readings, help);
    let decoded = given.tail.as_ref().map(|(from, rest)| decode(from, rest));
    helped_end(start, &readings, given, decoded)
}

/// The moves given, placed on the readings; what is left to decode after them (None: nothing)
fn follow_help(start: &State, readings: &[Reading], help: &[usize]) -> Followed {
    let mut states = vec![*start];
    for &m in help {
        let next = apply_move(states.last().unwrap(), m);
        states.push(next);
    }
    let (k, n) = (states.len(), readings.len());
    let neighbours: Vec<Vec<State>> = states.iter().map(|s| (0..MOVES.len()).map(|m| apply_move(s, m)).collect()).collect();
    let mut margin = vec![0f64; n * k];
    for (i, reading) in readings.iter().enumerate() {
        let prepared = scorer(reading, GARBAGE);
        for (j, state) in states.iter().enumerate() {
            let next = neighbours[j].iter().fold(f64::NEG_INFINITY, |next, other| next.max(score(other, &prepared, None)));
            margin[i * k + j] = score(state, &prepared, None) - next;
        }
    }

    // best[i, j]: reading i on state j, the readings before on states up to j
    let (mut best, mut from) = (vec![0f64; n * k], vec![0isize; n * k]);
    let (mut end, mut end_state, mut total) = (-1isize, 0usize, 0f64);
    for i in 0..n {
        let (mut run, mut arg) = (if i > 0 { f64::NEG_INFINITY } else { 0.0 }, -1isize);
        for j in 0..k {
            if i > 0 && best[(i - 1) * k + j] > run {
                run = best[(i - 1) * k + j];
                arg = j as isize;
            }
            best[i * k + j] = run + margin[i * k + j];
            from[i * k + j] = arg;
            if best[i * k + j] > total {
                total = best[i * k + j];
                end = i as isize;
                end_state = j;
            }
        }
    }
    let mut on = vec![0usize; (end + 1) as usize];
    let mut j = end_state as isize;
    for i in (0..on.len()).rev() {
        on[i] = j as usize;
        j = from[i * k + j as usize];
    }

    // Each run of moves between the last reading before them and the first after
    let laid = on.len();
    let mut moves = Vec::new();
    let mut i = 0;
    let mut j = 1;
    while j < k {
        while i < laid && on[i] < j {
            i += 1;
        }
        let reached = if i < laid { on[i] } else { k - 1 };
        let count = reached - j + 1;
        let unseen = UNSEEN_MS * (count + 1) as f64;
        let before = if i < laid { readings[i].from } else { (if laid > 0 { readings[laid - 1].to } else { 0.0 }) + unseen };
        let after = if i > 0 { readings[i - 1].to } else { (before - unseen).max(0.0) };
        for q in 0..count {
            let t = js_round(after + ((before - after) * (q + 1) as f64) / (count + 1) as f64);
            moves.push(TimedMove { token: MOVES[help[j - 1 + q]].to_string(), t });
        }
        j = reached + 1;
    }

    let last = states[k - 1];
    let done = last == solved_state();
    let rest = &readings[laid..];
    Followed { moves, end, done, tail: (!done && !rest.is_empty()).then(|| (last, rest.to_vec())) }
}

fn helped_end(start: &State, readings: &[Reading], given: Followed, decoded: Option<DecodeResult>) -> DecodeResult {
    let Followed { moves, end, done, .. } = given;
    let (tail, solved, gaps) = match decoded {
        Some(d) => (d.search.moves, d.search.solved, d.gaps),
        None => (Vec::new(), done, Vec::new()),
    };
    // The decoder times a first turn from the start of the video: never before the moves given
    let floor = (if end >= 0 { readings[end as usize].to } else { 0.0 }).max(moves.last().map_or(0.0, |m| m.t));
    let all: Vec<TimedMove> = moves.into_iter().chain(tail.into_iter().map(|m| TimedMove { t: m.t.max(floor + 1.0), ..m })).collect();
    explain(start, readings, &all, solved, gaps)
}
