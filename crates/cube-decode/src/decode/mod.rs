//! The moves of a solve from the readings, after it (ChronoCube's decode.ts): the scrambled state is known, and the
//! solved one at the end. A beam goes forwards from the scramble, reading after reading, each state kept going on by
//! no turn, a turn, or a few when the time between readings allows; another goes backwards from the solved cube,
//! turns undone. The path is where they meet; when they never do (a stretch no frame saw), they are joined across the
//! gap by up to eight turns, else the moves on either side are kept and the gap told.

mod bridge;
mod pass;
mod prepare;
mod sequences;

use std::collections::HashMap;

use bridge::bridge;
use pass::{Step, chain, pass, state_at};
use prepare::{Prepared, prepare};
use sequences::{sequences, tokens};

use crate::reading::{Reading, State, TimedMove};
use crate::state::{MOVES, ROT, SearchResult, alike, apply_move, reading_score, solved_state};

/// The fastest turns of a person, ms
pub const MIN_TURN_MS: f64 = 70.0;

#[derive(Clone, Debug, PartialEq)]
pub struct DecodeOptions {
    /// States kept after each reading
    pub beam: usize,
    /// States that also try three turns in a row
    pub deep: usize,
    /// States that also try four (when the time between readings holds them)
    pub deeper: usize,
    /// Cost of a turn (log-likelihood a turn must earn)
    pub move_cost: f64,
    /// Slices as one turn of the hands (M rather than R L'), and what one costs more than a face turn
    pub slices: bool,
    pub slice_cost: f64,
    /// Probability that a face read is wrong altogether
    pub misread: f64,
    /// A view counts fully once this many ms of video back it
    pub correlation: f64,
    /// Weight of a reading unlike the one before
    pub fresh: f64,
    /// Share of the stickers two readings must agree on to be the same view
    pub alike_share: f64,
    /// No turn between two readings when nothing moved between them (Reading::moved); decoded without it too when it
    /// leaves the cube unsolved
    pub still: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions { beam: 1024, deep: 64, deeper: 8, move_cost: 1.8, slices: true, slice_cost: 2.2, misread: 0.5, correlation: 300.0, fresh: 0.5, alike_share: 0.5, still: true }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodeResult {
    pub search: SearchResult,
    /// Times (ms) where the turns could not be followed: the moves on either side are right, those between unknown
    pub gaps: Vec<f64>,
}

/// JavaScript's Math.round: halves up
pub(crate) fn js_round(x: f64) -> f64 {
    let r = x.floor();
    if x - r >= 0.5 { r + 1.0 } else { r }
}

/// The gated decode, else the free one when it beats it
fn pick(gated: DecodeResult, free: DecodeResult) -> DecodeResult {
    if gated.search.solved {
        gated
    } else if free.search.solved || free.search.agreement > gated.search.agreement {
        free
    } else {
        gated
    }
}

/// The turns from `start` to the solved cube that agree best with the readings (faces named by centre colour), with
/// their times, and how well they explain the readings. Readings are sanitized first (a NaN: not seen).
pub fn decode_solve(start: &State, readings: &[Reading], o: &DecodeOptions) -> DecodeResult {
    let readings: Vec<Reading> = readings.iter().map(|r| Reading { faces: r.faces.iter().cloned().map(|f| f.sanitized()).collect(), ..r.clone() }).collect();
    let free = DecodeOptions { still: false, ..o.clone() };
    if !o.still || !readings.iter().any(|r| r.moved == Some(false)) {
        return decode_with(start, &readings, &free);
    }
    // Both at once: the free one is needed whenever the gated one leaves the cube unsolved
    let (gated, unrestricted) = rayon::join(|| decode_with(start, &readings, o), || decode_with(start, &readings, &free));
    if gated.search.solved { gated } else { pick(gated, unrestricted) }
}

struct Plan {
    kept: Vec<Reading>,
    prepared: Vec<Prepared>,
    fits: Vec<usize>,
    weights: Vec<f64>,
}

fn decode_with(start: &State, readings: &[Reading], o: &DecodeOptions) -> DecodeResult {
    let Some(plan) = plan_decode(readings, o) else {
        return explain(start, &[], &[], false, vec![]);
    };
    let n = plan.prepared.len();
    let solved = solved_state();
    let forward_readings: Vec<&Prepared> = plan.prepared.iter().collect();
    let forward_fits: Vec<Option<usize>> = plan.fits.iter().map(|&f| Some(f)).collect();
    // Backwards: from the solved cube (a few turns after the last reading), through the readings in reverse, turns undone
    let backward_readings: Vec<&Prepared> = plan.prepared.iter().rev().collect();
    let backward_fits: Vec<Option<usize>> = std::iter::once(None).chain(plan.fits[1..].iter().rev().map(|&f| Some(f))).collect();
    let backward_weights: Vec<f64> = plan.weights.iter().rev().copied().collect();
    let (forward, reversed) = rayon::join(
        || pass(start, &forward_readings, &forward_fits, &plan.weights, o, false),
        || pass(&solved, &backward_readings, &backward_fits, &backward_weights, o, true),
    );
    debug_assert_eq!(forward.len(), n);
    join_passes(start, &plan, &forward, &reversed, o)
}

/// The readings kept and weighed; None when none shows a cell
fn plan_decode(readings: &[Reading], o: &DecodeOptions) -> Option<Plan> {
    // A reading left out (no cell seen) leaves what moved since it to the next one kept
    let mut moved = false;
    let (mut kept, mut prepared) = (Vec::new(), Vec::new());
    for r in readings {
        moved |= r.moved != Some(false);
        let Some(p) = prepare(r, o.misread) else { continue };
        kept.push(Reading { moved: r.moved.map(|_| moved), ..r.clone() });
        prepared.push(p);
        moved = false;
    }
    if prepared.is_empty() {
        return None;
    }
    // How much each reading counts: a view counts fully once `correlation` ms of video back it, a new one `fresh` at least
    let weights = kept
        .iter()
        .enumerate()
        .map(|(k, reading)| {
            let gap = reading.from - if k > 0 { kept[k - 1].to } else { 0.0 };
            let backed = if o.correlation > 0.0 { ((reading.to - reading.from + gap + 1.0) / o.correlation).min(1.0) } else { 1.0 };
            if k > 0 && alike(&kept[k - 1], reading, o.alike_share) { backed } else { backed.max(o.fresh) }
        })
        .collect();
    // Nothing moved since the reading before: no turn. A reading before the one before it fits none either (ChronoCube's
    // beam would die there)
    let fits = prepared
        .iter()
        .enumerate()
        .map(|(k, r)| {
            if o.still && k > 0 && kept[k].moved == Some(false) {
                return 0;
            }
            let fit = 1.0 + ((r.from - if k > 0 { prepared[k - 1].to } else { 0.0 }) / MIN_TURN_MS).floor();
            if fit > 0.0 { fit.min(1e9) as usize } else { 0 }
        })
        .collect();
    Some(Plan { kept, prepared, fits, weights })
}

/// Where the two passes meet, or the join across a gap: the moves
fn join_passes(start: &State, plan: &Plan, forward: &[Step], reversed: &[Step], o: &DecodeOptions) -> DecodeResult {
    let Plan { kept, prepared, fits, .. } = plan;
    let n = prepared.len();
    let solved = solved_state();
    // The backward step of reading k
    let backward = |k: usize| &reversed[n - 1 - k];

    // Where both beams hold one state at a reading: the reading's own score counted once
    let mut best: Option<(f64, usize, usize, usize)> = None;
    for (k, f) in forward.iter().enumerate() {
        let b = backward(k);
        let mut index: HashMap<i64, usize> = HashMap::with_capacity(b.hashes.len());
        for (j, &h) in b.hashes.iter().enumerate() {
            index.entry(h).or_insert(j);
        }
        for (i, h) in f.hashes.iter().enumerate() {
            let Some(&j) = index.get(h) else { continue };
            let total = f.scores[i] + b.scores[j] - f.emitted[i];
            if best.is_none_or(|(t, ..)| total > t) {
                best = Some((total, k, i, j));
            }
        }
    }
    let after = |k: isize| if k >= 0 { prepared[k as usize].to } else { 0.0 };
    let before = |k: usize| prepared[k].from;
    let mut moves: Vec<TimedMove> = Vec::new();
    // A slice's two face turns at one time
    let add = |moves: &mut Vec<TimedMove>, seq: &[usize], after: f64, before: f64| {
        for (j, &token) in seq.iter().enumerate() {
            let t = js_round(after + ((before - after) * (j + 1) as f64) / seq.len() as f64);
            for &m in &tokens()[token] {
                moves.push(TimedMove { token: MOVES[m].to_string(), t });
            }
        }
    };
    let forward_moves = |moves: &mut Vec<TimedMove>, k: usize, i: usize| {
        for (step, len, seq) in chain(forward, k, i) {
            if len > 0 {
                add(moves, &sequences(len, o.slices).moves[seq], after(step as isize - 1), before(step));
            }
        }
    };
    // Backwards, the step of reading k holds the turns made between reading k and the next (the end after the last)
    let backward_moves = |moves: &mut Vec<TimedMove>, k: usize, j: usize| {
        for (step, len, seq) in chain(reversed, n - 1 - k, j).into_iter().rev() {
            if len == 0 {
                continue;
            }
            let r = n - 1 - step;
            let end = if r + 1 < n { before(r + 1) } else { prepared[n - 1].to + 150.0 * len as f64 };
            add(moves, &sequences(len, o.slices).moves[seq], after(r as isize), end);
        }
    };
    if let Some((_, k, i, j)) = best {
        forward_moves(&mut moves, k, i);
        backward_moves(&mut moves, k, j);
        return explain(start, kept, &moves, true, vec![]);
    }

    // The beams never meet: a stretch no frame saw. Join them across the longest gaps, within eight turns (and those
    // the gap's time holds), the gaps that could score most first: one that cannot beat the join found is not tried
    let top_score = |k: usize| forward[k - 1].scores[0] + backward(k).scores[0];
    let mut gap_order: Vec<usize> = (1..n).collect();
    let gap = |k: usize| prepared[k].from - prepared[k - 1].to;
    gap_order.sort_by(|&a, &b| gap(b).partial_cmp(&gap(a)).unwrap_or(std::cmp::Ordering::Equal));
    gap_order.truncate(6);
    gap_order.sort_by(|&a, &b| top_score(b).partial_cmp(&top_score(a)).unwrap_or(std::cmp::Ordering::Equal));
    let mut joined: Option<(bridge::Bridge, usize)> = None;
    for k in gap_order {
        let floor = joined.as_ref().map_or(f64::NEG_INFINITY, |(b, _)| b.total);
        if top_score(k) <= floor {
            continue;
        }
        let top = |step: &Step| step.scores.len().min(16);
        let starts: Vec<State> = (0..top(&forward[k - 1])).map(|i| state_at(start, forward, k - 1, i, false, o.slices)).collect();
        let ends: Vec<State> = (0..top(backward(k))).map(|j| state_at(&solved, reversed, n - 1 - k, j, true, o.slices)).collect();
        let met = bridge(&starts, &forward[k - 1].scores[..starts.len()], &ends, &backward(k).scores[..ends.len()], o, fits[k].min(8), floor);
        if let Some(met) = met
            && joined.as_ref().is_none_or(|(b, _)| met.total > b.total)
        {
            joined = Some((met, k));
        }
    }
    if let Some((met, k)) = joined {
        forward_moves(&mut moves, k - 1, met.a);
        let both: Vec<usize> = met.first.iter().chain(&met.second).copied().collect();
        add(&mut moves, &both, after(k as isize - 1), before(k));
        backward_moves(&mut moves, k, met.b);
        return explain(start, kept, &moves, true, vec![]);
    }
    // Not even so: the turns before the gap where the two beams are best and after it, the gap left unknown
    if n == 1 {
        forward_moves(&mut moves, 0, 0);
        return explain(start, kept, &moves, false, vec![]);
    }
    let mut k = 1;
    for c in 2..n {
        if top_score(c) > top_score(k) {
            k = c;
        }
    }
    forward_moves(&mut moves, k - 1, 0);
    backward_moves(&mut moves, k, 0);
    explain(start, kept, &moves, false, vec![prepared[k].from])
}

/// The states after each reading as the moves leave them, and the share of the stickers read they explain
pub fn explain(start: &State, readings: &[Reading], moves: &[TimedMove], solved: bool, gaps: Vec<f64>) -> DecodeResult {
    let mut state = *start;
    let (mut m, mut agree, mut total) = (0, 0usize, 0usize);
    let mut path = Vec::with_capacity(readings.len());
    for reading in readings {
        while m < moves.len() && moves[m].t <= reading.from {
            let mv = MOVES.iter().position(|&x| x == moves[m].token).expect("a face turn");
            state = apply_move(&state, mv);
            m += 1;
        }
        let (_, rotations) = reading_score(&state, reading);
        for (f, face) in reading.faces.iter().enumerate() {
            for (i, p) in face.stickers.iter().enumerate() {
                let Some(p) = p else { continue };
                if i == 4 {
                    continue;
                }
                total += 1;
                let top = p.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                if p[state[face.center as usize * 9 + ROT[rotations[f]][i]] as usize] >= top {
                    agree += 1;
                }
            }
        }
        path.push((state, rotations));
    }
    let agreement = if total > 0 { agree as f64 / total as f64 } else { 0.0 };
    DecodeResult { search: SearchResult { moves: moves.to_vec(), path, agreement, solved }, gaps }
}
