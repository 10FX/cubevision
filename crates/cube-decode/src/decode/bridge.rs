//! Turns joining the two beams across a stretch no frame saw

use super::DecodeOptions;
use super::sequences::{Sequences, Table, sequences, zobrist, zobrist_table};
use crate::reading::State;

pub struct Bridge {
    pub total: f64,
    /// Index of the state before the gap, and of the one after
    pub a: usize,
    pub b: usize,
    /// Turns from the state before, then those to the state after
    pub first: Vec<usize>,
    pub second: Vec<usize>,
}

/// Turns joining one of the best states before a gap to one of the best after (up to four from each side, `most`
/// in all), scoring above `floor`
pub fn bridge(
    before: &[State],
    before_scores: &[f64],
    after: &[State],
    after_scores: &[f64],
    o: &DecodeOptions,
    most: usize,
    floor: f64,
) -> Option<Bridge> {
    let (move_cost, slice_cost) = (o.move_cost, o.slice_cost);
    let z = zobrist_table();
    let depth = most.min(4);
    let seqs: Vec<&Sequences> = (0..=depth).map(|len| sequences(len, o.slices)).collect();
    let count = after.len() * seqs.iter().map(|q| q.moves.len()).sum::<usize>();
    // From after: the states up to four turns before it (hashed from its own, over the stickers turned), the best way to each
    let mut reach = Table::default();
    reach.reset(count);
    let (mut total, mut from, mut length, mut sequence) =
        (vec![0f64; count], vec![0usize; count], vec![0usize; count], vec![0usize; count]);
    let mut used = 0usize;
    for (b, state) in after.iter().enumerate() {
        let h0 = zobrist(state);
        for (len, q) in seqs.iter().enumerate() {
            for s in 0..q.moves.len() {
                let score = after_scores[b] - move_cost * len as f64 - slice_cost * q.slices[s] as f64;
                let mut h = h0;
                for &i in &q.moved[q.offsets[s]..q.offsets[s + 1]] {
                    let i = i as usize;
                    h += z[i * 6 + state[q.inverse[s * 54 + i] as usize] as usize] - z[i * 6 + state[i] as usize];
                }
                let slot = reach.slot(h);
                let c;
                if reach.has(slot) {
                    c = reach.value(slot) as usize;
                    if total[c] >= score {
                        continue;
                    }
                } else {
                    c = used;
                    used += 1;
                    reach.set(slot, h, c as u32);
                }
                total[c] = score;
                from[c] = b;
                length[c] = len;
                sequence[c] = s;
            }
        }
    }
    let mut best: Option<Bridge> = None;
    // No way through scores more than the best after state without a turn: a start that cannot beat `floor` is skipped
    let reach_best = after_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    for (a, state) in before.iter().enumerate() {
        let h0 = zobrist(state);
        for (len, q) in seqs.iter().enumerate() {
            if before_scores[a] - move_cost * len as f64 + reach_best <= floor.max(best.as_ref().map_or(f64::NEG_INFINITY, |b| b.total)) {
                break;
            }
            for s in 0..q.moves.len() {
                let mut h = h0;
                for &i in &q.moved[q.offsets[s]..q.offsets[s + 1]] {
                    let i = i as usize;
                    h += z[i * 6 + state[q.perms[s * 54 + i] as usize] as usize] - z[i * 6 + state[i] as usize];
                }
                let slot = reach.slot(h);
                if !reach.has(slot) {
                    continue;
                }
                let c = reach.value(slot) as usize;
                let sum = before_scores[a] + total[c] - move_cost * len as f64 - slice_cost * q.slices[s] as f64;
                if len + length[c] > most || sum <= floor || best.as_ref().is_some_and(|b| sum <= b.total) {
                    continue;
                }
                // One hash, but the same cube? (two cubes may share a hash, rarely)
                let back = &seqs[length[c]].inverse;
                let (at, there) = (sequence[c] * 54, &after[from[c]]);
                if !(0..54).all(|i| state[q.perms[s * 54 + i] as usize] == there[back[at + i] as usize]) {
                    continue;
                }
                best = Some(Bridge {
                    total: sum,
                    a,
                    b: from[c],
                    first: q.moves[s].clone(),
                    second: seqs[length[c]].moves[sequence[c]].clone(),
                });
            }
        }
    }
    best
}
