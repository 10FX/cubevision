//! One beam through the readings, forwards from the scramble or backwards from the solved cube

use super::DecodeOptions;
use super::prepare::{Prepared, bound, likelihood_above, rest_bounds};
use super::sequences::{Sequences, Table, sequences, zobrist, zobrist_table};
use crate::reading::State;

/// The states kept at a reading, best first: the score so far, the reading's own part, how each came
pub struct Step {
    /// Only the last step's states are kept (the others are rebuilt from their turns when needed)
    pub states: Vec<u8>,
    pub hashes: Vec<i64>,
    pub scores: Vec<f64>,
    pub emitted: Vec<f64>,
    pub parent: Vec<u32>,
    pub length: Vec<u8>,
    pub sequence: Vec<u32>,
}

/// A beam through `readings` from `start`: before each reading, `fits[k]` turns fit in the time, up to four (None:
/// up to three); `backwards`: the turns are undone (from the solved cube, readings reversed).
///
/// Candidates come best first by their score so far (each state kept, sorted, with no turn, one, two…); a
/// candidate's hash is its parent's with the terms of the stickers turned changed, and a state met again came a
/// worse way. The reading scores them until none can enter the beam.
pub fn pass(
    start: &State,
    readings: &[&Prepared],
    fits: &[Option<usize>],
    weights: &[f64],
    o: &DecodeOptions,
    backwards: bool,
) -> Vec<Step> {
    let keep = (1.0 - o.misread).ln();
    let seqs: Vec<&Sequences> = (0..=4).map(|len| sequences(len, o.slices)).collect();
    let z = zobrist_table();
    // How many of the states kept go on by each number of turns
    let tries = |len: usize, n: usize| match len {
        4 => n.min(o.deeper),
        3 => n.min(o.deep),
        _ => n,
    };
    let mut states: Vec<u8> = start.to_vec();
    let mut hashes: Vec<i64> = vec![zobrist(start)];
    let mut scores: Vec<f64> = vec![0.0];
    let mut steps: Vec<Step> = Vec::with_capacity(readings.len());
    let mut table = Table::default();
    let (mut base, mut hash, mut parent, mut length, mut sequence) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut totals, mut emitted_of): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for (k, reading) in readings.iter().enumerate() {
        let top = fits[k].map_or(3, |f| f.min(4));
        let n = scores.len();
        let count: usize = (0..=top).map(|len| tries(len, n) * seqs[len].moves.len()).sum();
        if count > base.len() {
            base.resize(count, 0.0);
            hash.resize(count, 0);
            parent.resize(count, 0u32);
            length.resize(count, 0u8);
            sequence.resize(count, 0u32);
            totals.resize(count, 0.0);
            emitted_of.resize(count, 0.0);
        }
        table.reset(count);
        let most = weights[k] * bound(reading, keep);
        let margin = 1e-3 + 1e-6 * most.abs();
        let rest = rest_bounds(reading, keep);
        // The beam as a min-heap on the total: its root is the last state kept
        // (total, candidate): the totals beside the indexes, not looked up in the candidates' array at each comparison
        let mut heap: Vec<(f64, u32)> = Vec::with_capacity(o.beam);
        let mut next = [0usize; 5];
        let ends: [usize; 5] = std::array::from_fn(|len| if len > top { 0 } else { tries(len, n) });
        let mut used = 0usize;
        let mut group: Vec<i64> = Vec::new();
        loop {
            let (mut len, mut so) = (usize::MAX, f64::NEG_INFINITY);
            for l in 0..=top {
                if next[l] >= ends[l] {
                    continue;
                }
                let v = scores[next[l]] - o.move_cost * l as f64;
                if v > so {
                    so = v;
                    len = l;
                }
            }
            if len == usize::MAX || (heap.len() >= o.beam && so + most <= heap[0].0) {
                break;
            }
            let a = next[len];
            next[len] += 1;
            let at = a * 54;
            let q = seqs[len];
            let gather = if backwards { &q.inverse } else { &q.perms };
            let pairs = if backwards { &q.back_pairs } else { &q.pairs };
            let state: &[u8; 54] = states[at..at + 54].try_into().unwrap();
            // Each sticker's term in the parent's hash, once for all its sequences
            let own: [i64; 54] = std::array::from_fn(|i| z[i * 6 + state[i] as usize]);
            // The hashes of the group first, their slots fetched meanwhile: the probes then find them in the cache
            group.clear();
            for s in 0..q.moves.len() {
                let mut h = hashes[a];
                for &(i, from) in &pairs[q.offsets[s]..q.offsets[s + 1]] {
                    let i = i as usize;
                    h += z[i * 6 + state[from as usize] as usize] - own[i];
                }
                table.prefetch(h);
                group.push(h);
            }
            for (s, &h) in group.iter().enumerate() {
                let cost = so - o.slice_cost * q.slices[s] as f64;
                let slot = table.slot(h);
                if table.has(slot) {
                    continue;
                }
                let c = used;
                used += 1;
                table.set(slot, h, c as u32);
                hash[c] = h;
                base[c] = cost;
                parent[c] = a as u32;
                length[c] = len as u8;
                sequence[c] = s as u32;
                // A state that cannot enter the beam even read at best is not scored: it would be left out anyway
                // (the margin covers the float rounding of the reading's logs, so as to leave out no state ChronoCube keeps)
                if heap.len() >= o.beam && cost + most + margin <= heap[0].0 {
                    continue;
                }
                let gathered = &gather[s * 54..s * 54 + 54];
                // Scored face by face, given up once it cannot enter the beam
                let need = if heap.len() >= o.beam { heap[0].0 - cost - margin } else { f64::NEG_INFINITY };
                let Some(likely) = likelihood_above(|i| state[gathered[i] as usize], reading, keep, weights[k], need, &rest) else {
                    continue;
                };
                emitted_of[c] = weights[k] * likely;
                totals[c] = cost + emitted_of[c];
                let less = |heap: &[(f64, u32)], x: usize, y: usize| heap[x].0 < heap[y].0;
                if heap.len() >= o.beam {
                    if totals[c] <= heap[0].0 {
                        continue;
                    }
                    // Replace the root and sift it down
                    heap[0] = (totals[c], c as u32);
                    let mut x = 0;
                    loop {
                        let (l, r) = (2 * x + 1, 2 * x + 2);
                        let mut min = x;
                        if l < heap.len() && less(&heap, l, min) {
                            min = l;
                        }
                        if r < heap.len() && less(&heap, r, min) {
                            min = r;
                        }
                        if min == x {
                            break;
                        }
                        heap.swap(x, min);
                        x = min;
                    }
                } else {
                    heap.push((totals[c], c as u32));
                    let mut x = heap.len() - 1;
                    while x > 0 {
                        let up = (x - 1) >> 1;
                        if !less(&heap, x, up) {
                            break;
                        }
                        heap.swap(x, up);
                        x = up;
                    }
                }
            }
        }
        // Best first; equal totals keep the heap's order (a stable sort, as JavaScript's)
        heap.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        let m = heap.len();
        let mut step = Step {
            states: vec![0; m * 54],
            hashes: Vec::with_capacity(m),
            scores: Vec::with_capacity(m),
            emitted: Vec::with_capacity(m),
            parent: Vec::with_capacity(m),
            length: Vec::with_capacity(m),
            sequence: Vec::with_capacity(m),
        };
        for (j, &(_, c)) in heap.iter().enumerate() {
            let c = c as usize;
            let q = seqs[length[c] as usize];
            let gather = if backwards { &q.inverse } else { &q.perms };
            let (from, seq) = (parent[c] as usize * 54, sequence[c] as usize * 54);
            for i in 0..54 {
                step.states[j * 54 + i] = states[from + gather[seq + i] as usize];
            }
            step.hashes.push(hash[c]);
            step.scores.push(totals[c]);
            step.emitted.push(emitted_of[c]);
            step.parent.push(parent[c]);
            step.length.push(length[c]);
            step.sequence.push(sequence[c]);
        }
        states = std::mem::take(&mut step.states);
        hashes = step.hashes.clone();
        scores = step.scores.clone();
        step.states = states.clone();
        // Only the states of the last step are needed again (the bridge rebuilds its own)
        if let Some(previous) = steps.last_mut() {
            previous.states = Vec::new();
        }
        steps.push(step);
    }
    steps
}

/// (step, length, sequence) of the turns that led to state `index` at step `k`, from the first step
pub fn chain(steps: &[Step], k: usize, mut index: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::with_capacity(k + 1);
    for i in (0..=k).rev() {
        out.push((i, steps[i].length[index] as usize, steps[i].sequence[index] as usize));
        index = steps[i].parent[index] as usize;
    }
    out.reverse();
    out
}

/// The state `index` of step `k`, its turns replayed from the start
pub fn state_at(start: &State, steps: &[Step], k: usize, index: usize, backwards: bool, slices: bool) -> State {
    let mut s = *start;
    for (_, len, seq) in chain(steps, k, index) {
        if len == 0 {
            continue;
        }
        let q = sequences(len, slices);
        let gather = if backwards { &q.inverse } else { &q.perms };
        s = std::array::from_fn(|i| s[gather[seq * 54 + i] as usize]);
    }
    s
}
