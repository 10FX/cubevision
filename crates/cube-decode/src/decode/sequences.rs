//! Turn sequences as gathers of the 54 stickers, Zobrist hashes, and a table from hashes to indexes

use std::sync::OnceLock;

use crate::state::{MOVES, apply_move};

/// Sequences up to this length take slices too (longer ones would hold too many)
pub const SLICE_DEPTH: usize = 3;

/// What the hands do in one turn: the 18 face turns, then the 9 slices. A slice turns the cube with it, so in the
/// frame of the centres it is the two faces of its axis turned the other way (M = R L', E = U D', S = F' B…)
pub fn tokens() -> &'static [Vec<usize>] {
    static TOKENS: OnceLock<Vec<Vec<usize>>> = OnceLock::new();
    TOKENS.get_or_init(|| {
        let mut all: Vec<Vec<usize>> = (0..MOVES.len()).map(|m| vec![m]).collect();
        for axis in 0..3 {
            for k in 0..3 {
                all.push(vec![axis * 3 + k, (axis + 3) * 3 + 2 - k]);
            }
        }
        all
    })
}

fn axis_of(t: usize) -> usize {
    tokens()[t][0] / 3 % 3
}

fn is_slice(t: usize) -> bool {
    tokens()[t].len() > 1
}

/// Where each sticker comes from after a turn of the hands (the two faces of a slice commute)
fn token_sources() -> &'static [[u8; 54]] {
    static SOURCES: OnceLock<Vec<[u8; 54]>> = OnceLock::new();
    SOURCES.get_or_init(|| {
        let identity: [u8; 54] = std::array::from_fn(|i| i as u8);
        tokens().iter().map(|t| t.iter().fold(identity, |acc, &m| apply_move(&acc, m))).collect()
    })
}

pub struct Sequences {
    /// Turns of the hands of each sequence (tokens())
    pub moves: Vec<Vec<usize>>,
    /// Slices in each sequence
    pub slices: Vec<u8>,
    /// state'[i] = state[perms[s * 54 + i]] after the turns, and the state before them from the state after
    pub perms: Vec<u8>,
    pub inverse: Vec<u8>,
    /// The stickers the turns move: sequence s moves moved[offsets[s]..offsets[s + 1]]
    pub moved: Vec<u8>,
    pub offsets: Vec<usize>,
    /// (sticker, where it comes from) for each sticker moved, forwards (perms) and backwards (inverse), in `offsets`
    pub pairs: Vec<(u8, u8)>,
    pub back_pairs: Vec<(u8, u8)>,
}

/// Sequences of `length` turns (`slices`: slices too, up to SLICE_DEPTH), without two turns of one face in a row,
/// both orders of opposite faces, nor a slice next to a turn of its axis: each of the sequences one turn shorter
/// followed by a turn (their order kept)
pub fn sequences(length: usize, slices: bool) -> &'static Sequences {
    static CACHE: [[OnceLock<Sequences>; 5]; 2] = [const { [const { OnceLock::new() }; 5] }; 2];
    assert!(length <= 4, "sequences of up to four turns");
    let with_slices = slices && length <= SLICE_DEPTH;
    CACHE[with_slices as usize][length].get_or_init(|| build(length, with_slices))
}

fn build(length: usize, with_slices: bool) -> Sequences {
    let identity: Vec<u8> = (0..54).collect();
    if length == 0 {
        return Sequences {
            moves: vec![vec![]],
            slices: vec![0],
            perms: identity.clone(),
            inverse: identity,
            moved: vec![],
            offsets: vec![0, 0],
            pairs: vec![],
            back_pairs: vec![],
        };
    }
    let shorter = sequences(length - 1, with_slices);
    let count = tokens().len() - if with_slices { 0 } else { 9 };
    let (mut all, mut parents) = (Vec::new(), Vec::new());
    for (s, seq) in shorter.moves.iter().enumerate() {
        for t in 0..count {
            if let Some(&last) = seq.last() {
                let (face, before) = (t / 3, last / 3);
                if axis_of(t) == axis_of(last) && (is_slice(t) || is_slice(last)) {
                    continue;
                }
                // Faces U R F D L B: opposite faces share face % 3; U D is kept, D U is the same turns
                if !is_slice(t) && !is_slice(last) && (face == before || (face % 3 == before % 3 && before > face)) {
                    continue;
                }
            }
            let mut next = seq.clone();
            next.push(t);
            all.push(next);
            parents.push(s);
        }
    }
    let (mut perms, mut inverse) = (vec![0u8; all.len() * 54], vec![0u8; all.len() * 54]);
    let (mut moved, mut offsets) = (Vec::new(), vec![0usize; all.len() + 1]);
    let slices = all.iter().map(|seq| seq.iter().filter(|&&t| is_slice(t)).count() as u8).collect();
    for (s, seq) in all.iter().enumerate() {
        let source = &token_sources()[*seq.last().unwrap()];
        let (from, at) = (parents[s] * 54, s * 54);
        for i in 0..54 {
            let p = shorter.perms[from + source[i] as usize];
            perms[at + i] = p;
            inverse[at + p as usize] = i as u8;
            if p as usize != i {
                moved.push(i as u8);
            }
        }
        offsets[s + 1] = moved.len();
    }
    let pairs_of = |gather: &[u8]| -> Vec<(u8, u8)> {
        (0..all.len()).flat_map(|s| moved[offsets[s]..offsets[s + 1]].iter().map(move |&i| (i, gather[s * 54 + i as usize]))).collect()
    };
    let (pairs, back_pairs) = (pairs_of(&perms), pairs_of(&inverse));
    Sequences { moves: all, slices, perms, inverse, moved, offsets, pairs, back_pairs }
}

/// Zobrist hashing: a state's hash is the sum of a random number per (sticker, colour), 46 bits each (sums of 54
/// stay exact, as in ChronoCube's doubles). A turn changes only the terms of the stickers it moves.
pub fn zobrist_table() -> &'static [i64; 324] {
    static ZOBRIST: OnceLock<[i64; 324]> = OnceLock::new();
    ZOBRIST.get_or_init(|| {
        // mulberry32 from 0x9e3779b9
        let mut seed = 0x9e3779b9u32;
        let mut next = || {
            seed = seed.wrapping_add(0x6d2b79f5);
            let mut t = (seed ^ (seed >> 15)).wrapping_mul(seed | 1);
            t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
            t ^ (t >> 14)
        };
        std::array::from_fn(|_| {
            let high = (next() & 0x3fffff) as i64;
            high * (1 << 24) + (next() & 0xffffff) as i64
        })
    })
}

pub fn zobrist(state: &[u8]) -> i64 {
    let z = zobrist_table();
    (0..54).map(|i| z[i * 6 + state[i] as usize]).sum()
}

/// An open-addressing table from hashes to indexes, emptied in constant time between uses. It grows with what it
/// holds rather than with what it might (a beam's candidates are mostly met again or never stored): a small table
/// stays in the cache. Each slot holds its key, value and stamp together, one cache line read per probe.
#[derive(Default)]
pub struct Table {
    slots: Vec<Slot>,
    stamp: u32,
    mask: usize,
    len: usize,
}

#[derive(Clone, Copy, Default)]
struct Slot {
    key: i64,
    value: u32,
    stamp: u32,
}

const SMALLEST: usize = 1 << 12;

impl Table {
    /// Empty (`_count`, the most keys it may get, is not reserved: the table grows when needed)
    pub fn reset(&mut self, _count: usize) {
        if self.slots.is_empty() {
            self.slots = vec![Slot::default(); SMALLEST];
            self.mask = SMALLEST - 1;
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.slots.fill(Slot::default());
            self.stamp = 1;
        }
        self.len = 0;
    }

    /// The slot of a key: its value is `value(slot)` when `has(slot)`
    #[inline]
    pub fn slot(&self, key: i64) -> usize {
        let mut i = key as usize & self.mask;
        loop {
            let slot = &self.slots[i];
            if slot.stamp != self.stamp || slot.key == key {
                return i;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Start loading a key's slot into the cache (a probe soon after then rarely waits on memory)
    #[inline]
    pub fn prefetch(&self, key: i64) {
        #[cfg(target_arch = "x86_64")]
        {
            let at: *const Slot = &self.slots[key as usize & self.mask];
            // SAFETY: a prefetch is a hint, it reads nothing and cannot fault; the pointer is to a live slot
            #[allow(unused_unsafe)]
            unsafe {
                std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(at.cast());
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
    }

    #[inline]
    pub fn has(&self, slot: usize) -> bool {
        self.slots[slot].stamp == self.stamp
    }

    #[inline]
    pub fn value(&self, slot: usize) -> u32 {
        self.slots[slot].value
    }

    /// Store at the slot `slot(key)` gave; a new key may grow the table (the slot found is then looked up again)
    #[inline]
    pub fn set(&mut self, mut slot: usize, key: i64, value: u32) {
        if !self.has(slot) {
            if (self.len + 1) * 2 > self.slots.len() {
                self.grow();
                slot = self.slot(key);
            }
            self.len += 1;
        }
        self.slots[slot] = Slot { key, value, stamp: self.stamp };
    }

    #[cold]
    fn grow(&mut self) {
        let old = std::mem::take(&mut self.slots);
        let stamp = self.stamp;
        self.slots = vec![Slot::default(); old.len() * 2];
        self.mask = self.slots.len() - 1;
        self.stamp = 1;
        for s in old.into_iter().filter(|s| s.stamp == stamp) {
            let i = self.slot(s.key);
            self.slots[i] = Slot { stamp: 1, ..s };
        }
    }
}
