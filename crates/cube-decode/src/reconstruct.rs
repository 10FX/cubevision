//! A solve turned into cubing terms (ChronoCube's smartcube/analysis.ts): CFOP splits, moves, TPS, pauses

use cube_core::{Cube, FACES, cfop_progress};

use crate::reading::TimedMove;

/// Colours of the 54 stickers, U R F D L B (the Kociemba facelet order)
pub fn facelets(cube: &Cube) -> String {
    cube.stickers().iter().map(|&s| FACES[s as usize / 9]).collect()
}

/// Same-face turns in a row become one: R R = R2, R R' = nothing
pub fn collapse(moves: &[String]) -> Vec<String> {
    let mut out: Vec<(char, u32)> = Vec::new();
    for m in moves {
        let face = m.chars().next().unwrap_or(' ');
        let turns = quarter_turns(m);
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
    out.into_iter().map(|(face, turns)| format!("{face}{}", suffix(turns))).collect()
}

/// Tokens in a row on the same face or axis become one (R R = R2, y y' = nothing); for any notation
pub fn collapse_tokens(tokens: &[String]) -> Vec<String> {
    let mut out: Vec<(&str, u32)> = Vec::new();
    for token in tokens {
        let key = token.trim_end_matches(['\'', '2']);
        let turns = quarter_turns(token);
        match out.last_mut() {
            Some(last) if last.0 == key => {
                last.1 = (last.1 + turns) % 4;
                if last.1 == 0 {
                    out.pop();
                }
            }
            _ => out.push((key, turns)),
        }
    }
    out.into_iter().map(|(key, turns)| format!("{key}{}", suffix(turns))).collect()
}

fn quarter_turns(token: &str) -> u32 {
    if token.ends_with('2') {
        2
    } else if token.ends_with('\'') {
        3
    } else {
        1
    }
}

fn suffix(turns: u32) -> &'static str {
    match turns {
        2 => "2",
        3 => "'",
        _ => "",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Cross,
    F2l1,
    F2l2,
    F2l3,
    F2l4,
    Oll,
    Pll,
}

pub const PHASES: [Phase; 7] = [Phase::Cross, Phase::F2l1, Phase::F2l2, Phase::F2l3, Phase::F2l4, Phase::Oll, Phase::Pll];

impl Phase {
    /// As ChronoCube names it
    pub fn name(self) -> &'static str {
        match self {
            Phase::Cross => "cross",
            Phase::F2l1 => "f2l1",
            Phase::F2l2 => "f2l2",
            Phase::F2l3 => "f2l3",
            Phase::F2l4 => "f2l4",
            Phase::Oll => "oll",
            Phase::Pll => "pll",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Split {
    pub phase: Phase,
    /// ms from the first move to the end of the phase
    pub end: f64,
    pub duration: f64,
    /// Pause before the first turn of the phase (recognition)
    pub recognition: f64,
    /// Face turns in the phase (outer block metric, R R counted as one R2)
    pub moves: u32,
    pub alg: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reconstruction {
    pub moves: u32,
    /// Turns per second over the whole solve
    pub tps: f64,
    pub time: f64,
    pub splits: Vec<Split>,
    pub solution: String,
    /// Time spent still between turns (gaps of PAUSE ms or more), ms
    pub pause: f64,
    /// Time of the first turn, in the clock of the moves
    pub start: f64,
}

/// A gap between two turns this long is a pause (looking for the next pieces)
pub const PAUSE: f64 = 500.0;

/// Splits of a solve from the scrambled state and the timed turns of the solve. `written`: the tokens to show for each
/// turn when the moves are written otherwise (the cuber's rotations, wide moves, slices); the counts stay the turns'.
pub fn reconstruct(start: &Cube, solve: &[TimedMove], written: Option<&[Vec<String>]>) -> Reconstruction {
    let mut cube = start.clone();
    let mut reached = cfop_progress(&cube);
    let t0 = solve.first().map_or(0.0, |m| m.t);
    let mut ends: Vec<(Phase, usize)> = Vec::new();
    for (i, m) in solve.iter().enumerate() {
        cube = cube.apply(&m.token);
        let progress = cfop_progress(&cube);
        // Several steps can end on one turn (two pairs at once, an OLL skip)
        for k in reached + 1..=progress {
            ends.push((PHASES[k as usize - 1], i));
        }
        reached = reached.max(progress);
    }

    let tokens = |from: usize, to: usize| -> Vec<String> { solve[from..to].iter().map(|m| m.token.clone()).collect() };
    let mut splits = Vec::with_capacity(ends.len());
    let mut from = 0;
    for (phase, index) in ends {
        let turns = collapse(&tokens(from, index + 1));
        let start_t = if from == 0 { t0 } else { solve[from - 1].t };
        let alg = match written {
            Some(w) => collapse_tokens(&w[from..=index].concat()).join(" "),
            None => turns.join(" "),
        };
        splits.push(Split {
            phase,
            end: solve[index].t - t0,
            duration: solve[index].t - start_t,
            recognition: if from < solve.len() && from > 0 { (solve[from].t - solve[from - 1].t).max(0.0) } else { 0.0 },
            moves: turns.len() as u32,
            alg,
        });
        from = index + 1;
    }
    let time = solve.last().map_or(0.0, |m| m.t - t0);
    let all = collapse(&tokens(0, solve.len()));
    let pause = solve.windows(2).fold(0.0, |sum, w| {
        let gap = w[1].t - w[0].t;
        if gap >= PAUSE { sum + gap } else { sum }
    });
    let solution = match written {
        Some(w) => collapse_tokens(&w.concat()).join(" "),
        None => all.join(" "),
    };
    let moves = all.len() as u32;
    Reconstruction { moves, tps: if time > 0.0 { moves as f64 / (time / 1000.0) } else { 0.0 }, time, splits, solution, pause, start: t0 }
}
