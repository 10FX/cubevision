//! A solve written as a cuber writes a reconstruction (ChronoCube's smartcube/present.ts): the cube turned at the
//! start so that its cross is down (z2 for a white cross scrambled white on top, green still in front), and before a
//! step a y rotation when it makes the step's turns R and U ones. The turns come in the frame of the centres (the hold
//! that scrambled); only their letters change.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cube_core::{Cube, FACES, ORIENTATIONS, cfop_progress, cross, js_space, solved_slots};

/// How far a face is from the hands, per turn: R and U are free, B is awkward
fn face_cost(face: char) -> f64 {
    match face {
        'R' | 'U' => 0.0,
        'F' => 0.7,
        'L' => 0.8,
        'D' => 0.9,
        _ => 1.6,
    }
}

const Y_TOKENS: [&str; 4] = ["", "y", "y2", "y'"];
/// A regrip before a step, against the turns it saves
const Y_COST: [f64; 4] = [0.0, 1.2, 1.8, 1.2];
/// The D face and the white centre's colour, in the cube's order (U R F D L B, white on U)
const FACE_D: usize = 3;
const WHITE: u8 = 0;

/// Where each face (U R F D L B, letters in the scrambling hold) is once the cube is turned by `rotation`
pub fn face_map(rotation: &str) -> [char; 6] {
    static MAPS: OnceLock<Mutex<HashMap<String, [char; 6]>>> = OnceLock::new();
    let maps = MAPS.get_or_init(Default::default);
    if let Some(map) = maps.lock().unwrap().get(rotation) {
        return *map;
    }
    let turned = Cube::solved().apply(rotation);
    let map = std::array::from_fn(|colour| FACES[(0..6).find(|&f| turned.color(f, 1, 1) as usize == colour).unwrap()]);
    maps.lock().unwrap().insert(rotation.to_string(), map);
    map
}

fn face_index(face: char) -> usize {
    FACES.iter().position(|&f| f == face).unwrap()
}

/// A face turn (scrambling hold) written once the cube is turned by `rotation`
pub fn turned_move(m: &str, rotation: &str) -> String {
    let mut chars = m.chars();
    let face = chars.next().unwrap();
    format!("{}{}", face_map(rotation)[face_index(face)], chars.as_str())
}

/// The rotation that holds the cube with its cross down, from the scrambled cube and the turns: the first cross made
/// that the F2L goes on with, green (else white) kept in front; without a cross made, the white one. `colour`: the
/// colour of the cross the cuber solves (a centre colour, 0 white), down whatever was made.
pub fn cross_down(start: &Cube, moves: &[String], colour: Option<u8>) -> String {
    let mut cubes = vec![start.clone()];
    for m in moves {
        let next = cubes.last().unwrap().apply(m);
        cubes.push(next);
    }
    let down_is = |c: u8| ORIENTATIONS.iter().copied().filter(|o| start.apply(o).color(FACE_D, 1, 1) == c).collect::<Vec<_>>();
    if let Some(c) = colour {
        return held_down(&cubes, &down_is(c));
    }
    // The cross the F2L goes on with: the first one made with two pairs, else one, else the first made. A cross of
    // another colour made before it (by chance, or a turn misread) is not it; nor all the crosses of the cube solved.
    for pairs in [2, 1, 0] {
        let made = |c: &Cube, o: &str| {
            let held = c.apply(o);
            cross(&held, true) && solved_slots(&held, true) >= pairs
        };
        let at = cubes.iter().position(|c| (pairs == 0 || cfop_progress(c) < 7) && ORIENTATIONS.iter().any(|o| made(c, o)));
        if let Some(at) = at {
            let holds: Vec<&str> = ORIENTATIONS.iter().copied().filter(|o| made(&cubes[at], o)).collect();
            return held_down(&cubes, &holds);
        }
    }
    // No cross made (the turns lost before it): the white one, the most made, rather than the cube as scrambled
    held_down(&cubes, &down_is(WHITE))
}

/// Of the holds with a cross down, the one written
fn held_down(cubes: &[Cube], holds: &[&str]) -> String {
    // Two crosses at once (rare): the one with the most pairs at the end
    let down = |o: &str| FACES[face_map(o).iter().position(|&to| to == 'D').unwrap()];
    let last = cubes.last().unwrap();
    let mut ranked = holds.to_vec();
    ranked.sort_by_key(|o| std::cmp::Reverse(solved_slots(&last.apply(o), true)));
    let same: Vec<&str> = holds.iter().copied().filter(|&o| down(o) == down(ranked[0])).collect();
    let chosen = same.iter().find(|o| face_map(o)[2] == 'F').or_else(|| same.iter().find(|o| face_map(o)[0] == 'F')).unwrap_or(&same[0]);
    simplest(chosen)
}

/// Names of the rotations, shortest first: each hold gets the simplest ("z2", not "x2 y2")
fn names() -> &'static [String] {
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut all: Vec<String> = ["", "x", "x'", "x2", "y", "y'", "y2", "z", "z'", "z2"].map(String::from).to_vec();
        for r in ["x", "x'", "x2", "z", "z'", "z2"] {
            for y in ["y", "y'", "y2"] {
                all.push(format!("{r} {y}"));
            }
        }
        all
    })
}

fn simplest(rotation: &str) -> String {
    let key = face_map(rotation);
    names().iter().find(|name| face_map(name) == key).cloned().unwrap_or_else(|| rotation.to_string())
}

/// `a b` trimmed, as ChronoCube joins rotations
fn join(a: &str, b: &str) -> String {
    format!("{a} {b}").trim().to_string()
}

/// The tokens written for each turn: the turn in the cuber's frame, after the rotations made before it (the one that
/// puts the cross down before the first turn, a y before a step). For `reconstruct`.
pub fn present_solve(start: &Cube, moves: &[String], cross: Option<u8>) -> Vec<Vec<String>> {
    if moves.is_empty() {
        return Vec::new();
    }
    let base = cross_down(start, moves, cross);
    // Steps: the turns up to each CFOP step reached (cross, pairs, OLL, PLL)
    let mut steps: Vec<usize> = Vec::new();
    let mut cube = start.clone();
    let mut reached = cfop_progress(&cube);
    let cross_first = reached == 0;
    for (i, m) in moves.iter().enumerate() {
        cube = cube.apply(m);
        let progress = cfop_progress(&cube);
        if progress > reached {
            steps.push(i + 1);
            reached = progress;
        }
    }
    if steps.last() != Some(&moves.len()) {
        steps.push(moves.len());
    }

    let mut written: Vec<Vec<String>> = vec![Vec::new(); moves.len()];
    let mut rotation = base.clone();
    let mut from = 0;
    for (s, &to) in steps.iter().enumerate() {
        let mut rotate = 0;
        // The cross keeps the hold it is solved in; then the best y for each step
        if s > 0 || !cross_first {
            let mut best = f64::INFINITY;
            for (k, y) in Y_TOKENS.iter().enumerate() {
                let turned = join(&rotation, y);
                let turns = moves[from..to].iter().fold(0.0, |sum, m| sum + face_cost(turned_move(m, &turned).chars().next().unwrap()));
                let cost = Y_COST[k] + turns;
                if cost < best - 1e-9 {
                    best = cost;
                    rotate = k;
                }
            }
        }
        if rotate > 0 {
            rotation = join(&rotation, Y_TOKENS[rotate]);
        }
        for i in from..to {
            if i == 0 && !base.is_empty() {
                written[i].extend(base.split(' ').map(String::from));
            }
            if i == from && rotate > 0 {
                written[i].push(Y_TOKENS[rotate].to_string());
            }
            written[i].push(turned_move(&moves[i], &rotation));
        }
        from = to;
    }
    written
}

/// The rotations written at the start of a solution, and the rest ("z2 y R U…" → ("z2 y", "R U…"))
pub fn leading_rotation(solution: &str) -> (String, String) {
    let tokens: Vec<&str> = solution.split(js_space).filter(|t| !t.is_empty()).collect();
    let n = tokens.iter().position(|t| !t.starts_with(['x', 'y', 'z'])).unwrap_or(tokens.len());
    (tokens[..n].join(" "), tokens[n..].join(" "))
}

/// A solve to replay move by move: the cube held as scrambled, scrambled, then turned to the hold the solve starts in
/// (the cross down), and the moves after: (setup, algorithm)
pub fn replay_of(scramble: &str, solution: &str, hold: &str) -> (String, String) {
    let (lead, rest) = leading_rotation(solution);
    let setup: Vec<&str> = [hold, scramble, &lead].into_iter().filter(|s| !s.is_empty()).collect();
    (setup.join(" "), rest)
}
