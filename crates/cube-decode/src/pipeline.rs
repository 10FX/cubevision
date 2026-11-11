//! Frames read → reconstruction (ChronoCube's vision/pipeline.ts, the trained model's path): frames that read alike
//! merged, the decoder from the scrambled cube (held white on top, green in front, or as the live tracker saw it),
//! turns that undo each other left out, then the moves written as a cuber writes them.

use cube_core::{Cube, FACES, ORIENTATIONS, invert_alg};

use crate::decode::{DecodeOptions, DecodeResult, decode_solve};
use crate::help::{help_moves, helped_solve};
use crate::orientation::{Mat3, OrientationResult, OrientedFrame, cuber_view};
use crate::present::present_solve;
use crate::reading::{FrameReadings, Motion, Reading, State, TimedMove};
use crate::reconstruct::{Reconstruction, reconstruct};
use crate::state::state_of;

/// A frame whose crop changed more than this since the one before moved
pub const MOTION: f64 = 0.02;
/// Nor did one the reader thinks this likely mid-turn
pub const STILL_TURNING: f64 = 0.15;

/// The likeliest colour of a cell, as JavaScript's `p.indexOf(Math.max(...p))`: the first, -1 with a NaN
fn likeliest(p: &[f32; 6]) -> i32 {
    if p.iter().any(|x| x.is_nan()) {
        return -1;
    }
    let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    p.iter().position(|&x| x == max).map_or(-1, |i| i as i32)
}

/// Frames that read alike are merged (more evidence for one reading); frames without a face are left out. Each reading
/// tells whether anything moved since the one before began, when the frames say (motion, turning). Also: the reading
/// of each frame (-1: nothing read).
pub fn merge_readings(frames: &[FrameReadings]) -> (Vec<Reading>, Vec<i32>) {
    let mut readings: Vec<Reading> = Vec::new();
    let mut sums: Vec<Vec<[[f32; 6]; 9]>> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut reading_of = Vec::with_capacity(frames.len());
    let mut last_signature: Vec<u8> = Vec::new();
    // Something moved since the first frame of the last reading
    let mut moving = false;
    for frame in frames {
        // A frame mid-turn, the cube lost or no motion measured (the first, after a loss) all count as moving
        let still = match frame.motion {
            Motion::NotMeasured => None,
            Motion::Unknown => Some(false),
            Motion::Change(m) => Some(m <= MOTION && frame.turning.unwrap_or(0.0) <= STILL_TURNING),
        };
        if frame.faces.is_empty() {
            moving = true;
            reading_of.push(-1);
            continue;
        }
        let mut faces = frame.faces.clone();
        faces.sort_by_key(|f| f.center);
        let mut signature = Vec::with_capacity(faces.len() * 11);
        for (i, f) in faces.iter().enumerate() {
            if i > 0 {
                signature.push(b'|');
            }
            signature.extend(f.center.to_string().bytes());
            for p in &f.stickers {
                match p {
                    Some(p) => signature.extend(likeliest(p).to_string().bytes()),
                    None => signature.push(b'.'),
                }
            }
        }
        if signature == last_signature {
            // Same view as the previous frame: more evidence for one reading
            let r = readings.len() - 1;
            readings[r].to = frame.t;
            for (i, f) in faces.iter().enumerate() {
                for (k, p) in f.stickers.iter().enumerate() {
                    if let Some(p) = p {
                        for (s, x) in sums[r][i][k].iter_mut().zip(p) {
                            *s = (*s as f64 + *x as f64) as f32;
                        }
                    }
                }
            }
            counts[r] += 1;
            if still == Some(false) {
                moving = true;
            }
        } else {
            let moved = match still {
                Some(still) if !readings.is_empty() => Some(moving || !still),
                _ => None,
            };
            sums.push(faces.iter().map(|f| f.stickers.map(|p| p.unwrap_or([0.0; 6]))).collect());
            readings.push(Reading { from: frame.t, to: frame.t, faces, moved });
            // The move into this reading may go on after it (not the cube found again: nothing to compare with)
            moving = still == Some(false) && frame.motion != Motion::Unknown;
            counts.push(1);
            last_signature = signature;
        }
        reading_of.push(readings.len() as i32 - 1);
    }
    // Average of the merged frames
    for (r, reading) in readings.iter_mut().enumerate() {
        let count = counts[r] as f64;
        for (i, face) in reading.faces.iter_mut().enumerate() {
            for (k, cell) in face.stickers.iter_mut().enumerate() {
                if cell.is_some() {
                    *cell = Some(sums[r][i][k].map(|x| (x as f64 / count) as f32));
                }
            }
        }
    }
    (readings, reading_of)
}

fn axis(face: char) -> u8 {
    match face {
        'U' | 'D' => 0,
        'R' | 'L' => 1,
        _ => 2,
    }
}

fn first_char(token: &str) -> char {
    token.chars().next().unwrap_or(' ')
}

/// One pass of cancel_turns
fn cancel_once(moves: &[TimedMove]) -> Vec<TimedMove> {
    let mut out = Vec::with_capacity(moves.len());
    let (mut i, mut j) = (0, 0);
    while i < moves.len() {
        let a = axis(first_char(&moves[i].token));
        while j < moves.len() && axis(first_char(&moves[j].token)) == a {
            j += 1;
        }
        if j - i == 1 {
            out.push(moves[i].clone());
            i = j;
            continue;
        }
        // Net quarter turns of each face, faces in the order they first came
        let mut net: Vec<(char, f64, u32)> = Vec::new();
        for m in &moves[i..j] {
            let face = first_char(&m.token);
            let quarters = match &m.token[face.len_utf8()..] {
                "2" => 2,
                "'" => 3,
                _ => 1,
            };
            match net.iter_mut().find(|n| n.0 == face) {
                Some(n) => {
                    n.1 = m.t;
                    n.2 = (n.2 + quarters) % 4;
                }
                None => net.push((face, m.t, quarters % 4)),
            }
        }
        net.retain(|n| n.2 != 0);
        net.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out.extend(net.into_iter().map(|(face, t, q)| TimedMove { token: format!("{face}{}", ["", "", "2", "'"][q as usize]), t }));
        i = j;
    }
    out
}

/// Turns that undo each other left out, the others of their axis merged: turns of one axis in a row commute, so
/// "L R' L' R" (a slice and its inverse, read while the cube was turned to look at a side) is nothing, "B2 B2" nothing,
/// "D2 U2 D2" U2. The cube is the same before and after each run; a turn kept takes the time of its face's last. A run
/// gone may bring two others of one axis together ("R U U' R'"): again, until nothing changes.
pub fn cancel_turns(moves: &[TimedMove]) -> Vec<TimedMove> {
    let mut moves = moves.to_vec();
    loop {
        let out = cancel_once(&moves);
        if out.len() >= moves.len() {
            return out;
        }
        moves = out;
    }
}

/// A face's stickers turned 0 to 3 quarters: index of the sticker each one comes from
const TURNS: [[usize; 9]; 4] = {
    const QUARTER: [usize; 9] = [6, 3, 0, 7, 4, 1, 8, 5, 2];
    let mut turns = [[0, 1, 2, 3, 4, 5, 6, 7, 8]; 4];
    let mut k = 1;
    while k < 4 {
        let mut i = 0;
        while i < 9 {
            turns[k][i] = turns[k - 1][QUARTER[i]];
            i += 1;
        }
        k += 1;
    }
    turns
};

/// Frames read against the moves found (centre frame, from the scrambled cube): for each, the share of the stickers
/// read the cube has there, each face under the quarter turn that fits it best (None: no face read)
pub fn frame_agreement(start: &Cube, moves: &[TimedMove], frames: &[FrameReadings]) -> Vec<(f64, Option<f64>)> {
    let mut cube = start.clone();
    let mut m = 0;
    frames
        .iter()
        .map(|frame| {
            while m < moves.len() && moves[m].t <= frame.t {
                cube = cube.apply(&moves[m].token);
                m += 1;
            }
            let (mut agree, mut seen) = (0usize, 0usize);
            for face in &frame.faces {
                let read: [i32; 9] = std::array::from_fn(|i| if i == 4 { -1 } else { face.stickers[i].as_ref().map_or(-1, likeliest) });
                let centre = face.center as usize;
                let best = TURNS
                    .iter()
                    .map(|turn| (0..9).filter(|&i| read[i] >= 0 && cube.color(centre, turn[i] / 3, turn[i] % 3) as i32 == read[i]).count())
                    .max()
                    .unwrap();
                agree += best;
                seen += read.iter().filter(|&&c| c >= 0).count();
            }
            (frame.t, (seen > 0).then(|| agree as f64 / seen as f64))
        })
        .collect()
}

/// How the cube was held for the scramble. The decoder names faces by their centre colour (white U, green F…); a
/// scramble applied with yellow on top leaves other colours on each face.
struct Holding {
    hold: &'static str,
    /// The scrambled cube in centre colours
    centred: Cube,
    start: State,
    /// The scrambled cube as held
    start_cube: Cube,
    /// The letter of each colour's face in the scrambler's hands
    letter_of: [char; 6],
}

/// The hold given, else white on top, green in front: scrambles are always done so
fn holding_of(scramble: &str, hold: Option<&str>) -> Holding {
    let hold = hold.and_then(|h| ORIENTATIONS.iter().copied().find(|&o| o == h.trim())).unwrap_or("");
    let held = Cube::solved().apply(hold);
    let letter_of = std::array::from_fn(|colour| FACES[(0..6).find(|&f| held.color(f, 1, 1) as usize == colour).unwrap()]);
    let centred = Cube::solved().apply(&format!("{hold} {scramble} {}", invert_alg(hold)));
    Holding { hold, start: state_of(&centred), centred, start_cube: held.apply(scramble), letter_of }
}

#[derive(Clone, Debug, Default)]
pub struct AnalyzeOptions {
    /// How the cube was held to scramble, when known (the live tracker sees it)
    pub hold: Option<String>,
    /// The camera's rotation from the cuber's frame, when the live mode learnt it
    pub camera: Option<Mat3>,
    pub decode: DecodeOptions,
    /// Moves the cuber vouches for, from the start of the solve, in their notation: the decoder finds the rest
    pub help: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionResult {
    /// The moves, as turned in the hands that scrambled
    pub timed: Vec<TimedMove>,
    pub reconstruction: Reconstruction,
    /// Share of the stickers read that agree with the moves found (0–1)
    pub agreement: f64,
    /// The moves found end on a solved cube
    pub solved: bool,
    /// Frames where the cube could be read, out of all frames
    pub read_frames: usize,
    pub frames: usize,
    /// How the cube was held to scramble (a rotation: "", "x2 y"…)
    pub hold: String,
    /// The cuber's rotations and the cube's orientation through the solve (models that read it)
    pub orientation: Option<OrientationResult>,
    /// Tokens written for each turn of `timed`: cross down, a y before a step when it helps
    pub notation: Vec<Vec<String>>,
    /// Times (ms) where the turns could not be followed: the moves between are unknown
    pub gaps: Vec<f64>,
    /// Each frame: the share of its stickers read that the cube has there once the moves found are made
    pub frame_agreement: Vec<(f64, Option<f64>)>,
}

/// Frames read by the trained model, to a reconstruction (None: no face read in any frame)
pub fn analyze_readings(frames: &[FrameReadings], scramble: &str, o: &AnalyzeOptions) -> Option<VisionResult> {
    if frames.iter().all(|f| f.faces.is_empty()) {
        return None;
    }
    let (readings, _) = merge_readings(frames);
    let holding = holding_of(scramble, o.hold.as_deref());
    let decode = |start: &State, r: &[Reading]| decode_solve(start, r, &o.decode);
    let help = o.help.as_deref().and_then(|h| help_moves(h, holding.hold));
    let result = match help {
        Some(help) if !help.is_empty() => helped_solve(&holding.start, &readings, &help, decode),
        _ => decode(&holding.start, &readings),
    };
    Some(finish(&holding, result, frames, o.camera.as_ref()))
}

fn finish(holding: &Holding, result: DecodeResult, frames: &[FrameReadings], camera: Option<&Mat3>) -> VisionResult {
    // Moves as turned in the hands that scrambled, those that undo each other left out
    let moves = cancel_turns(&result.search.moves);
    let timed: Vec<TimedMove> = moves
        .iter()
        .map(|m| {
            let face = FACES.iter().position(|&f| m.token.starts_with(f)).unwrap();
            TimedMove { token: format!("{}{}", holding.letter_of[face], &m.token[1..]), t: m.t }
        })
        .collect();
    // With the cube's orientation: moves as the cuber turned them, rotations included
    let oriented: Vec<OrientedFrame> = frames
        .iter()
        .filter(|f| !f.faces.is_empty())
        .filter_map(|f| f.orientation.map(|orientation| OrientedFrame { t: f.t, orientation }))
        .collect();
    let orientation = if oriented.is_empty() { None } else { cuber_view(&oriented, &moves, holding.hold, camera) };
    // Written as a reconstruction: the cross down from the start, rotations only between steps
    let tokens: Vec<String> = timed.iter().map(|m| m.token.clone()).collect();
    let notation = present_solve(&holding.start_cube, &tokens, None);
    VisionResult {
        reconstruction: reconstruct(&holding.start_cube, &timed, Some(&notation)),
        timed,
        orientation,
        notation,
        agreement: result.search.agreement,
        solved: result.search.solved,
        read_frames: frames.iter().filter(|f| !f.faces.is_empty()).count(),
        frames: frames.len(),
        hold: holding.hold.to_string(),
        gaps: result.gaps,
        frame_agreement: frame_agreement(&holding.centred, &moves, frames),
    }
}
