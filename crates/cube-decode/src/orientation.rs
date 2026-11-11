//! The whole cube's orientation through a solve (ChronoCube's vision/orientation.ts), from the reader's orientation
//! output (a rotation per frame, in the camera's frame):
//!
//! - the camera stays put relative to the cuber, so one fixed rotation K takes the cuber's frame to the camera's. It is
//!   learnt while the cuber holds the cube as they scrambled;
//! - in the cuber's frame, the orientation is snapped to the 24 ways of holding a cube, with hysteresis: a tilt stays
//!   the same hold, a new hold must be seen clearly (within ACCEPT degrees) for MIN_HOLD ms;
//! - a change of hold is a rotation (x, y, z…); with a turn at the same moment it is a wide move (L + x = Rw) or a
//!   slice (R L' + x' = M);
//! - the continuous orientation, smoothed, is kept for the 3D replay.
//!
//! Matrices are row-major arrays of 9, axes of the cube (x R, y U, z F, the cuber looking along -z). A cube orientation
//! O takes the WCA directions of the centres (white +y, red +x, green +z) to where they point.

use std::sync::OnceLock;

use cube_core::{Cube, FACES, ORIENTATIONS};

use crate::decode::js_round;
use crate::reading::TimedMove;

pub type Mat3 = [f64; 9];
/// w x y z
pub type Quat = [f64; 4];

/// Outward normal of each face, faces U R F D L B
pub const FACE_NORMALS: [[f64; 3]; 6] =
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];
/// A hold is accepted within this angle of the orientation seen, degrees
pub const ACCEPT: f64 = 35.0;
/// A new hold must be seen this long to count, ms
pub const MIN_HOLD: f64 = 150.0;
/// A rotation this close to a turn makes a wide move or a slice with it, ms
const TOGETHER: f64 = 250.0;

pub const IDENTITY: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|k| {
        let (i, j) = (k / 3, k % 3);
        a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]
    })
}

pub fn transpose(a: &Mat3) -> Mat3 {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

/// Angle of the rotation between a and b, degrees
pub fn angle_between(a: &Mat3, b: &Mat3) -> f64 {
    let m = mul(&transpose(a), b);
    (((m[0] + m[4] + m[8] - 1.0) / 2.0).clamp(-1.0, 1.0).acos() * 180.0) / std::f64::consts::PI
}

fn inverse(a: &Mat3) -> Mat3 {
    let [m0, m1, m2, m3, m4, m5, m6, m7, m8] = *a;
    let (c0, c1, c2) = (m4 * m8 - m5 * m7, m5 * m6 - m3 * m8, m3 * m7 - m4 * m6);
    let det = m0 * c0 + m1 * c1 + m2 * c2;
    // JavaScript's `|| 1e-12`: zero and NaN alike
    let det = if det == 0.0 || det.is_nan() { 1e-12 } else { det };
    [c0, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4, c1, m0 * m8 - m2 * m6, m2 * m3 - m0 * m5, c2, m1 * m6 - m0 * m7, m0 * m4 - m1 * m3]
        .map(|x| x / det)
}

/// The rotation nearest a matrix (polar decomposition, Newton's iteration)
pub fn nearest_rotation(a: &Mat3) -> Mat3 {
    let mut x = *a;
    for _ in 0..20 {
        let it = transpose(&inverse(&x));
        x = std::array::from_fn(|i| (x[i] + it[i]) / 2.0);
    }
    x
}

/// Mean of rotations (the rotation nearest their sum)
pub fn mean_rotation<'a>(rotations: impl IntoIterator<Item = &'a Mat3>) -> Mat3 {
    let mut sum = [0.0; 9];
    for r in rotations {
        for (s, v) in sum.iter_mut().zip(r) {
            *s += v;
        }
    }
    nearest_rotation(&sum)
}

/// Matrix of a whole-cube rotation ("x2 y"…): where it sends each centre
pub fn rotation_of(alg: &str) -> Mat3 {
    let cube = Cube::solved().apply(alg);
    let mut m = [0.0; 9];
    for (f, n) in FACE_NORMALS.iter().enumerate() {
        let c = FACE_NORMALS[cube.color(f, 1, 1) as usize];
        for i in 0..3 {
            for j in 0..3 {
                m[i * 3 + j] += (n[i] * c[j]) / 2.0;
            }
        }
    }
    m
}

/// One of the 24 ways to hold a cube
pub struct Hold {
    pub name: &'static str,
    pub matrix: Mat3,
    /// The face letter of each centre colour
    pub letter_of: [char; 6],
}

/// The 24 holds, in ORIENTATIONS' order
pub fn holds() -> &'static [Hold] {
    static HOLDS: OnceLock<Vec<Hold>> = OnceLock::new();
    HOLDS.get_or_init(|| {
        ORIENTATIONS
            .iter()
            .map(|&name| {
                let held = Cube::solved().apply(name);
                let letter_of = std::array::from_fn(|colour| FACES[(0..6).find(|&f| held.color(f, 1, 1) as usize == colour).unwrap()]);
                Hold { name, matrix: rotation_of(name), letter_of }
            })
            .collect()
    })
}

/// The hold nearest an orientation, and how far it is (degrees)
pub fn snap(o: &Mat3) -> (usize, f64) {
    let (mut best, mut angle) = (0, f64::INFINITY);
    for (i, h) in holds().iter().enumerate() {
        let a = angle_between(&h.matrix, o);
        if a < angle {
            angle = a;
            best = i;
        }
    }
    (best, angle)
}

fn hold_index(name: &str) -> usize {
    holds().iter().position(|h| h.name == name.trim()).unwrap_or(0)
}

/// Name of a whole-cube rotation given as a matrix
fn rotation_name(q: &Mat3) -> &'static str {
    holds()[snap(q).0].name
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientedFrame {
    pub t: f64,
    pub orientation: Mat3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoldSpan {
    pub from: f64,
    pub hold: usize,
}

/// Holds through the solve, from the orientations in the cuber's frame, with hysteresis
pub fn hold_spans(frames: &[OrientedFrame], start: usize) -> Vec<HoldSpan> {
    let mut spans = vec![HoldSpan { from: f64::NEG_INFINITY, hold: start }];
    let mut candidate: Option<(usize, f64)> = None;
    for f in frames {
        let (hold, angle) = snap(&f.orientation);
        if angle > ACCEPT {
            continue; // between two holds, or a poor reading
        }
        let current = spans.last().unwrap().hold;
        match candidate {
            _ if hold == current => candidate = None,
            Some((h, since)) if h == hold => {
                if f.t - since >= MIN_HOLD {
                    spans.push(HoldSpan { from: since, hold });
                    candidate = None;
                }
            }
            _ => candidate = Some((hold, f.t)),
        }
    }
    spans
}

fn hold_at(spans: &[HoldSpan], t: f64) -> usize {
    spans.iter().rfind(|s| s.from <= t).map_or(spans[0].hold, |s| s.hold)
}

fn opposite(face: char) -> char {
    match face {
        'U' => 'D',
        'D' => 'U',
        'R' => 'L',
        'L' => 'R',
        'F' => 'B',
        _ => 'F',
    }
}

/// The whole-cube rotation that turns like each face
fn like(face: char) -> &'static str {
    match face {
        'R' => "x",
        'L' => "x'",
        'U' => "y",
        'D' => "y'",
        'F' => "z",
        _ => "z'",
    }
}

fn slice(face: char) -> Option<char> {
    match face {
        'L' => Some('M'),
        'D' => Some('E'),
        'F' => Some('S'),
        _ => None,
    }
}

fn quarters(m: &str) -> u32 {
    if m.ends_with('2') {
        2
    } else if m.ends_with('\'') {
        3
    } else {
        1
    }
}

fn suffix(k: u32) -> &'static str {
    ["", "", "2", "'"][(k % 4) as usize]
}

fn turned_like(face: char, k: u32) -> Mat3 {
    rotation_of(&vec![like(face); (k % 4) as usize].join(" "))
}

fn same_rotation(a: &Mat3, b: &Mat3) -> bool {
    angle_between(a, b) < 1.0
}

struct Event {
    t: f64,
    q: Mat3,
    used: bool,
}

/// The turns as the cuber made them. `moves`: the search's turns, faces named by centre colour (U = white…). Returns,
/// for each turn, the tokens written for it (a rotation just before it, a wide move or a slice; nothing for the
/// second turn of a slice).
pub fn cuber_notation(moves: &[TimedMove], spans: &[HoldSpan], scramble_hold: usize) -> Vec<Vec<String>> {
    let holds = holds();
    let mut events: Vec<Event> = spans
        .windows(2)
        .map(|w| Event { t: w[1].from, q: mul(&holds[w[1].hold].matrix, &transpose(&holds[w[0].hold].matrix)), used: false })
        .collect();
    // Turned from the scramble's hold before the first turn (inspection)
    if spans[0].hold != scramble_hold {
        let q = mul(&holds[spans[0].hold].matrix, &transpose(&holds[scramble_hold].matrix));
        events.insert(0, Event { t: f64::NEG_INFINITY, q, used: false });
    }
    let mut out: Vec<Vec<String>> = vec![Vec::new(); moves.len()];
    let letter = |m: &str, hold: usize| holds[hold].letter_of[FACES.iter().position(|&f| m.starts_with(f)).unwrap()];
    let near_event = |events: &[Event], t: f64, fits: &dyn Fn(&Mat3) -> bool| {
        events.iter().position(|e| !e.used && (e.t - t).abs() <= TOGETHER && fits(&e.q))
    };
    let mut i = 0;
    while i < moves.len() {
        let TimedMove { token, t } = &moves[i];
        let t = *t;
        // Rotations done before this turn
        for e in events.iter_mut() {
            if !e.used && e.t < t - TOGETHER {
                out[i].push(rotation_name(&e.q).to_string());
                e.used = true;
            }
        }
        let before = hold_at(spans, t - TOGETHER);
        let (face, k) = (letter(token, before), quarters(token));
        // A slice: two opposite turns undoing each other's direction, with the matching rotation
        if let Some(next) = moves.get(i + 1)
            && next.t - t <= TOGETHER
            && letter(&next.token, before) == opposite(face)
            && (k + quarters(&next.token)).is_multiple_of(4)
        {
            let x = if slice(face).is_some() { face } else { opposite(face) };
            let kx = if x == face { (4 - k) % 4 } else { k };
            let target = turned_like(x, kx);
            if let Some(e) = near_event(&events, t, &|q| same_rotation(q, &target)) {
                events[e].used = true;
                out[i].push(format!("{}{}", slice(x).unwrap(), suffix(kx)));
                i += 2;
                continue;
            }
        }
        // A wide move: this face with a rotation like the opposite face (L + x = Rw)
        let target = turned_like(opposite(face), k);
        if let Some(e) = near_event(&events, t, &|q| same_rotation(q, &target)) {
            events[e].used = true;
            out[i].push(format!("{}w{}", opposite(face), suffix(k)));
        } else {
            out[i].push(format!("{face}{}", &token[1..]));
        }
        i += 1;
    }
    out
}

/// Cost of a frame read in another of the cube's 24 symmetries than the one it shows
const FLIP_COST: f64 = 1.0;
/// Cost of a jump of 90° from one frame to the next, beyond what a hand turns the cube in the time
const JUMP_COST: f64 = 30.0;
/// How fast a hand turns the whole cube, degrees per ms (a y in 150 ms)
const HAND_SPEED: f64 = 0.6;

/// The orientations with the reader's symmetry slips undone. Looking straight at a face, the reader may read the cube
/// turned by 90° or 180° about it, for half a second at times: from one frame to the next, a jump no hand makes. Each
/// frame is taken as read in one of the 24 symmetries, the one read cheap, a jump costly (Viterbi): a slip that comes
/// back is undone, a rotation that stays is kept.
pub fn unflip(frames: &[OrientedFrame]) -> Vec<OrientedFrame> {
    if frames.len() < 2 {
        return frames.to_vec();
    }
    let syms: Vec<Mat3> = holds().iter().map(|h| transpose(&h.matrix)).collect();
    let identity = holds().iter().position(|h| h.name.is_empty()).unwrap();
    let flip = |e: usize| if e == identity { 0.0 } else { FLIP_COST };
    let candidates: Vec<Vec<Mat3>> = frames.iter().map(|f| syms.iter().map(|s| mul(&f.orientation, s)).collect()).collect();
    let mut cost: Vec<f64> = (0..syms.len()).map(flip).collect();
    let mut back: Vec<Vec<usize>> = Vec::with_capacity(frames.len() - 1);
    for i in 1..frames.len() {
        let free = HAND_SPEED * (frames[i].t - frames[i - 1].t);
        let mut from = Vec::with_capacity(syms.len());
        cost = (0..syms.len())
            .map(|e| {
                let (mut best, mut arg) = (f64::INFINITY, 0);
                for (p, c) in cost.iter().enumerate() {
                    let jump = (angle_between(&candidates[i - 1][p], &candidates[i][e]) - free).max(0.0);
                    let c = c + (JUMP_COST * jump) / 90.0;
                    if c < best {
                        best = c;
                        arg = p;
                    }
                }
                from.push(arg);
                best + flip(e)
            })
            .collect();
        back.push(from);
    }
    let low = cost.iter().copied().fold(f64::INFINITY, f64::min);
    let mut e = cost.iter().position(|&c| c == low).unwrap_or(0);
    let mut out = frames.to_vec();
    for i in (0..frames.len()).rev() {
        out[i].orientation = candidates[i][e];
        if i > 0 {
            e = back[i - 1][e];
        }
    }
    out
}

// ─── The track kept for the replay ────────────────────────────────────────────

pub fn quat_of(m: &Mat3) -> Quat {
    let [m00, m01, m02, m10, m11, m12, m20, m21, m22] = *m;
    let tr = m00 + m11 + m22;
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [s / 4.0, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [(m21 - m12) / s, s / 4.0, (m01 + m10) / s, (m02 + m20) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m02 - m20) / s, (m01 + m10) / s, s / 4.0, (m12 + m21) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, s / 4.0]
    }
}

/// Rotation `a` after rotation `b` (Hamilton product)
pub fn quat_mul(a: &Quat, b: &Quat) -> Quat {
    let ([aw, ax, ay, az], [bw, bx, by, bz]) = (*a, *b);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// The inverse of a unit quaternion
pub fn quat_conj(&[w, x, y, z]: &Quat) -> Quat {
    [w, -x, -y, -z]
}

fn normalize(q: &Quat) -> Quat {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let n = if n == 0.0 || n.is_nan() { 1.0 } else { n };
    q.map(|x| x / n)
}

/// Spherical interpolation of unit quaternions
pub fn slerp(a: &Quat, b: &Quat, u: f64) -> Quat {
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let s = if dot < 0.0 { -1.0 } else { 1.0 };
    dot *= s;
    if dot > 0.9995 {
        return normalize(&std::array::from_fn(|i| a[i] + (s * b[i] - a[i]) * u));
    }
    let th = dot.acos();
    let sin = th.sin();
    let (wa, wb) = (((1.0 - u) * th).sin() / sin, (s * (u * th).sin()) / sin);
    std::array::from_fn(|i| wa * a[i] + wb * b[i])
}

/// The cube's orientation through a solve, for the replay: unit quaternions at `hz`, from `start` ms, as signed bytes
/// (×127), w x y z, in base64
#[derive(Clone, Debug, PartialEq)]
pub struct OrientationTrack {
    pub hz: f64,
    pub start: f64,
    pub q: String,
}

/// Samples per second of the replay track
const HZ: f64 = 15.0;
/// Half-width of the smoothing window, ms
const SMOOTH: f64 = 90.0;

/// How the cube turned since `reference`, smoothed and sampled for the replay
pub fn encode_track(frames: &[OrientedFrame], reference: &Mat3) -> Option<OrientationTrack> {
    if frames.len() < 2 {
        return None;
    }
    let back = transpose(reference);
    let rel: Vec<(f64, Quat)> = frames.iter().map(|f| (f.t, quat_of(&mul(&f.orientation, &back)))).collect();
    let (start, end) = (rel[0].0, rel[rel.len() - 1].0);
    let mut samples: Vec<Option<Quat>> = Vec::new();
    let (mut t, mut j) = (start, 0);
    while t <= end {
        while j < rel.len() && rel[j].0 < t - SMOOTH {
            j += 1;
        }
        let near: Vec<&Quat> = rel[j..].iter().take_while(|r| r.0 <= t + SMOOTH).map(|r| &r.1).collect();
        if near.is_empty() {
            samples.push(None);
        } else {
            // Mean of the quaternions, signs aligned with the first
            let first = near[0];
            let mut sum = [0.0; 4];
            for q in &near {
                let s = if q[0] * first[0] + q[1] * first[1] + q[2] * first[2] + q[3] * first[3] < 0.0 { -1.0 } else { 1.0 };
                for (acc, x) in sum.iter_mut().zip(q.iter()) {
                    *acc += s * x;
                }
            }
            samples.push(Some(normalize(&sum)));
        }
        t += 1000.0 / HZ;
    }
    // Gaps: the nearest sample before (or after)
    let mut last = samples.iter().flatten().next().copied().unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for q in &samples {
        if let Some(q) = q {
            last = *q;
        }
        bytes.extend(last.iter().map(|x| js_round(x * 127.0) as i8 as u8));
    }
    Some(OrientationTrack { hz: HZ, start: js_round(start), q: base64(&bytes) })
}

/// The track's samples as unit quaternions
pub fn decode_track(track: &OrientationTrack) -> Vec<Quat> {
    unbase64(&track.q).chunks_exact(4).map(|c| normalize(&std::array::from_fn(|i| c[i] as i8 as f64 / 127.0))).collect()
}

/// The cube's orientation at `t` ms in the video (w x y z, cuber's frame), interpolated
pub fn orientation_at(track: &OrientationTrack, samples: &[Quat], t: f64) -> Quat {
    if samples.is_empty() {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let x = ((t - track.start) * track.hz) / 1000.0;
    if x <= 0.0 {
        return samples[0];
    }
    if x >= (samples.len() - 1) as f64 {
        return samples[samples.len() - 1];
    }
    let i = x.floor() as usize;
    slerp(&samples[i], &samples[i + 1], x - i as f64)
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { BASE64[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

fn unbase64(text: &str) -> Vec<u8> {
    let values: Vec<u32> = text.bytes().filter_map(|c| BASE64.iter().position(|&b| b == c).map(|v| v as u32)).collect();
    let mut out = Vec::with_capacity(values.len() * 3 / 4);
    for chunk in values.chunks(4) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &v)| n | v << (18 - 6 * i));
        for i in 0..chunk.len().saturating_sub(1) {
            out.push((n >> (16 - 8 * i)) as u8);
        }
    }
    out
}

// ─── From the reader's frames ─────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct OrientationResult {
    /// Tokens written for each turn (see cuber_notation)
    pub notation: Vec<Vec<String>>,
    pub track: Option<OrientationTrack>,
    /// Rotations found (wide moves and slices included)
    pub rotations: u32,
}

/// The cuber's view of a solve. `frames`: orientations read (camera's frame) on frames where a face was read;
/// `camera`: K when the live mode learnt it, else it is learnt from the frames before the first turn, the cube held as
/// scrambled. None when the orientations do not hold up (most frames far from any hold): the moves stay written in the
/// scramble's hold.
pub fn cuber_view(frames: &[OrientedFrame], moves: &[TimedMove], scramble_hold: &str, camera: Option<&Mat3>) -> Option<OrientationResult> {
    if frames.len() < 5 || moves.is_empty() {
        return None;
    }
    let frames = unflip(frames);
    let scrambled = hold_index(scramble_hold);
    let held = &holds()[scrambled];
    let before_first = |f: &&OrientedFrame| f.t < moves[0].t - 50.0;
    let k = match camera {
        Some(k) => *k,
        None => {
            let early: Vec<&OrientedFrame> = frames.iter().filter(before_first).collect();
            let used = if early.len() >= 3 { early } else { frames[..5].iter().collect() };
            mul(&mean_rotation(used.iter().map(|f| &f.orientation)), &transpose(&held.matrix))
        }
    };
    let kt = transpose(&k);
    let cuber: Vec<OrientedFrame> = frames.iter().map(|f| OrientedFrame { t: f.t, orientation: mul(&kt, &f.orientation) }).collect();
    let close = cuber.iter().filter(|f| snap(&f.orientation).1 <= ACCEPT).count();
    if (close as f64) < cuber.len() as f64 * 0.6 {
        return None;
    }
    let first: Vec<&OrientedFrame> = cuber.iter().filter(before_first).collect();
    let first = if first.is_empty() { cuber[..3].iter().collect() } else { first };
    let start = snap(&mean_rotation(first.iter().map(|f| &f.orientation))).0;
    let spans = hold_spans(&cuber, start);
    let notation = cuber_notation(moves, &spans, scrambled);
    let rotations = notation
        .iter()
        .flatten()
        .filter(|tok| tok.starts_with(['x', 'y', 'z']) || tok.contains('w') || tok.starts_with(['M', 'E', 'S']))
        .count() as u32;
    // Relative to the hold that scrambled: the replay starts from the cube as it was scrambled
    Some(OrientationResult { notation, track: encode_track(&cuber, &held.matrix), rotations })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_goes_both_ways() {
        for n in 0..10u8 {
            let bytes: Vec<u8> = (0..n).map(|i| i.wrapping_mul(97).wrapping_add(200)).collect();
            assert_eq!(unbase64(&base64(&bytes)), bytes);
        }
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
    }

    #[test]
    fn a_track_decodes_to_its_orientations() {
        let frames: Vec<OrientedFrame> =
            (0..30).map(|i| OrientedFrame { t: i as f64 * 33.0, orientation: rotation_of(if i < 15 { "" } else { "y" }) }).collect();
        let track = encode_track(&frames, &IDENTITY).unwrap();
        let samples = decode_track(&track);
        let q = orientation_at(&track, &samples, 0.0);
        assert!((q[0] - 1.0).abs() < 1e-3);
        let y = quat_of(&rotation_of("y"));
        let end = orientation_at(&track, &samples, 29.0 * 33.0);
        assert!((end.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>().abs() - 1.0).abs() < 1e-3);
    }
}
