//! The result of a video, written as JSON: what the window shows, for a program to read

use cube_decode::pipeline::VisionResult;
use cube_decode::{FrameReadings, Motion};
use serde_json::{Value, json};

/// A number JSON can hold (it has neither infinities nor NaN: null)
fn number(v: f64) -> Value {
    if v.is_finite() { json!(v) } else { Value::Null }
}

/// How well the scrambled cube explains the frames read before the first move found: the mean, over those
/// frames, of the share of their stickers the cube has there (None: fewer than 3 such frames)
pub fn start_agreement(r: &VisionResult) -> Option<f64> {
    let first = r.timed.first().map_or(f64::INFINITY, |m| m.t);
    let before: Vec<f64> = r.frame_agreement.iter().filter(|(t, _)| *t < first).filter_map(|(_, share)| *share).collect();
    (before.len() >= 3).then(|| before.iter().sum::<f64>() / before.len() as f64)
}

/// Under this share of the first frames' stickers explained by the scrambled cube, the scramble is not the
/// video's. On four synthetic solves read by the released models, each tried with its own scramble and four
/// others: 0.63 to 0.86 with its own, 0.44 at most with another.
pub const START: f64 = 0.5;
/// The same for the whole solve, when too few frames come before the first move to tell: 0.41 to 0.89 with the
/// video's scramble, 0.26 with another where this rule was needed
pub const WHOLE: f64 = 0.35;

/// Whether the scramble given does not seem to be the video's: the moves found do not end on a solved cube,
/// and the scrambled cube does not explain what the first frames show
pub fn mismatch(r: &VisionResult) -> bool {
    !r.solved && start_agreement(r).map_or(r.agreement < WHOLE, |start| start < START)
}

/// `mismatch`: the scramble does not seem to be the video's; the moves found then explain too little of what was
/// read to be given as a solve, and are left out
pub fn result(r: &VisionResult, mismatch: bool) -> Value {
    let moves: Vec<Value> =
        r.timed.iter().zip(&r.notation).map(|(m, written)| json!({ "move": m.token, "t": number(m.t), "written": written })).collect();
    let splits: Vec<Value> = r
        .reconstruction
        .splits
        .iter()
        .map(|s| {
            json!({
                "step": format!("{:?}", s.phase).to_lowercase(), "end": number(s.end), "duration": number(s.duration),
                "recognition": number(s.recognition), "moves": s.moves, "alg": s.alg,
            })
        })
        .collect();
    let mut out = json!({
        "solved": r.solved,
        "scrambleMismatch": mismatch,
        "agreement": number(r.agreement),
        "startAgreement": start_agreement(r).map(number),
        "frames": r.frames,
        "readFrames": r.read_frames,
        "hold": r.hold,
        "solution": r.reconstruction.solution,
        "moves": moves,
        "reconstruction": {
            "moves": r.reconstruction.moves, "tps": number(r.reconstruction.tps), "time": number(r.reconstruction.time),
            "start": number(r.reconstruction.start), "pause": number(r.reconstruction.pause), "steps": splits,
        },
        "rotations": r.orientation.as_ref().map(|o| o.rotations),
        "gaps": r.gaps.iter().map(|&g| number(g)).collect::<Vec<_>>(),
    });
    if mismatch {
        for invented in ["solution", "moves", "reconstruction", "rotations", "gaps"] {
            out[invented] = Value::Null;
        }
    }
    out
}

/// The frames as read, in the shape ChronoCube dumps them
pub fn frames(frames: &[FrameReadings]) -> Value {
    let all: Vec<Value> = frames
        .iter()
        .map(|f| {
            let faces: Vec<Value> =
                f.faces.iter().map(|face| json!({ "center": face.center, "stickers": face.stickers, "rotation": face.rotation })).collect();
            let motion = match f.motion {
                Motion::Change(m) => number(m),
                _ => Value::Null,
            };
            json!({ "t": number(f.t), "faces": faces, "orientation": f.orientation, "turning": f.turning, "motion": motion })
        })
        .collect();
    Value::Array(all)
}
