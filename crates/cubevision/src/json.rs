//! The result of a video, written as JSON: what the window shows, for a program to read

use cube_decode::pipeline::VisionResult;
use cube_decode::{FrameReadings, Motion};
use serde_json::{Value, json};

/// A number JSON can hold (it has neither infinities nor NaN: null)
fn number(v: f64) -> Value {
    if v.is_finite() { json!(v) } else { Value::Null }
}

pub fn result(r: &VisionResult) -> Value {
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
    json!({
        "solved": r.solved,
        "agreement": number(r.agreement),
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
    })
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
