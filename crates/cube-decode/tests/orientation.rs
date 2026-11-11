mod common;

use common::*;
use cube_decode::TimedMove;
use cube_decode::orientation::{HoldSpan, Mat3, OrientedFrame, cuber_view, hold_spans, snap, unflip};
use serde_json::Value;

fn mat(v: &Value) -> Mat3 {
    let all: Vec<f64> = v.as_array().unwrap().iter().map(num).collect();
    all.try_into().unwrap()
}

fn oriented(v: &Value) -> Vec<OrientedFrame> {
    v.as_array().unwrap().iter().map(|f| OrientedFrame { t: num(&f["t"]), orientation: mat(&f["orientation"]) }).collect()
}

fn moves(v: &Value) -> Vec<TimedMove> {
    v.as_array().unwrap().iter().map(|m| TimedMove { token: m["move"].as_str().unwrap().to_string(), t: num(&m["t"]) }).collect()
}

#[test]
fn cuber_view_matches_typescript() {
    let all = fixture("orientations.json");
    let cases = all["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 40);
    let mut rotations = 0;
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let camera = c.get("camera").map(mat);
        let got = cuber_view(&oriented(&c["frames"]), &moves(&c["moves"]), "", camera.as_ref());
        rotations += got.as_ref().map_or(0, |r| r.rotations);
        same_view(name, &got, &c["result"]);
    }
    // The views found the cuber's rotations
    assert!(rotations > 100, "{rotations} rotations");
    for key in ["held", "few", "lost"] {
        let c = &all[key];
        let hold = c.get("hold").and_then(Value::as_str).unwrap_or("");
        same_view(key, &cuber_view(&oriented(&c["frames"]), &moves(&c["moves"]), hold, None), &c["result"]);
    }
}

#[test]
fn cuber_view_matches_typescript_on_the_videos() {
    let mut views = 0;
    for (name, video) in video_fixtures() {
        let result = &video["result"];
        // The moves as finish gives them: in the hold that scrambled (white on top, green in front)
        assert_eq!(result["hold"].as_str().unwrap(), "", "{name}");
        let frames: Vec<OrientedFrame> = video["frames"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f.get("orientation").is_some_and(|o| !o.is_null()) && !f["faces"].as_array().unwrap().is_empty())
            .map(|f| OrientedFrame { t: num(&f["t"]), orientation: mat(&f["orientation"]) })
            .collect();
        let got = cuber_view(&frames, &moves(&result["timed"]), "", None);
        views += got.is_some() as u32;
        same_view(&name, &got, &result["orientation"]);
    }
    assert_eq!(views, 3);
}

#[test]
fn the_pieces_match_typescript() {
    let all = fixture("orientations.json");
    let frames = oriented(&all["cases"][0]["frames"]);
    let round4 = |m: &Mat3| m.map(|v| (v * 1e4 + 0.5).floor() / 1e4);
    let unflipped: Vec<Mat3> = unflip(&frames).iter().map(|f| round4(&f.orientation)).collect();
    let want: Vec<Mat3> = all["unflipped"].as_array().unwrap().iter().map(mat).collect();
    assert_eq!(unflipped, want);
    let lost = oriented(&all["lost"]["frames"]);
    for (f, w) in lost.iter().zip(all["snaps"].as_array().unwrap()) {
        let (hold, angle) = snap(&f.orientation);
        assert_eq!(hold as u64, w["hold"].as_u64().unwrap());
        assert!(close(angle, num(&w["angle"])), "{angle} against {}", w["angle"]);
    }
    let spans: Vec<HoldSpan> = all["spans"].as_array().unwrap().iter().map(|s| HoldSpan { from: num(&s["from"]), hold: s["hold"].as_u64().unwrap() as usize }).collect();
    assert_eq!(hold_spans(&frames, 0), spans);
}
