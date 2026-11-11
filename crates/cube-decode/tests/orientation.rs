mod common;

use common::*;
use cube_decode::TimedMove;
use cube_decode::orientation::{HoldSpan, Mat3, OrientationResult, OrientedFrame, cuber_view, hold_spans, snap, unflip};
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

fn same_view(name: &str, got: &Option<OrientationResult>, want: &Value) {
    let Some(got) = got else {
        assert!(want.is_null(), "{name}: no view, ChronoCube has one");
        return;
    };
    assert!(!want.is_null(), "{name}: a view, ChronoCube has none");
    let notation: Vec<Vec<String>> =
        want["notation"].as_array().unwrap().iter().map(|t| t.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()).collect();
    assert_eq!(got.notation, notation, "{name}: notation");
    assert_eq!(got.rotations as u64, want["rotations"].as_u64().unwrap(), "{name}: rotations");
    match (&got.track, want.get("track")) {
        (None, None) => {}
        (Some(track), Some(w)) => {
            assert_eq!(track.hz, num(&w["hz"]), "{name}: hz");
            assert_eq!(track.start, num(&w["start"]), "{name}: start");
            assert_eq!(track.q, w["q"].as_str().unwrap(), "{name}: track");
        }
        (got, want) => panic!("{name}: track {got:?} against {want:?}"),
    }
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
