//! Fixtures written by ChronoCube's scripts/vision/fixtures.test.ts, read into this crate's types
#![allow(dead_code)]

use cube_decode::*;
use cube_decode::decode::DecodeResult;
use cube_decode::orientation::OrientationResult;
use cube_decode::reconstruct::Reconstruction;
use serde_json::Value;

pub fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap()
}

/// A number, with "Infinity", "-Infinity" and "NaN" as the script writes them
pub fn num(v: &Value) -> f64 {
    match v.as_str() {
        Some("Infinity") => f64::INFINITY,
        Some("-Infinity") => f64::NEG_INFINITY,
        Some("NaN") => f64::NAN,
        _ => v.as_f64().unwrap_or_else(|| panic!("not a number: {v}")),
    }
}

/// As Float32Array.from: the double, rounded to f32
pub fn f32s<const N: usize>(v: &Value) -> [f32; N] {
    let xs: Vec<f32> = v.as_array().unwrap().iter().map(|x| num(x) as f32).collect();
    xs.try_into().unwrap()
}

pub fn ints(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

pub fn state(v: &Value) -> State {
    ints(v).iter().map(|&x| x as u8).collect::<Vec<_>>().try_into().unwrap()
}

pub fn face(v: &Value) -> FaceReading {
    let cells: Vec<Option<[f32; 6]>> = v["stickers"].as_array().unwrap().iter().map(|c| (!c.is_null()).then(|| f32s(c))).collect();
    FaceReading {
        center: v["center"].as_u64().unwrap() as u8,
        stickers: cells.try_into().unwrap(),
        rotation: v.get("rotation").filter(|r| !r.is_null()).map(f32s),
    }
}

pub fn faces(v: &Value) -> Vec<FaceReading> {
    v.as_array().unwrap().iter().map(face).collect()
}

pub fn reading(v: &Value) -> Reading {
    Reading { from: num(&v["from"]), to: num(&v["to"]), faces: faces(&v["faces"]), moved: v.get("moved").and_then(Value::as_bool) }
}

pub fn frame(v: &Value) -> FrameReadings {
    let motion = match v.get("motion") {
        None => Motion::NotMeasured,
        Some(Value::Null) => Motion::Unknown,
        Some(m) => Motion::Change(num(m)),
    };
    let orientation = v.get("orientation").filter(|o| !o.is_null()).map(|o| {
        let xs: Vec<f64> = o.as_array().unwrap().iter().map(num).collect();
        xs.try_into().unwrap()
    });
    FrameReadings { t: num(&v["t"]), faces: faces(&v["faces"]), orientation, turning: v.get("turning").filter(|t| !t.is_null()).map(num), motion }
}

/// Equal to 1e-9, relatively; infinities equal to themselves
pub fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

/// The synthetic videos' fixtures: (name, content)
pub fn video_fixtures() -> Vec<(String, Value)> {
    let dir = format!("{}/tests/fixtures/videos", env!("CARGO_MANIFEST_DIR"));
    let mut names: Vec<_> = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{dir}: {e}")).map(|e| e.unwrap().file_name().into_string().unwrap()).filter(|n| n.ends_with(".json")).collect();
    names.sort();
    names.into_iter().map(|n| (n.trim_end_matches(".json").to_string(), fixture(&format!("videos/{n}")))).collect()
}

pub fn frames(v: &Value) -> Vec<FrameReadings> {
    v.as_array().unwrap().iter().map(frame).collect()
}

pub fn readings(v: &Value) -> Vec<Reading> {
    v.as_array().unwrap().iter().map(reading).collect()
}

/// The decode against what decodeSolve gave: the same moves at the same times, the same states, the same verdict
pub fn same_decode(name: &str, got: &DecodeResult, want: &Value) {
    let moves: Vec<(String, f64)> = got.search.moves.iter().map(|m| (m.token.clone(), m.t)).collect();
    let expected: Vec<(String, f64)> = want["moves"].as_array().unwrap().iter().map(|m| (m["move"].as_str().unwrap().to_string(), num(&m["t"]))).collect();
    assert_eq!(moves, expected, "{name}: moves");
    assert_eq!(got.search.solved, want["solved"].as_bool().unwrap(), "{name}: solved");
    assert_eq!(got.gaps, want["gaps"].as_array().unwrap().iter().map(num).collect::<Vec<_>>(), "{name}: gaps");
    assert!(close(got.search.agreement, num(&want["agreement"])), "{name}: agreement {} against {}", got.search.agreement, want["agreement"]);
    let states: Vec<_> = got.search.path.iter().map(|(s, _)| s.to_vec()).collect();
    let expected: Vec<_> = want["path"].as_array().unwrap().iter().map(|s| state(s).to_vec()).collect();
    assert_eq!(states, expected, "{name}: path");
}

/// A reconstruction against what reconstruct gave
pub fn same_reconstruction(name: &str, got: &Reconstruction, want: &Value) {
    assert_eq!(got.moves as u64, want["moves"].as_u64().unwrap(), "{name}: moves");
    assert!(close(got.tps, num(&want["tps"])), "{name}: tps {} against {}", got.tps, want["tps"]);
    for (field, value) in [("time", got.time), ("pause", got.pause), ("start", got.start)] {
        assert_eq!(value, num(&want[field]), "{name}: {field}");
    }
    assert_eq!(got.solution, want["solution"].as_str().unwrap(), "{name}: solution");
    let splits = want["splits"].as_array().unwrap();
    assert_eq!(got.splits.len(), splits.len(), "{name}: splits");
    for (s, w) in got.splits.iter().zip(splits) {
        assert_eq!(s.phase.name(), w["phase"].as_str().unwrap(), "{name}");
        for (field, value) in [("end", s.end), ("duration", s.duration), ("recognition", s.recognition)] {
            assert_eq!(value, num(&w[field]), "{name} {}: {field}", s.phase.name());
        }
        assert_eq!(s.moves as u64, w["moves"].as_u64().unwrap(), "{name} {}: moves", s.phase.name());
        assert_eq!(s.alg, w["alg"].as_str().unwrap(), "{name} {}: alg", s.phase.name());
    }
}

/// A view of the cuber against what cuberView gave
pub fn same_view(name: &str, got: &Option<OrientationResult>, want: &Value) {
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
