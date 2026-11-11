//! Fixtures written by ChronoCube's scripts/vision/fixtures.test.ts, read into this crate's types
#![allow(dead_code)]

use cube_decode::*;
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
