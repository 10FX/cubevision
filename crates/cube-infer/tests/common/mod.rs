//! Fixtures written by ChronoCube's scripts/vision/fixtures.test.ts
#![allow(dead_code)]

use cube_infer::BoxF;
use serde_json::Value;

pub fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap()
}

pub fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// As Float32Array.from: the double, rounded to f32
pub fn f32s(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| num(x) as f32).collect()
}

pub fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
}

pub fn hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn boxf(v: &Value) -> Option<BoxF> {
    (!v.is_null()).then(|| BoxF { x: num(&v["x"]), y: num(&v["y"]), w: num(&v["w"]), h: num(&v["h"]) })
}

pub fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

pub fn same_box(a: &BoxF, b: &BoxF) -> bool {
    close(a.x, b.x) && close(a.y, b.y) && close(a.w, b.w) && close(a.h, b.h)
}

/// The script's hash of bytes, and of the bits of 32-bit floats
pub fn byte_hash(data: &[u8]) -> u64 {
    data.iter().fold(7u64, |h, &b| (h * 31 + b as u64) % 1_000_000_007)
}

pub fn float_hash(data: &[f32]) -> u64 {
    data.iter().fold(7u64, |h, &v| (h * 31 + v.to_bits() as u64) % 1_000_000_007)
}
