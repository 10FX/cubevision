//! Times the decoder on a synthetic video's readings: decode_time <fixture.json> [repeat]
use cube_decode::decode::{DecodeOptions, decode_solve};
use cube_decode::{FaceReading, Reading};
use serde_json::Value;

fn f32s<const N: usize>(v: &Value) -> [f32; N] {
    let xs: Vec<f32> = v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    xs.try_into().unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let repeat: usize = args.get(2).map_or(1, |r| r.parse().unwrap());
    let one: Vec<Reading> = v["readings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| Reading {
            from: r["from"].as_f64().unwrap(),
            to: r["to"].as_f64().unwrap(),
            moved: r.get("moved").and_then(Value::as_bool),
            faces: r["faces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| FaceReading {
                    center: f["center"].as_u64().unwrap() as u8,
                    stickers: f["stickers"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|c| (!c.is_null()).then(|| f32s(c)))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                    rotation: f.get("rotation").filter(|r| !r.is_null()).map(f32s),
                })
                .collect(),
        })
        .collect();
    let span = one.last().unwrap().to + 100.0;
    let readings: Vec<Reading> = (0..repeat)
        .flat_map(|k| one.iter().map(move |r| Reading { from: r.from + k as f64 * span, to: r.to + k as f64 * span, ..r.clone() }))
        .collect();
    let start = cube_decode::state::state_of(&cube_core::Cube::solved().apply(v["scramble"].as_str().unwrap()));
    let t = std::time::Instant::now();
    let got = decode_solve(&start, &readings, &DecodeOptions::default());
    println!(
        "{} readings, {} moves, solved {}, {:.2} s",
        readings.len(),
        got.search.moves.len(),
        got.search.solved,
        t.elapsed().as_secs_f64()
    );
}
