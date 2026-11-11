mod common;

use common::*;
use cube_decode::mirror::{MirrorCheck, flip_frame};
use cube_decode::motion::{Frame, MotionMeter, thumbnail};
use serde_json::Value;

fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn motion_matches_typescript() {
    for sequence in fixture("pixels.json")["sequences"].as_array().unwrap() {
        let (width, height) = (sequence["width"].as_u64().unwrap() as usize, sequence["height"].as_u64().unwrap() as usize);
        let mut meter = MotionMeter::default();
        for (k, f) in sequence["frames"].as_array().unwrap().iter().enumerate() {
            let what = format!("{width}×{height} frame {k}");
            let data = f["data"].as_str().map(bytes);
            let frame = data.as_ref().map(|data| Frame { data, width, height });
            let motion = meter.measure(frame.as_ref(), num(&f["t"]));
            match motion {
                None => assert!(f["motion"].is_null(), "{what}: no motion, want {}", f["motion"]),
                Some(m) => assert!(!f["motion"].is_null() && close(m, num(&f["motion"])), "{what}: motion {m}, want {}", f["motion"]),
            }
            let Some(frame) = frame else { continue };
            let thumb: Vec<f64> = thumbnail(&frame).iter().map(|&x| x as f64).collect();
            let want: Vec<f64> = f["thumbnail"].as_array().unwrap().iter().map(num).collect();
            assert_eq!(thumb, want, "{what}: thumbnail");
            let flipped = flip_frame(&frame);
            assert_eq!(flipped.len(), frame.data.len(), "{what}: flipped size");
            assert_eq!(
                flipped.iter().fold(7u64, |h, &b| (h * 31 + b as u64) % 1_000_000_007),
                f["flipped"].as_u64().unwrap(),
                "{what}: flipped"
            );
            let start: Vec<u8> = ints(&f["flippedStart"]).iter().map(|&b| b as u8).collect();
            assert_eq!(&flipped[..start.len()], start.as_slice(), "{what}: flipped start");
            assert_eq!(flip_frame(&Frame { data: &flipped, width, height }), frame.data, "{what}: flipped twice");
        }
    }
}

fn answer(v: &Value) -> Option<bool> {
    v.as_bool()
}

#[test]
fn mirror_matches_typescript() {
    let fixture = fixture("pixels.json");
    let mut answers = [0; 3];
    for (c, case) in fixture["mirrors"].as_array().unwrap().iter().enumerate() {
        let mut check = MirrorCheck::new(case["scramble"].as_str().unwrap(), case.get("hold").and_then(Value::as_str));
        for (k, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let got = check.add(&faces(&step["normal"]), &faces(&step["flipped"]));
            assert_eq!(got, answer(&step["answer"]), "case {c} frame {k}: answer");
            assert_eq!(check.mirrored, answer(&step["mirrored"]), "case {c} frame {k}: mirrored");
            assert!(close(check.lead(), num(&step["lead"])), "case {c} frame {k}: lead {} want {}", check.lead(), step["lead"]);
        }
        answers[match check.mirrored {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }] += 1;
    }
    // Each answer is met: unknown, as it is, mirrored
    assert!(answers.iter().all(|&n| n >= 5), "{answers:?}");
}
