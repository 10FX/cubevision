mod common;

use common::*;
use cube_infer::output::{MIN_CONFIDENCE, PRESENT, ReaderOutput, TURNING, confidence, readings_from_output};

#[test]
fn readings_match_typescript() {
    let fixture = fixture("outputs.json");
    let cases = fixture["cases"].as_array().unwrap();
    let (mut faces, mut turned, mut together) = (0usize, 0usize, 0usize);
    for (k, case) in cases.iter().enumerate() {
        let out = ReaderOutput {
            present: f32s(&case["present"]).try_into().unwrap(),
            stickers: f32s(&case["stickers"]),
            turning: num(&case["turning"]) as f32,
            orientation: None,
        };
        let (ordered, still) = (case["ordered"].as_bool().unwrap(), case["still"].as_bool().unwrap());
        let got = readings_from_output(&out, ordered, still);
        let want = case["faces"].as_array().unwrap();
        assert_eq!(
            got.iter().map(|f| f.center as u64).collect::<Vec<_>>(),
            want.iter().map(|f| f["center"].as_u64().unwrap()).collect::<Vec<_>>(),
            "case {k}: faces"
        );
        for (face, want) in got.iter().zip(want) {
            let what = format!("case {k} face {}", face.center);
            let hidden: Vec<bool> = want["hidden"].as_array().unwrap().iter().map(|h| h.as_bool().unwrap()).collect();
            assert_eq!(face.stickers.iter().map(|c| c.is_none()).collect::<Vec<_>>(), hidden, "{what}: hidden");
            let cells: Vec<f32> = face.stickers.iter().flatten().flatten().copied().collect();
            assert_eq!(cells[..6], f32s(&want["first"])[..], "{what}: first cell");
            assert_eq!(float_hash(&cells), want["hash"].as_u64().unwrap(), "{what}: cells");
            assert_eq!(
                face.rotation.map(|r| r.to_vec()),
                (!want["rotation"].is_null()).then(|| f32s(&want["rotation"])),
                "{what}: rotation"
            );
            together += face.rotation.is_some() as usize;
        }
        faces += got.len();
        turned += (out.turning > TURNING && got.is_empty()) as usize;
    }
    // The fixture shows every branch
    assert!(faces > 60 && turned > 5 && together > 20, "{faces} faces, {turned} frames mid-turn, {together} faces read together");
}

fn sure() -> ReaderOutput {
    // Every face present, every sticker surely of its face's colour
    let mut stickers = vec![0.0; 6 * 9 * 7];
    for face in 0..6 {
        for cell in 0..9 {
            stickers[(face * 9 + cell) * 7 + face] = 1.0;
        }
    }
    ReaderOutput { present: [0.9; 6], stickers, turning: 0.1, orientation: None }
}

#[test]
fn a_cube_surely_read_gives_its_faces() {
    let faces = readings_from_output(&sure(), true, false);
    assert_eq!(faces.iter().map(|f| f.center).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
    // Floored: one confident mistake must never rule the true state out
    assert_eq!(faces[2].stickers[0], Some([0.02, 0.02, 0.9, 0.02, 0.02, 0.02]));
    assert!(faces.iter().all(|f| f.rotation == Some([0.97, 0.01, 0.01, 0.01])));
    assert_eq!(confidence(&faces[0].stickers), 0.9f32 as f64);
    assert_eq!((PRESENT, MIN_CONFIDENCE, TURNING), (0.5, 0.6, 0.5));
    // A layer mid-turn: not read, unless the crop did not move
    let turning = ReaderOutput { turning: 0.8, ..sure() };
    assert!(readings_from_output(&turning, true, false).is_empty());
    assert_eq!(readings_from_output(&turning, true, true).len(), 6);
    // One face alone is turned anyhow: its rotation is left free
    let alone = ReaderOutput { present: [0.9, 0.1, 0.1, 0.1, 0.1, 0.1], ..sure() };
    assert_eq!(readings_from_output(&alone, true, false).iter().map(|f| (f.center, f.rotation)).collect::<Vec<_>>(), [(0, None)]);
}

#[test]
fn outputs_no_model_should_give_invent_no_face() {
    for odd in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 7.0] {
        let all = ReaderOutput { present: [odd; 6], stickers: vec![odd; 6 * 9 * 7], turning: odd, orientation: None };
        for (ordered, still) in [(false, false), (true, false), (true, true)] {
            for face in readings_from_output(&all, ordered, still) {
                assert!(face.stickers.iter().flatten().flatten().all(|p| p.is_finite() && *p >= 0.0), "{odd}: {face:?}");
            }
        }
        // Not a number where a probability belongs: the face is not read, the others are
        let mut one = sure();
        one.present[1] = f32::NAN;
        one.stickers[(2 * 9 + 3) * 7 + 2] = odd;
        let faces = readings_from_output(&one, true, false);
        assert_eq!(faces.iter().map(|f| f.center).collect::<Vec<_>>(), [0, 2, 3, 4, 5], "{odd}");
        assert!(faces.iter().all(|f| f.stickers.iter().flatten().flatten().all(|p| p.is_finite() && *p >= 0.0)), "{odd}");
    }
    // Fewer values than six faces of nine stickers: nothing read, no panic
    let short = ReaderOutput { stickers: vec![0.5; 100], ..sure() };
    assert!(readings_from_output(&short, true, false).is_empty());
    let nan = ReaderOutput { present: [f32::NAN; 6], stickers: vec![f32::NAN; 6 * 9 * 7], turning: f32::NAN, orientation: None };
    assert!(readings_from_output(&nan, true, true).is_empty());
}

#[test]
fn the_orientation_is_handed_on_in_doubles() {
    let out = ReaderOutput { orientation: Some([1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]), ..sure() };
    assert_eq!(out.orientation64(), Some([1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]));
    assert_eq!(sure().orientation64(), None);
    // A matrix with a value that is no number is no orientation
    assert_eq!(ReaderOutput { orientation: Some([f32::NAN; 9]), ..sure() }.orientation64(), None);
}
