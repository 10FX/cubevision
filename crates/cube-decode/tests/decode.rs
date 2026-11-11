mod common;

use common::*;

#[test]
fn fixtures_load() {
    let videos = video_fixtures();
    assert_eq!(videos.len(), 4);
    for (name, v) in &videos {
        assert!(v["scramble"].as_str().is_some_and(|s| !s.is_empty()), "{name}");
        let frames = frames(&v["frames"]);
        assert!(frames.len() >= 100, "{name}: {} frames", frames.len());
        assert!(frames.iter().any(|f| !f.faces.is_empty()), "{name}");
    }
    let solves = fixture("random.json");
    assert_eq!(solves["solves"].as_array().unwrap().len(), 20);
    for s in solves["solves"].as_array().unwrap() {
        assert!(!readings(&s["readings"]).is_empty());
    }
}

use cube_decode::decode::{DecodeOptions, DecodeResult, decode_solve};
use cube_decode::state::solved_state;
use serde_json::Value;

/// The decode against what decodeSolve gave: the same moves at the same times, the same states, the same verdict
fn same_decode(name: &str, got: &DecodeResult, want: &Value) {
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

#[test]
fn random_decodes_match_typescript() {
    for (k, solve) in fixture("random.json")["solves"].as_array().unwrap().iter().enumerate() {
        let got = decode_solve(&state(&solve["start"]), &readings(&solve["readings"]), &DecodeOptions::default());
        same_decode(&format!("solve {k}"), &got, &solve["decode"]);
    }
}

#[test]
fn video_decodes_match_typescript() {
    for (name, v) in video_fixtures() {
        let start = cube_decode::state::state_of(&cube_core::Cube::solved().apply(v["scramble"].as_str().unwrap()));
        let got = decode_solve(&start, &readings(&v["readings"]), &DecodeOptions::default());
        same_decode(&name, &got, &v["decode"]);
    }
}

#[test]
fn no_reading_gives_no_move() {
    let solved = solved_state();
    let scrambled = cube_decode::state::state_of(&cube_core::Cube::solved().apply("R U F"));
    let blind = cube_decode::Reading {
        from: 0.0,
        to: 30.0,
        faces: vec![cube_decode::FaceReading { center: 0, stickers: [None; 9], rotation: None }],
        moved: None,
    };
    for start in [solved, scrambled] {
        for readings in [vec![], vec![blind.clone()]] {
            let got = decode_solve(&start, &readings, &DecodeOptions::default());
            assert!(got.search.moves.is_empty());
            assert!(got.gaps.is_empty());
            // explain() of nothing kept: never "solved", as decodeSolve
            assert!(!got.search.solved);
        }
    }
}

#[test]
fn a_long_video_decodes() {
    let (_, v) = video_fixtures().remove(0);
    let one = readings(&v["readings"]);
    let span = one.last().unwrap().to + 100.0;
    let mut long = Vec::new();
    for round in 0.. {
        for r in &one {
            if long.len() == 3000 {
                break;
            }
            long.push(cube_decode::Reading { from: r.from + round as f64 * span, to: r.to + round as f64 * span, ..r.clone() });
        }
        if long.len() == 3000 {
            break;
        }
    }
    let start = cube_decode::state::state_of(&cube_core::Cube::solved().apply(v["scramble"].as_str().unwrap()));
    let begun = std::time::Instant::now();
    let got = decode_solve(&start, &long, &DecodeOptions::default());
    assert_eq!(got.search.path.len(), 3000);
    if !cfg!(debug_assertions) {
        // A guard against a blowup: 84 s alone on an i5-12400F, the video repeated (each round back to the scramble)
        assert!(begun.elapsed().as_secs() < 180, "{:?}", begun.elapsed());
    }
}

#[test]
fn readings_back_in_time_do_not_panic() {
    let solves = fixture("random.json");
    let mut rs = readings(&solves["solves"][0]["readings"]);
    // Every other reading moved before the one before it
    for k in (1..rs.len()).step_by(2) {
        rs[k].from -= 500.0;
        rs[k].to -= 500.0;
    }
    let got = decode_solve(&state(&solves["solves"][0]["start"]), &rs, &DecodeOptions::default());
    assert_eq!(got.search.path.len(), rs.len());
}
