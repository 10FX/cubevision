mod common;

use common::*;
use cube_decode::state::*;

#[test]
fn moves_match_typescript() {
    let f = fixture("state.json");
    let names: Vec<&str> = f["moves"].as_array().unwrap().iter().map(|m| m.as_str().unwrap()).collect();
    assert_eq!(MOVES.to_vec(), names);
    for case in f["applyMove"].as_array().unwrap() {
        assert_eq!(apply_move(&state(&case["state"]), case["move"].as_u64().unwrap() as usize), state(&case["out"]));
    }
}

#[test]
fn hashes_match_typescript() {
    for case in fixture("state.json")["hashes"].as_array().unwrap() {
        assert_eq!(hash_of(&state(&case["state"])), case["hash"].as_u64().unwrap());
    }
}

#[test]
fn scores_match_typescript() {
    for (k, case) in fixture("state.json")["scores"].as_array().unwrap().iter().enumerate() {
        let (s, rotations) = reading_score(&state(&case["state"]), &reading(&case["reading"]));
        assert!(close(s, num(&case["score"])), "case {k}: {s} against {}", case["score"]);
        assert_eq!(rotations.iter().map(|&r| r as i64).collect::<Vec<_>>(), ints(&case["rotations"]), "case {k}");
    }
}

#[test]
fn alike_matches_typescript() {
    for (k, case) in fixture("state.json")["alike"].as_array().unwrap().iter().enumerate() {
        assert_eq!(alike(&reading(&case["a"]), &reading(&case["b"]), 0.5), case["alike"].as_bool().unwrap(), "case {k}");
    }
}

fn one_face(cell: Option<[f32; 6]>) -> cube_decode::Reading {
    let mut stickers = [Some([0.9, 0.02, 0.02, 0.02, 0.02, 0.02]); 9];
    stickers[0] = cell;
    cube_decode::Reading { from: 0.0, to: 0.0, faces: vec![cube_decode::FaceReading { center: 0, stickers, rotation: None }.sanitized()], moved: None }
}

#[test]
fn nan_probabilities_count_as_hidden() {
    let solved = state_of(&cube_core::Cube::solved());
    let nan = reading_score(&solved, &one_face(Some([f32::NAN; 6]))).0;
    let hidden = reading_score(&solved, &one_face(None)).0;
    assert_eq!(nan, hidden);
    assert!(nan.is_finite());
}

#[test]
fn zero_probability_scores_minus_infinity_like_javascript() {
    let solved = state_of(&cube_core::Cube::solved());
    // The face read is U; its first cell says "not U" for sure: only a misreading explains it
    let (s, _) = reading_score(&solved, &one_face(Some([0.0, 0.2, 0.2, 0.2, 0.2, 0.2])));
    let misread = 0.5f64.ln() + 8.0 * (1.0f64 / 6.0).ln();
    assert!(close(s, misread), "{s} against {misread}");
}
