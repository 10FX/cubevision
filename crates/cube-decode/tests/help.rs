mod common;

use common::*;
use cube_decode::decode::{DecodeOptions, decode_solve};
use cube_decode::help::{help_moves, helped_solve};

#[test]
fn help_moves_match_typescript() {
    let help = fixture("help.json");
    for case in help["cases"].as_array().unwrap() {
        let (text, hold) = (case["text"].as_str().unwrap(), case["hold"].as_str().unwrap());
        let want = case["moves"].as_array().map(|ms| ms.iter().map(|m| m.as_u64().unwrap() as usize).collect::<Vec<_>>());
        assert_eq!(help_moves(text, hold), want, "{text:?} held {hold:?}");
    }
}

#[test]
fn helped_solve_matches_typescript() {
    let help = fixture("help.json");
    let helped = &help["helped"];
    let (_, video) = video_fixtures().into_iter().find(|(name, _)| name == helped["video"].as_str().unwrap()).unwrap();
    let start = cube_decode::state::state_of(&cube_core::Cube::solved().apply(video["scramble"].as_str().unwrap()));
    let given: Vec<usize> = helped["help"].as_array().unwrap().iter().map(|m| m.as_u64().unwrap() as usize).collect();
    let got = helped_solve(&start, &readings(&video["readings"]), &given, |s, r| decode_solve(s, r, &DecodeOptions::default()));
    same_decode("helped", &got, &helped["decode"]);
}

#[test]
fn a_whole_solution_given_leaves_nothing_to_decode() {
    let (_, video) = video_fixtures().remove(0);
    let start = cube_decode::state::state_of(&cube_core::Cube::solved().apply(video["scramble"].as_str().unwrap()));
    let solution = video["truth"]["solution"].as_str().unwrap();
    let given = help_moves(solution, "").unwrap();
    let got = helped_solve(&start, &readings(&video["readings"]), &given, |_, _| panic!("nothing left to decode"));
    assert!(got.search.solved);
    assert_eq!(got.search.moves.len(), given.len());
}
