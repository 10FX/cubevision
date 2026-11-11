mod common;

use common::*;
use cube_decode::pipeline::{AnalyzeOptions, analyze_readings};
use cube_decode::truth::{Quarter, Truth, align, rotations_in, score_against_truth};
use serde_json::Value;

fn truth(v: &Value) -> Truth {
    let text = |v: &Value| v.as_str().unwrap().to_string();
    Truth {
        scramble: text(&v["scramble"]),
        solution: text(&v["solution"]),
        moves: v["moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| (text(&m["move"]), num(&m["start"]), num(&m["end"]), text(&m["stage"])))
            .collect(),
        stages: ["cross", "f2l", "oll", "pll"].map(|s| num(&v["stages"][s])),
    }
}

fn opt(v: &Value) -> Option<f64> {
    (!v.is_null()).then(|| num(v))
}

#[test]
fn scores_match_typescript() {
    for (name, video) in video_fixtures() {
        let truth = truth(&video["truth"]);
        let result = analyze_readings(&frames(&video["frames"]), &truth.scramble, &AnalyzeOptions::default()).unwrap();
        let got = score_against_truth(&result, &truth);
        let want = &video["score"];
        assert_eq!(got.quarters as u64, want["quarters"].as_u64().unwrap(), "{name}: quarters");
        assert_eq!(got.distance as u64, want["distance"].as_u64().unwrap(), "{name}: distance");
        assert!(close(got.exact, num(&want["exact"])), "{name}: exact");
        assert_eq!(got.time_median, opt(&want["timeMedian"]), "{name}: time median");
        assert_eq!(got.stages, ["cross", "f2l", "oll", "pll"].map(|s| opt(&want["stages"][s])), "{name}: stages");
        assert_eq!(got.rotations.0 as u64, want["rotations"]["truth"].as_u64().unwrap(), "{name}: rotations played");
        assert_eq!(got.rotations.1.map(|r| r as u64), want["rotations"]["found"].as_u64(), "{name}: rotations found");
    }
}

#[test]
fn a_turn_missed_added_or_mistaken_is_one_apart() {
    let q = |face, dir, t| Quarter { face, dir, t };
    let played = [q('R', 1, 0.0), q('U', 1, 100.0), q('R', -1, 200.0), q('U', -1, 300.0)];
    assert_eq!(align(&played, &played), (0, played.iter().map(|&p| (p, p)).collect()));
    // One missed: the three others pair up, each with its own
    let found = [q('R', 1, 10.0), q('R', -1, 220.0), q('U', -1, 330.0)];
    assert_eq!(align(&played, &found), (1, vec![(played[0], found[0]), (played[2], found[1]), (played[3], found[2])]));
    // One mistaken, one added
    let found = [q('R', 1, 0.0), q('U', -1, 100.0), q('R', -1, 200.0), q('U', -1, 300.0), q('F', 1, 400.0)];
    assert_eq!(align(&played, &found).0, 2);
    assert_eq!(align(&[], &played), (4, vec![]));
}

#[test]
fn rotations_count_what_turns_the_cube_in_the_hands() {
    assert_eq!(rotations_in("x2 F B2 y' Rw U M2 E' S 3Rw R U"), 7);
    assert_eq!(rotations_in("  R U R' U'  "), 0);
    assert_eq!(rotations_in(""), 0);
}
