mod common;

use common::*;
use cube_core::{Cube, ORIENTATIONS};
use cube_decode::TimedMove;
use cube_decode::present::{cross_down, face_map, leading_rotation, present_solve, replay_of};
use cube_decode::reconstruct::{Reconstruction, collapse, reconstruct};
use serde_json::Value;

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn written(v: &Value) -> Vec<Vec<String>> {
    v.as_array().unwrap().iter().map(strings).collect()
}

fn same_reconstruction(name: &str, got: &Reconstruction, want: &Value) {
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

fn timed(v: &Value) -> Vec<TimedMove> {
    v.as_array().unwrap().iter().map(|m| TimedMove { token: m["move"].as_str().unwrap().to_string(), t: num(&m["t"]) }).collect()
}

#[test]
fn reconstructions_match_typescript() {
    let all = fixture("solves.json");
    for s in all["solves"].as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let start = Cube::solved().apply(s["scramble"].as_str().unwrap());
        let solve = timed(&s["solve"]);
        same_reconstruction(name, &reconstruct(&start, &solve, None), &s["reconstruct"]);
        same_reconstruction(name, &reconstruct(&start, &solve, Some(&written(&s["written"]))), &s["reconstructWritten"]);
        let half = &solve[..solve.len() / 2];
        same_reconstruction(name, &reconstruct(&start, half, None), &s["half"]["reconstruct"]);
        let moves: Vec<String> = solve.iter().map(|m| m.token.clone()).collect();
        assert_eq!(collapse(&moves), strings(&s["collapse"]), "{name}: collapse");
    }
    let edges = &all["edges"];
    same_reconstruction("empty", &reconstruct(&Cube::solved(), &[], None), &edges["empty"]);
    let tokens: Vec<String> = ["R", "R", "R", "R", "U", "U'", "F", "F2", "D", "D'", "D"].map(String::from).to_vec();
    assert_eq!(collapse(&tokens), strings(&edges["collapse"]));
}

#[test]
fn presentations_match_typescript() {
    let all = fixture("solves.json");
    for s in all["solves"].as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let start = Cube::solved().apply(s["scramble"].as_str().unwrap());
        let moves: Vec<String> = timed(&s["solve"]).into_iter().map(|m| m.token).collect();
        assert_eq!(present_solve(&start, &moves, None), written(&s["written"]), "{name}: written");
        assert_eq!(present_solve(&start, &moves, Some(0)), written(&s["writtenWhite"]), "{name}: written white");
        assert_eq!(present_solve(&start, &moves, Some(3)), written(&s["writtenYellow"]), "{name}: written yellow");
        assert_eq!(cross_down(&start, &moves, None), s["crossDown"].as_str().unwrap(), "{name}: cross down");
        assert_eq!(cross_down(&start, &moves, Some(0)), s["crossDownWhite"].as_str().unwrap(), "{name}: white down");
        assert_eq!(cross_down(&start, &moves, Some(5)), s["crossDownBlue"].as_str().unwrap(), "{name}: blue down");
        let half = &moves[..moves.len() / 2];
        assert_eq!(present_solve(&start, half, None), written(&s["half"]["written"]), "{name}: half written");
        assert_eq!(cross_down(&start, half, None), s["half"]["crossDown"].as_str().unwrap(), "{name}: half cross down");
        let (scramble, solution) = (s["scramble"].as_str().unwrap(), s["solution"].as_str().unwrap());
        for (hold, key) in [("", "replay"), ("x2", "replayHeld")] {
            let (setup, algorithm) = replay_of(scramble, solution, hold);
            assert_eq!(setup, s[key]["setup"].as_str().unwrap(), "{name}: {key} setup");
            assert_eq!(algorithm, s[key]["algorithm"].as_str().unwrap(), "{name}: {key} algorithm");
        }
        let (lead, rest) = leading_rotation(solution);
        assert_eq!(vec![lead, rest], strings(&s["leading"]), "{name}: leading");
    }
    for m in all["faceMaps"].as_array().unwrap() {
        let rotation = m["rotation"].as_str().unwrap();
        let want: Vec<char> = strings(&m["map"]).iter().map(|s| s.chars().next().unwrap()).collect();
        assert_eq!(face_map(rotation).to_vec(), want, "{rotation:?}");
    }
    assert_eq!(all["faceMaps"].as_array().unwrap().len(), ORIENTATIONS.len() + 4);
    let edges = &all["edges"];
    assert!(present_solve(&Cube::solved(), &[], None).is_empty());
    let samples = ["", "x y2", "z2 y R U", "  R  ", "y' x2 z R"];
    for (s, want) in samples.iter().zip(edges["leading"].as_array().unwrap()) {
        let (lead, rest) = leading_rotation(s);
        assert_eq!(vec![lead, rest], strings(want), "{s:?}");
    }
}
