//! A solve read, decoded and played back

use cube_core::Cube;
use cube_decode::pipeline::{AnalyzeOptions, analyze_readings};
use cube_decode::state::state_of;
use cube_decode::{FaceReading, FrameReadings, Motion};
use cubevision::replay::Replay;

const SCRAMBLE: &str = "R U2 F' L D2 B R' U F2 L'";
const SOLUTION: &str = "L F2 U' R B' D2 L' F U2 R'";

/// Every face of a cube, surely read
fn sure(cube: &Cube) -> Vec<FaceReading> {
    let state = state_of(cube);
    (0..6)
        .map(|face| FaceReading {
            center: state[face * 9 + 4],
            stickers: std::array::from_fn(|cell| {
                let mut p = [0.02f32; 6];
                p[state[face * 9 + cell] as usize] = 0.9;
                Some(p)
            }),
            rotation: Some([0.97, 0.01, 0.01, 0.01]),
        })
        .collect()
}

/// A video of the solve: each state shown for six frames
fn frames() -> Vec<FrameReadings> {
    let mut cube = Cube::solved().apply(SCRAMBLE);
    let mut frames = Vec::new();
    for token in std::iter::once("").chain(SOLUTION.split(' ')) {
        cube = cube.apply(token);
        for _ in 0..6 {
            frames.push(FrameReadings {
                t: frames.len() as f64 * 33.0,
                faces: sure(&cube),
                orientation: None,
                turning: Some(0.1),
                motion: Motion::Change(0.05),
            });
        }
    }
    frames
}

fn uniform(cube: &Cube) -> bool {
    cube.colors().chunks(9).all(|face| face.iter().all(|c| *c == face[0]))
}

#[test]
fn the_replay_ends_on_a_solved_cube() {
    for cross in [None, Some(0), Some(3), Some(5)] {
        let result = analyze_readings(&frames(), SCRAMBLE, &AnalyzeOptions { cross, ..AnalyzeOptions::default() }).unwrap();
        assert!(result.solved, "cross {cross:?}");
        let replay = Replay::new(SCRAMBLE, &result);
        assert_eq!(replay.len(), result.timed.len());
        assert!(replay.len() >= 8 && !replay.is_empty());
        // Scrambled at the start, solved at the end, whatever way it is written
        assert_eq!(replay.cube_at(0).colors(), Cube::solved().apply(SCRAMBLE).colors());
        assert!(!uniform(&replay.cube_at(replay.len() - 1)));
        assert!(uniform(&replay.cube_at(replay.len())), "cross {cross:?}: {:?}", replay.written);
        assert!(uniform(&replay.cube_at(replay.len() + 5)), "past the end is the end");
        // The clock: before the first turn nothing is made, after the last all are
        let (from, to) = replay.span();
        assert!(from < replay.times[0] && to > *replay.times.last().unwrap());
        assert_eq!((replay.moves_at(from), replay.moves_at(to)), (0, replay.len()));
        assert_eq!(replay.moves_at(replay.times[2]), 3);
        assert_eq!((replay.time_of(0), replay.time_of(3), replay.time_of(99)), (from, replay.times[2], *replay.times.last().unwrap()));
    }
}

fn rotation(m: &[[f32; 3]; 3]) -> bool {
    let dot = |a: usize, b: usize| (0..3).map(|k| m[a][k] * m[b][k]).sum::<f32>();
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    (0..3).all(|a| (0..3).all(|b| (dot(a, b) - (a == b) as u8 as f32).abs() < 0.02)) && (det - 1.0).abs() < 0.02
}

#[test]
fn without_an_orientation_read_the_cube_is_held_as_the_solve_is_written() {
    // The cross asked yellow: the solve is written the cube turned over, and so it is shown, all along
    let result = analyze_readings(&frames(), SCRAMBLE, &AnalyzeOptions { cross: Some(0), ..AnalyzeOptions::default() }).unwrap();
    let replay = Replay::new(SCRAMBLE, &result);
    assert!(!replay.follows());
    let lead = cube_decode::present::leading_rotation(&result.reconstruction.solution).0;
    let want = cube_decode::orientation::rotation_of(&lead);
    let (from, to) = replay.span();
    for clock in [from, (from + to) / 2.0, to] {
        let held = replay.orientation(clock);
        assert!(rotation(&held));
        for (i, row) in held.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!((*v as f64 - want[i * 3 + j]).abs() < 1e-5, "lead {lead:?}: {held:?} want {want:?}");
            }
        }
    }
    // The turns played are plain face turns: each one is shown turning
    assert!(replay.turns.iter().all(|t| cube_view::widget::face_turn(t).is_some()), "{:?}", replay.turns);
}

/// The frames of a video of the decoder's fixtures
fn fixture_frames(name: &str) -> (String, Vec<FrameReadings>) {
    let path = format!("{}/../cube-decode/tests/fixtures/videos/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let video: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let floats = |v: &serde_json::Value| -> Vec<f32> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
    let frames = video["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| FrameReadings {
            t: f["t"].as_f64().unwrap(),
            faces: f["faces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|face| FaceReading {
                    center: face["center"].as_u64().unwrap() as u8,
                    stickers: std::array::from_fn(|i| {
                        (!face["stickers"][i].is_null()).then(|| floats(&face["stickers"][i]).try_into().unwrap())
                    }),
                    rotation: (!face["rotation"].is_null()).then(|| floats(&face["rotation"]).try_into().unwrap()),
                })
                .collect(),
            orientation: f["orientation"].as_array().map(|o| std::array::from_fn(|i| o[i].as_f64().unwrap())),
            turning: f["turning"].as_f64(),
            motion: f["motion"].as_f64().map_or(Motion::Unknown, Motion::Change),
        })
        .collect();
    (video["scramble"].as_str().unwrap().to_owned(), frames)
}

#[test]
fn with_the_orientation_read_the_cube_turns_as_in_the_hands() {
    let (scramble, frames) = fixture_frames("synth-23-022");
    let result = analyze_readings(&frames, &scramble, &AnalyzeOptions::default()).unwrap();
    let replay = Replay::new(&scramble, &result);
    assert!(result.solved && replay.follows());
    assert!(uniform(&replay.cube_at(replay.len())));
    let (from, to) = replay.span();
    // Just before the first turn: held as the solve is written
    let lead = cube_decode::present::leading_rotation(&result.reconstruction.solution).0;
    let want = cube_decode::orientation::rotation_of(&lead);
    let start = replay.orientation(replay.times[0] - 50.0);
    for (i, row) in start.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert!((*v as f64 - want[i * 3 + j]).abs() < 0.03, "lead {lead:?}: {start:?} want {want:?}");
        }
    }
    // Through the solve: always a rotation, and it does turn (the cuber's rotations)
    let mut moved = 0.0f32;
    let mut clock = from;
    while clock <= to {
        let now = replay.orientation(clock);
        assert!(rotation(&now), "at {clock}: {now:?}");
        moved = moved.max((0..3).map(|i| (now[i][i] - start[i][i]).abs()).sum());
        clock += 100.0;
    }
    assert!(moved > 0.5, "the cube never turns: {moved}");
}

#[test]
fn stepping_goes_turn_by_turn_even_when_two_end_together() {
    use cubevision::replay::{Change, Playback};
    // Two pairs of turns end at the same time, as the decoder gives for turns it cannot tell apart in time
    let turns: Vec<String> = "R U R' U' F D".split(' ').map(String::from).collect();
    let replay = Replay::of(Cube::solved(), turns, vec![1000.0, 1000.0, 1500.0, 2000.0, 2000.0, 2600.0]);
    let mut play = Playback::new(&replay);
    assert_eq!((play.made, play.playing), (0, true));
    // Forward to the end, one turn a step
    for made in 1..=6 {
        assert_eq!(play.step(&replay, true), Change::Play(made - 1..made));
        assert_eq!((play.made, play.playing), (made, false));
    }
    assert_eq!(play.step(&replay, true), Change::Nothing);
    // And back to the start, one turn a step
    for made in (0..6).rev() {
        assert_eq!(play.step(&replay, false), Change::Set(made), "back to {made}");
        assert_eq!(play.made, made);
    }
    assert_eq!(play.step(&replay, false), Change::Nothing);
    assert_eq!(play.clock, replay.span().0);
}

#[test]
fn playing_hands_over_the_turns_as_their_time_comes() {
    use cubevision::replay::{Change, Playback};
    let turns: Vec<String> = "R U R' U'".split(' ').map(String::from).collect();
    let replay = Replay::of(Cube::solved(), turns, vec![1000.0, 1000.0, 1500.0, 2000.0]);
    let (from, to) = replay.span();
    let mut play = Playback::new(&replay);
    let mut handed = Vec::new();
    let mut frames = 0;
    while play.playing {
        // A window hidden for ten seconds: the clock moves by a tenth of a second at most
        if let Change::Play(turns) = play.tick(&replay, if frames == 3 { 10.0 } else { 0.016 }) {
            handed.extend(turns);
        }
        frames += 1;
        assert!(frames < 10_000, "the replay never ends");
    }
    assert_eq!(handed, [0, 1, 2, 3]);
    assert_eq!((play.clock, play.made), (to, 4));
    assert!(frames > 100, "ten seconds were taken at once: {frames} frames");
    // Paused, nothing moves; at half speed, half as far
    assert_eq!(play.tick(&replay, 0.05), Change::Nothing);
    assert_eq!(play.seek(&replay, from - 500.0), Change::Set(0));
    assert_eq!(play.clock, from);
    play.playing = true;
    play.speed = 0.5;
    play.tick(&replay, 0.05);
    assert!((play.clock - from - 25.0).abs() < 1e-9);
    assert_eq!(play.seek(&replay, 1200.0), Change::Set(2));
    assert_eq!(play.seek(&replay, 1e9), Change::Set(4));
}

#[test]
fn a_result_for_another_scramble_is_not_given_as_a_solve() {
    let result = analyze_readings(&frames(), SCRAMBLE, &AnalyzeOptions::default()).unwrap();
    let given = cubevision::json::result(&result, false);
    assert_eq!(given["scrambleMismatch"], false);
    assert!(given["moves"].as_array().unwrap().len() >= 8 && given["solution"].is_string() && given["reconstruction"]["steps"].is_array());
    let withheld = cubevision::json::result(&result, true);
    assert_eq!(withheld["scrambleMismatch"], true);
    for invented in ["solution", "moves", "reconstruction", "rotations", "gaps"] {
        assert!(withheld[invented].is_null(), "{invented} is given: {}", withheld[invented]);
    }
    // What was measured stays
    assert!(withheld["agreement"].is_number() && withheld["frames"].is_number() && withheld["solved"].is_boolean());
}

#[test]
fn another_scramble_is_told_from_a_solve_hard_to_follow() {
    // The video's own scramble: solved, no mismatch
    let own = analyze_readings(&frames(), SCRAMBLE, &AnalyzeOptions::default()).unwrap();
    assert!(own.solved && !cubevision::json::mismatch(&own));
    assert!(cubevision::json::start_agreement(&own).unwrap() > 0.9);
    // Another scramble: the cube it gives is not the one the first frames show
    let other = analyze_readings(&frames(), "F R U B2 D' L F2", &AnalyzeOptions::default()).unwrap();
    assert!(!other.solved);
    assert!(cubevision::json::mismatch(&other), "start {:?}, whole {}", cubevision::json::start_agreement(&other), other.agreement);
}
