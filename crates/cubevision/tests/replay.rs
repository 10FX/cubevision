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
