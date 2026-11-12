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
