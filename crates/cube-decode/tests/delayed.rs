mod common;

use common::*;
use cube_core::Cube;
use cube_decode::delayed::{DelayedTracker, LAG};
use cube_decode::state::{MOVES, apply_move, solved_state, state_of};
use cube_decode::{FaceReading, State};

/// The faces U, R and F of a state, surely read
fn sure(state: &State) -> Vec<FaceReading> {
    (0..3u8)
        .map(|face| FaceReading {
            center: state[face as usize * 9 + 4],
            stickers: std::array::from_fn(|cell| {
                let mut p = [0.02f32; 6];
                p[state[face as usize * 9 + cell] as usize] = 0.9;
                Some(p)
            }),
            rotation: Some([0.97, 0.01, 0.01, 0.01]),
        })
        .collect()
}

fn after(state: &State, moves: &[usize]) -> State {
    moves.iter().fold(*state, |s, &m| apply_move(&s, m))
}

#[test]
fn a_move_is_written_ten_frames_after_it_is_seen() {
    let r = MOVES.iter().position(|&m| m == "R").unwrap();
    let mut tracker = DelayedTracker::new(solved_state());
    for _ in 0..3 {
        assert!(tracker.update(&sure(&solved_state())).is_empty());
    }
    let turned = apply_move(&solved_state(), r);
    for frame in 0..LAG {
        // Seen at once, not written yet
        assert!(tracker.update(&sure(&turned)).is_empty(), "frame {frame}");
        assert_eq!((tracker.confirmed(), tracker.pending()), (&[][..], &[r][..]), "frame {frame}");
    }
    assert_eq!(tracker.update(&sure(&turned)), [r]);
    assert_eq!((tracker.confirmed(), tracker.pending()), (&[r][..], &[][..]));
    assert_eq!(tracker.state(), &turned);
    assert_eq!(LAG, 10);
}

#[test]
fn a_move_taken_back_in_time_is_never_written() {
    let (r, u) = (MOVES.iter().position(|&m| m == "R").unwrap(), MOVES.iter().position(|&m| m == "U").unwrap());
    let mut tracker = DelayedTracker::new(solved_state());
    let turned = apply_move(&solved_state(), r);
    // Five frames of an R that was a misreading, then the solved cube again
    for _ in 0..5 {
        tracker.update(&sure(&turned));
    }
    for _ in 0..30 {
        tracker.update(&sure(&solved_state()));
    }
    assert_eq!((tracker.confirmed(), tracker.pending()), (&[][..], &[][..]));
    // A turn of the same face after one written is a new move: what is written does not change
    for state in [turned, apply_move(&turned, r), after(&turned, &[r, u])] {
        for _ in 0..12 {
            tracker.update(&sure(&state));
        }
    }
    assert_eq!(tracker.confirmed(), [r, r, u]);
    // A new start: nothing written any more
    tracker.reset(solved_state());
    assert_eq!((tracker.confirmed(), tracker.pending(), tracker.state()), (&[][..], &[][..], &solved_state()));
}

#[test]
fn what_is_written_never_changes_on_the_videos() {
    for (name, video) in video_fixtures() {
        let start = state_of(&Cube::solved().apply(video["scramble"].as_str().unwrap()));
        let mut tracker = DelayedTracker::new(start);
        let mut written: Vec<usize> = Vec::new();
        for (k, f) in frames(&video["frames"]).iter().enumerate() {
            let new = tracker.update(&f.faces);
            // Only added to, by what the update said
            assert!(tracker.confirmed().starts_with(&written), "{name} frame {k}: {written:?} became {:?}", tracker.confirmed());
            assert_eq!(&tracker.confirmed()[written.len()..], new.as_slice(), "{name} frame {k}");
            written = tracker.confirmed().to_vec();
            // What is written and what is still in doubt lead to the state followed
            assert_eq!(&after(&after(&start, &written), tracker.pending()), tracker.state(), "{name} frame {k}");
        }
        assert!(written.len() >= 20, "{name}: {} moves written", written.len());
    }
}
