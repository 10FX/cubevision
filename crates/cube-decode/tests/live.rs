mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use common::*;
use cube_core::{Cube, ORIENTATIONS};
use cube_decode::live::{LiveTracker, ScrambleFollower, ScrambleProgress, TrackerOptions, append_move, undo_moves};
use cube_decode::state::{MOVES, hash_of, state_of};
use serde_json::Value;

fn usizes(v: &Value) -> Vec<usize> {
    ints(v).iter().map(|&x| x as usize).collect()
}

/// JSON writes Infinity as a string
fn confidence(v: &Value) -> f64 {
    match v.as_str() {
        Some("Infinity") => f64::INFINITY,
        _ => num(v),
    }
}

fn same_tracker(name: &str, k: usize, tracker: &LiveTracker, want: &Value) {
    assert_eq!(hash_of(tracker.state()), want["hash"].as_u64().unwrap(), "{name} frame {k}: state");
    assert_eq!(tracker.moves(), usizes(&want["moves"]).as_slice(), "{name} frame {k}: moves");
    let (got, want) = (tracker.confidence(), confidence(&want["confidence"]));
    assert!(got == want || close(got, want), "{name} frame {k}: confidence {got} against {want}");
}

#[test]
fn tracker_matches_typescript() {
    // Each video, frame by frame, from the scrambled cube (held white on top, green in front)
    for (name, video) in video_fixtures() {
        let start = state_of(&Cube::solved().apply(video["scramble"].as_str().unwrap()));
        let mut tracker = LiveTracker::new(start, TrackerOptions::default());
        for (k, (f, want)) in video["frames"].as_array().unwrap().iter().zip(video["live"].as_array().unwrap()).enumerate() {
            tracker.update(&faces(&f["faces"]));
            same_tracker(&name, k, &tracker, want);
        }
    }

    // A scramble turned in front of the camera, with the follower's bonus and anchors as the live mode gives them
    let tracked = &fixture("follower.json")["tracked"];
    let follower = Arc::new(ScrambleFollower::new(tracked["scramble"].as_str().unwrap(), &[""]));
    let done = Arc::new(AtomicU64::new(0f64.to_bits()));
    let (f1, f2, d2) = (follower.clone(), follower.clone(), done.clone());
    let mut tracker = LiveTracker::new(
        state_of(&Cube::solved()),
        TrackerOptions {
            bonus: Some(Box::new(move |s| if f1.locate(s).is_some() { 2.5 } else { 0.0 })),
            anchors: Some(Box::new(move || {
                let done = f64::from_bits(d2.load(Ordering::Relaxed));
                f2.steps.iter().filter(|s| s.done > done + 1.0 && s.done <= done + 6.0).map(|s| (s.state, s.moves.clone())).collect()
            })),
            ..TrackerOptions::default()
        },
    );
    for (k, (f, want)) in tracked["frames"].as_array().unwrap().iter().zip(tracked["live"].as_array().unwrap()).enumerate() {
        tracker.update(&faces(f));
        if let Some(p) = follower.locate(tracker.state()) {
            done.store(p.done.to_bits(), Ordering::Relaxed);
        }
        same_tracker("tracked", k, &tracker, want);
        assert_eq!(f64::from_bits(done.load(Ordering::Relaxed)), num(&want["done"]), "tracked frame {k}: done");
    }
    assert_eq!(num(&tracked["live"].as_array().unwrap().last().unwrap()["done"]), 22.0);
}

fn same_progress(what: &str, got: &Option<ScrambleProgress>, want: &Value) {
    match got {
        None => assert!(want.is_null(), "{what}: none, want {want}"),
        Some(p) => {
            assert_eq!(p.done, num(&want["done"]), "{what}: done");
            assert_eq!(p.total, num(&want["total"]), "{what}: total");
            let holds: Vec<&str> = want["holds"].as_array().unwrap().iter().map(|h| h.as_str().unwrap()).collect();
            assert_eq!(p.holds, holds, "{what}: holds");
        }
    }
}

fn same_way_back(what: &str, got: &Option<(Vec<usize>, f64)>, want: &Value) {
    match got {
        None => assert!(want.is_null(), "{what}: none, want {want}"),
        Some((moves, done)) => {
            assert_eq!(moves, &usizes(&want["moves"]), "{what}: moves");
            assert_eq!(*done, num(&want["done"]), "{what}: done");
        }
    }
}

#[test]
fn follower_matches_typescript() {
    let fixture = fixture("follower.json");
    for case in fixture["cases"].as_array().unwrap() {
        let scramble = case["scramble"].as_str().unwrap();
        let all = ScrambleFollower::new(scramble, &ORIENTATIONS);
        let one = ScrambleFollower::new(scramble, &[""]);
        assert_eq!(all.total, num(&case["total"]), "{scramble}: total");
        let steps = case["steps"].as_array().unwrap();
        assert_eq!(one.steps.len(), steps.len(), "{scramble}: steps");
        for (got, want) in one.steps.iter().zip(steps) {
            assert_eq!(got.state, state(&want["state"]), "{scramble}: step state");
            assert_eq!(got.done, num(&want["done"]), "{scramble}: step done");
            assert_eq!(got.moves, usizes(&want["moves"]), "{scramble}: step moves");
        }
        for (k, p) in case["probes"].as_array().unwrap().iter().enumerate() {
            let (s, after) = (state(&p["state"]), num(&p["after"]));
            let what = format!("{scramble:?} probe {k}");
            same_progress(&format!("{what} all"), &all.locate(&s), &p["all"]);
            same_progress(&format!("{what} one"), &one.locate(&s), &p["one"]);
            same_way_back(&format!("{what} back all"), &all.way_back(&s, after), &p["backAll"]);
            same_way_back(&format!("{what} back one"), &one.way_back(&s, after), &p["backOne"]);
        }
        for (m, row) in case["written"].as_array().unwrap().iter().enumerate() {
            for (h, want) in row.as_array().unwrap().iter().enumerate() {
                assert_eq!(all.written(m, ORIENTATIONS[h]), want.as_str().unwrap(), "{} held {}", MOVES[m], ORIENTATIONS[h]);
            }
        }
    }
}

#[test]
fn moves_append_and_undo_as_typescript() {
    let fixture = fixture("follower.json");
    let sequences = fixture["sequences"].as_array().unwrap();
    for ((moves, appended), undone) in
        sequences.iter().zip(fixture["appended"].as_array().unwrap()).zip(fixture["undone"].as_array().unwrap())
    {
        let moves = usizes(moves);
        assert_eq!(moves.iter().fold(Vec::new(), |acc, &m| append_move(&acc, m)), usizes(appended), "{moves:?}");
        assert_eq!(undo_moves(&moves), usizes(undone), "{moves:?}");
    }
}
