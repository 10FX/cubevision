//! A labelling session: the moves asked for, the frames labelled with the cube's state

use cube_decode::FaceReading;
use cube_decode::state::{MOVES, apply_move, solved_state};
use cubevision::label::{Pause, Phase, STILL, Session, fit, plan};

/// A session past its first Space and its first still time: the first move is asked for
fn started() -> Session {
    let mut s = Session::new(7, 4);
    s.space(1.0);
    s.tick(1.0 + STILL);
    s
}

#[test]
fn a_plan_scrambles_then_comes_back_to_solved() {
    for seed in 0..50 {
        let moves = plan(seed, 20);
        assert_eq!(moves.len(), 40);
        for pair in moves[..20].windows(2) {
            assert_ne!(pair[0] / 3, pair[1] / 3, "the same face twice in a row: {moves:?}");
        }
        for three in moves[..20].windows(3) {
            let axis = |m: usize| (m / 3) % 3;
            assert!(!(axis(three[0]) == axis(three[1]) && axis(three[1]) == axis(three[2])), "three turns on one axis: {moves:?}");
        }
        let scrambled = moves[..20].iter().fold(solved_state(), |s, &m| apply_move(&s, m));
        assert_ne!(scrambled, solved_state());
        assert_eq!(moves.iter().fold(solved_state(), |s, &m| apply_move(&s, m)), solved_state());
    }
    assert_ne!(plan(1, 20), plan(2, 20));
    assert_eq!(plan(3, 20), plan(3, 20));
}

#[test]
fn nothing_is_labelled_before_the_solved_cube_is_shown() {
    let mut s = Session::new(7, 4);
    assert_eq!(s.phase(), Phase::Paused(Pause::Start));
    assert!(s.label(0.5).is_none());
    s.space(1.0);
    assert_eq!(s.phase(), Phase::Still);
    let label = s.label(1.2).unwrap();
    assert_eq!((label.sequence, label.step, label.phase, label.state, label.mv), (0, 0, "still", solved_state(), None));
}

#[test]
fn the_still_time_over_the_move_is_asked_for() {
    let mut s = Session::new(7, 4);
    s.space(1.0);
    s.tick(1.0 + STILL - 0.01);
    assert_eq!(s.phase(), Phase::Still);
    s.tick(1.0 + STILL);
    assert_eq!(s.phase(), Phase::Move);
    let first = plan(7, 4)[0];
    assert_eq!(s.prompt(), Some(first));
    let label = s.label(3.0).unwrap();
    assert_eq!((label.phase, label.state, label.mv), ("move", solved_state(), Some(first)));
}

#[test]
fn space_after_the_move_gives_the_next_state() {
    let mut s = started();
    let moves = plan(7, 4);
    let before = s.label(3.0).unwrap().segment;
    s.space(4.0);
    assert_eq!(s.phase(), Phase::Still);
    let label = s.label(4.1).unwrap();
    assert_eq!((label.step, label.phase, label.state), (1, "still", apply_move(&solved_state(), moves[0])));
    assert!(label.segment > before, "a phase's frames have a segment of their own");
    assert_eq!(s.prompt(), Some(moves[1]));
}

#[test]
fn space_during_the_still_time_is_not_a_move() {
    let mut s = Session::new(7, 4);
    s.space(1.0);
    s.space(1.3);
    assert_eq!(s.label(1.4).unwrap().step, 0);
    assert_eq!(s.state(), solved_state());
}

#[test]
fn undo_takes_the_move_back_and_discards_its_frames() {
    let mut s = started();
    let asked = s.label(3.0).unwrap().segment;
    s.space(4.0);
    let after = s.label(4.1).unwrap().segment;
    s.take_events();
    s.undo(4.5);
    assert_eq!(s.phase(), Phase::Paused(Pause::Undone));
    assert_eq!(s.state(), solved_state());
    assert!(s.label(4.6).is_none());
    let events = s.take_events();
    let discard = events.iter().find(|e| e["type"] == "discard").expect("a discard record");
    let segments: Vec<u64> = discard["segments"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    assert!(segments.contains(&(asked as u64)) && segments.contains(&(after as u64)), "{segments:?}");
    // The cube put back, the same move is asked for again
    s.space(6.0);
    s.tick(6.0 + STILL);
    assert_eq!(s.prompt(), Some(plan(7, 4)[0]));
    let again = s.label(8.0).unwrap();
    assert_eq!((again.step, again.state), (0, solved_state()));
    assert!(!segments.contains(&(again.segment as u64)));
}

#[test]
fn undo_with_no_move_made_does_nothing() {
    let mut s = started();
    s.undo(3.0);
    assert_eq!(s.phase(), Phase::Move);
}

#[test]
fn the_last_move_made_the_sequence_waits_for_the_verdict() {
    let mut s = started();
    let mut t = 3.0;
    for _ in 0..8 {
        s.space(t);
        s.tick(t + STILL);
        t += STILL + 1.0;
    }
    assert_eq!(s.phase(), Phase::Paused(Pause::Done));
    assert_eq!(s.state(), solved_state());
    s.take_events();
    // Solved: the next sequence starts, with other moves
    s.space(t);
    let events = s.take_events();
    assert!(events.iter().any(|e| e["type"] == "sequence" && e["index"] == 0 && e["ok"] == true), "{events:?}");
    let label = s.label(t + 0.1).unwrap();
    assert_eq!((label.sequence, label.step, label.state), (1, 0, solved_state()));
    assert_eq!(s.plan(), plan(8, 4));
}

#[test]
fn a_cube_that_no_longer_matches_ends_the_sequence_as_wrong() {
    let mut s = started();
    s.space(4.0);
    s.take_events();
    s.lost(4.5);
    assert_eq!(s.phase(), Phase::Paused(Pause::Lost));
    assert!(s.label(5.0).is_none());
    let events = s.take_events();
    assert!(events.iter().any(|e| e["type"] == "sequence" && e["index"] == 0 && e["ok"] == false), "{events:?}");
    // The cube solved again: a new sequence
    s.space(9.0);
    let label = s.label(9.1).unwrap();
    assert_eq!((label.sequence, label.step, label.state), (1, 0, solved_state()));
}

#[test]
fn presses_are_recorded_with_their_move() {
    let mut s = started();
    s.take_events();
    s.space(4.0);
    let events = s.take_events();
    let press = events.iter().find(|e| e["type"] == "press").expect("a press record");
    assert_eq!(press["move"], MOVES[plan(7, 4)[0]]);
    assert_eq!((press["sequence"].as_u64(), press["step"].as_u64(), press["t"].as_f64()), (Some(0), Some(0), Some(4.0)));
}

/// A face surely read as the state has it, turned by `rotation` quarter turns
fn read(state: &[u8; 54], face: usize, wrong: usize) -> FaceReading {
    FaceReading {
        center: state[face * 9 + 4],
        stickers: std::array::from_fn(|cell| {
            let mut p = [0.02f32; 6];
            let colour = state[face * 9 + cell] as usize;
            p[if cell < wrong { (colour + 1) % 6 } else { colour }] = 0.9;
            Some(p)
        }),
        rotation: None,
    }
}

#[test]
fn fit_is_the_share_of_stickers_read_as_the_state_has_them() {
    let state = apply_move(&apply_move(&solved_state(), 3), 0);
    assert_eq!(fit(&state, &[]), None);
    assert_eq!(fit(&state, &[read(&state, 0, 0), read(&state, 2, 0)]), Some(1.0));
    // Three stickers of 18 wrong
    let share = fit(&state, &[read(&state, 0, 3), read(&state, 2, 0)]).unwrap();
    assert!((share - 15.0 / 18.0).abs() < 1e-9, "{share}");
    // Another state fits less
    assert!(fit(&solved_state(), &[read(&state, 0, 0), read(&state, 2, 0)]).unwrap() < 1.0);
}

#[test]
fn a_frame_is_recorded_with_its_state_and_its_move() {
    use cubevision::label_window::frame_record;
    let mut s = started();
    s.space(4.0);
    s.tick(4.0 + STILL);
    let label = s.label(6.25049).unwrap();
    let boxed = cube_infer::BoxF { x: 10.4, y: 20.6, w: 300.0, h: 300.0 };
    let record = frame_record("images/000007.jpg", 6.25049, &label, &boxed, (1280, 720), &serde_json::json!({"faces": 2}));
    let moves = plan(7, 4);
    assert_eq!(record["type"], "frame");
    assert_eq!(record["image"], "images/000007.jpg");
    assert_eq!(record["t"], 6.25);
    assert_eq!((record["sequence"].as_u64(), record["step"].as_u64()), (Some(0), Some(1)));
    assert_eq!(record["phase"], "move");
    assert_eq!(record["move"], MOVES[moves[1]]);
    assert_eq!(record["state"], cubevision::label::state_text(&apply_move(&solved_state(), moves[0])));
    assert_eq!(record["box"], serde_json::json!([10.0, 21.0, 300.0, 300.0]));
    assert_eq!(record["frame"], serde_json::json!([1280, 720]));
    assert_eq!(record["read"]["faces"], 2);
}

#[test]
fn a_state_is_written_by_its_colours() {
    let text = cubevision::label::state_text(&solved_state());
    assert_eq!(text, "WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBB");
}

#[test]
fn a_session_never_writes_over_another() {
    use cubevision::label_window::{Log, save_jpeg};
    let dir = std::env::temp_dir().join(format!("cubevision-label-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut log = Log::create(&dir).unwrap();
    log.write(&serde_json::json!({"type": "session"})).unwrap();
    log.write(&serde_json::json!({"type": "end"})).unwrap();
    assert!(Log::create(&dir).is_err(), "a second session in the same folder");
    let lines = std::fs::read_to_string(dir.join("log.jsonl")).unwrap();
    assert_eq!(lines, "{\"type\":\"session\"}\n{\"type\":\"end\"}\n");
    // A crop saved is a JPEG of its size
    let rgba: Vec<u8> = (0..16 * 16).flat_map(|i| [(i % 256) as u8, 80, 200, 255]).collect();
    let file = dir.join("images").join("000000.jpg");
    save_jpeg(&file, &rgba, 16).unwrap();
    let bytes = std::fs::read(&file).unwrap();
    assert_eq!(&bytes[..2], &[0xFF, 0xD8]);
    std::fs::remove_dir_all(&dir).unwrap();
}
