mod common;

use common::*;
use cube_decode::pipeline::{AnalyzeOptions, VisionResult, analyze_readings, merge_readings};
use cube_decode::{FaceReading, FrameReadings, Motion};
use serde_json::Value;

fn same_result(name: &str, got: &VisionResult, want: &Value) {
    let timed: Vec<(String, f64)> = got.timed.iter().map(|m| (m.token.clone(), m.t)).collect();
    let expected: Vec<(String, f64)> =
        want["timed"].as_array().unwrap().iter().map(|m| (m["move"].as_str().unwrap().to_string(), num(&m["t"]))).collect();
    assert_eq!(timed, expected, "{name}: timed");
    assert_eq!(got.solved, want["solved"].as_bool().unwrap(), "{name}: solved");
    assert!(close(got.agreement, num(&want["agreement"])), "{name}: agreement {} against {}", got.agreement, want["agreement"]);
    assert_eq!(got.read_frames as u64, want["readFrames"].as_u64().unwrap(), "{name}: read frames");
    assert_eq!(got.frames as u64, want["frames"].as_u64().unwrap(), "{name}: frames");
    assert_eq!(got.hold, want["hold"].as_str().unwrap(), "{name}: hold");
    assert_eq!(got.gaps, want["gaps"].as_array().unwrap().iter().map(num).collect::<Vec<_>>(), "{name}: gaps");
    same_view(name, &got.orientation, &want["orientation"]);
    let notation: Vec<Vec<String>> = want["notation"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect())
        .collect();
    assert_eq!(got.notation, notation, "{name}: notation");
    same_reconstruction(name, &got.reconstruction, &want["reconstruction"]);
    let agree = want["frameAgreement"].as_array().unwrap();
    assert_eq!(got.frame_agreement.len(), agree.len(), "{name}: frame agreement");
    for ((t, a), w) in got.frame_agreement.iter().zip(agree) {
        assert_eq!(*t, num(&w["t"]), "{name}: frame agreement t");
        match (a, w["agree"].is_null()) {
            (None, true) => {}
            (Some(a), false) => assert!(close(*a, num(&w["agree"])), "{name} at {t}: {a} against {}", w["agree"]),
            _ => panic!("{name} at {t}: {a:?} against {}", w["agree"]),
        }
    }
}

#[test]
fn results_match_typescript() {
    for (name, video) in video_fixtures() {
        let frames = frames(&video["frames"]);
        // The readings the decoder gets, as mergeReadings made them
        let (merged, of) = merge_readings(&frames);
        assert_eq!(merged, readings(&video["readings"]), "{name}: readings");
        assert_eq!(of.len(), frames.len());
        let got = analyze_readings(&frames, video["scramble"].as_str().unwrap(), &AnalyzeOptions::default()).expect("a result");
        same_result(&name, &got, &video["result"]);
    }
}

#[test]
fn no_face_read_gives_none() {
    let blank = |t: f64| FrameReadings { t, faces: Vec::<FaceReading>::new(), orientation: None, turning: None, motion: Motion::Unknown };
    assert!(analyze_readings(&[], "R U", &AnalyzeOptions::default()).is_none());
    assert!(analyze_readings(&[blank(0.0), blank(33.0)], "R U", &AnalyzeOptions::default()).is_none());
}

#[test]
fn frames_at_one_time_or_going_back_do_not_panic() {
    let shuffled = fixture("shuffled.json");
    let frames = frames(&shuffled["frames"]);
    assert_eq!(frames[10].t, frames[9].t);
    assert!(frames[40].t < frames[39].t);
    let got = analyze_readings(&frames, shuffled["scramble"].as_str().unwrap(), &AnalyzeOptions::default()).expect("a result");
    same_result("shuffled", &got, &shuffled["result"]);
}
