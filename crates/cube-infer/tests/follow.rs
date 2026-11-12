mod common;

use anyhow::Result;
use common::*;
use cube_infer::follow::{BoxTracker, CubeFollower, Detection, Detector, Look, SMALL_CUBE, ZOOM, detect_with_zoom, is_small, zoom_region};
use cube_infer::{BoxF, Frame};
use serde_json::Value;

fn same(got: &Option<BoxF>, want: &Value) -> bool {
    match (got, boxf(want)) {
        (None, None) => true,
        (Some(a), Some(b)) => same_box(a, &b),
        _ => false,
    }
}

#[test]
fn the_tracker_matches_typescript() {
    for (k, steps) in fixture("follow.json")["trackers"].as_array().unwrap().iter().enumerate() {
        let mut tracker = BoxTracker::default();
        for (i, step) in steps.as_array().unwrap().iter().enumerate() {
            let what = format!("tracker {k} step {i}");
            if step["reset"].as_bool() == Some(true) {
                tracker.reset();
            } else {
                let out = tracker.update(boxf(&step["found"]));
                assert!(same(&out, &step["out"]), "{what}: {out:?} want {}", step["out"]);
            }
            assert!(same(&tracker.current(), &step["current"]), "{what}: current");
        }
    }
}

/// A detector that answers from a script: for each frame (its first byte is its index), what it sees looking
/// at the whole frame and looking closer; it notes every look asked
struct Scripted {
    answers: Vec<[Option<Detection>; 2]>,
    looks: Vec<(usize, Option<BoxF>)>,
}

impl Detector for Scripted {
    fn crop_margin(&self) -> f64 {
        1.5
    }

    fn look(&mut self, looks: &[Look]) -> Result<Vec<Option<Detection>>> {
        Ok(looks
            .iter()
            .map(|look| {
                let i = look.frame.data[0] as usize;
                self.looks.push((i, look.region));
                self.answers[i][look.region.is_some() as usize]
            })
            .collect())
    }
}

fn detection(v: &Value) -> Option<Detection> {
    (!v.is_null()).then(|| Detection { boxed: boxf(&v["box"]).unwrap(), score: num(&v["score"]) as f32 })
}

struct Scenario {
    what: String,
    kind: String,
    every: u32,
    batch: bool,
    width: usize,
    height: usize,
    answers: Vec<[Option<Detection>; 2]>,
    reads: Option<Vec<bool>>,
    boxes: Value,
    looks: Value,
}

fn scenarios() -> Vec<Scenario> {
    fixture("follow.json")["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| Scenario {
            what: format!("{} every {} by {}", s["kind"], s["every"], s["mode"]),
            kind: s["kind"].as_str().unwrap().into(),
            every: s["every"].as_u64().unwrap() as u32,
            batch: s["mode"] == "batch",
            width: s["width"].as_u64().unwrap() as usize,
            height: s["height"].as_u64().unwrap() as usize,
            answers: s["answers"].as_array().unwrap().iter().map(|a| [detection(&a["whole"]), detection(&a["zoom"])]).collect(),
            reads: s["reads"].as_array().map(|r| r.iter().map(|b| b.as_bool().unwrap()).collect()),
            boxes: s["boxes"].clone(),
            looks: s["looks"].clone(),
        })
        .collect()
}

/// A look asked of the detector: the frame's index and the region looked at
type Asked = (usize, Option<BoxF>);

/// The scenario played: the reader's square in each frame, and the looks the detector was asked
fn play(s: &Scenario, batch: bool) -> (Vec<Option<BoxF>>, Vec<Asked>) {
    let data: Vec<[u8; 4]> = (0..s.answers.len()).map(|i| [i as u8, 0, 0, 255]).collect();
    let frames: Vec<Frame> = data.iter().map(|d| Frame { data: d, width: s.width, height: s.height }).collect();
    let mut follower = CubeFollower::new(Scripted { answers: s.answers.clone(), looks: Vec::new() }, s.every);
    let mut boxes = Vec::new();
    if batch {
        for chunk in frames.chunks(8) {
            boxes.extend(follower.crop_batch(chunk).unwrap());
        }
    } else {
        for (i, frame) in frames.iter().enumerate() {
            boxes.push(follower.crop(frame).unwrap());
            if let Some(reads) = &s.reads {
                follower.read(reads[i]);
            }
        }
    }
    (boxes, follower.detector().looks.clone())
}

#[test]
fn the_follower_matches_typescript() {
    let scenarios = scenarios();
    assert_eq!(scenarios.len(), 24);
    for s in &scenarios {
        let (boxes, looks) = play(s, s.batch);
        let want = s.boxes.as_array().unwrap();
        assert_eq!(boxes.len(), want.len(), "{}", s.what);
        for (i, (got, want)) in boxes.iter().zip(want).enumerate() {
            assert!(same(got, want), "{} frame {i}: {got:?} want {want}", s.what);
        }
        let asked = s.looks.as_array().unwrap();
        assert_eq!(looks.len(), asked.len(), "{}: looks", s.what);
        for (k, ((frame, region), want)) in looks.iter().zip(asked).enumerate() {
            assert_eq!(*frame as u64, want["frame"].as_u64().unwrap(), "{} look {k}: frame", s.what);
            assert!(same(region, &want["region"]), "{} look {k}: {region:?} want {}", s.what, want["region"]);
        }
    }
}

#[test]
fn a_batch_crops_as_frame_by_frame() {
    let mut seen = 0;
    for s in scenarios().iter().filter(|s| s.every == 1 && !s.batch) {
        let (one, _) = play(s, false);
        let (batched, _) = play(s, true);
        assert_eq!(one, batched, "{}", s.kind);
        seen += one.iter().flatten().count();
    }
    assert!(seen > 100, "the scenarios show a cube: {seen}");
}

#[test]
fn a_small_cube_is_looked_at_closer() {
    let frame = Frame { data: &[0, 0, 0, 255], width: 1920, height: 1080 };
    assert_eq!((SMALL_CUBE, ZOOM), (0.25, 4.0));
    assert!(is_small(&frame, &BoxF { x: 0.0, y: 0.0, w: 470.0, h: 100.0 }));
    assert!(!is_small(&frame, &BoxF { x: 0.0, y: 0.0, w: 100.0, h: 480.0 }));
    assert_eq!(zoom_region(&BoxF { x: 100.0, y: 100.0, w: 50.0, h: 40.0 }), BoxF { x: 25.0, y: 20.0, w: 200.0, h: 200.0 });

    let small = Detection { boxed: BoxF { x: 900.0, y: 500.0, w: 100.0, h: 90.0 }, score: 0.9 };
    let closer = Detection { boxed: BoxF { x: 905.0, y: 498.0, w: 96.0, h: 92.0 }, score: 0.8 };
    // Found small: the closer look wins; nothing closer: the first look stands
    let mut detector = Scripted { answers: vec![[Some(small), Some(closer)]], looks: Vec::new() };
    assert_eq!(detect_with_zoom(&mut detector, &frame, None).unwrap(), Some(closer));
    assert_eq!(detector.looks, [(0, None), (0, Some(zoom_region(&small.boxed)))]);
    let mut detector = Scripted { answers: vec![[Some(small), None]], looks: Vec::new() };
    assert_eq!(detect_with_zoom(&mut detector, &frame, None).unwrap(), Some(small));
    // Not found, a small box followed so far: looked at there; no hint: one look
    let mut detector = Scripted { answers: vec![[None, Some(closer)]], looks: Vec::new() };
    assert_eq!(detect_with_zoom(&mut detector, &frame, Some(&small.boxed)).unwrap(), Some(closer));
    assert_eq!(detect_with_zoom(&mut detector, &frame, None).unwrap(), None);
    assert_eq!(detector.looks.len(), 3);
}

#[test]
fn a_box_no_frame_can_hold_counts_as_no_cube() {
    let real = BoxF { x: 10.0, y: 10.0, w: 400.0, h: 400.0 };
    for odd in [f64::NAN, f64::INFINITY, 0.0, -3.0] {
        let mut tracker = BoxTracker::default();
        assert_eq!(tracker.update(Some(real)), Some(real));
        // Kept as when the cube is lost, never followed
        assert_eq!(tracker.update(Some(BoxF { w: odd, ..real })), Some(real), "width {odd}");
        assert_eq!(tracker.update(Some(BoxF { x: f64::NAN, ..real })), Some(real));
        assert_eq!(tracker.current(), Some(real));

        let frame = Frame { data: &[0, 0, 0, 255], width: 1280, height: 720 };
        let nowhere = Detection { boxed: BoxF { h: odd, ..real }, score: 0.9 };
        let mut follower = CubeFollower::new(Scripted { answers: vec![[Some(nowhere), Some(nowhere)]], looks: Vec::new() }, 1);
        assert_eq!(follower.crop(&frame).unwrap(), None, "height {odd}");
        assert_eq!(follower.crop_batch(&[frame, frame]).unwrap(), [None, None]);
    }
}

#[test]
fn a_cube_lost_is_kept_eight_frames() {
    let real = BoxF { x: 10.0, y: 10.0, w: 400.0, h: 400.0 };
    let mut tracker = BoxTracker::default();
    tracker.update(Some(real));
    for _ in 0..8 {
        assert_eq!(tracker.update(None), Some(real));
    }
    assert_eq!(tracker.update(None), None);
    assert_eq!(tracker.current(), None);
}
