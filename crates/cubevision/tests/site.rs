//! The video mode against ChronoCube on the synthetic solves of the decoder's fixtures: the same frames read
//! by the same models. Needs the models, ONNX Runtime and the videos:
//!   CUBE_MODELS=… ORT_DYLIB_PATH=… CUBE_SYNTHETIC=<folder of synth-23-*.mp4/.webm> \
//!     cargo test --release -p cubevision --test site -- --include-ignored --nocapture

use std::path::PathBuf;

use cube_capture::{VideoOptions, VideoReader};
use cube_decode::pipeline::{AnalyzeOptions, analyze_readings};
use cube_infer::device::Device;
use cube_infer::models::Models;
use cubevision::read::FrameLoop;
use serde_json::Value;

fn fixtures() -> Vec<(String, Value)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../cube-decode/tests/fixtures/videos");
    let mut names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    names.sort();
    names
        .into_iter()
        .map(|p| {
            (p.file_stem().unwrap().to_string_lossy().into_owned(), serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap())
        })
        .collect()
}

fn centres(faces: &Value) -> Vec<u64> {
    faces.as_array().unwrap().iter().map(|f| f["center"].as_u64().unwrap()).collect()
}

#[test]
#[ignore = "needs CUBE_MODELS, ORT_DYLIB_PATH and CUBE_SYNTHETIC"]
fn frame_by_frame_on_the_processor_reads_as_the_site() {
    let models = PathBuf::from(std::env::var("CUBE_MODELS").expect("CUBE_MODELS names the models' folder"));
    let videos = PathBuf::from(std::env::var("CUBE_SYNTHETIC").expect("CUBE_SYNTHETIC names the folder of the synthetic videos"));
    let mut identical = 0;
    for (name, fixture) in fixtures() {
        let video = ["mp4", "webm"]
            .iter()
            .map(|ext| videos.join(format!("{name}.{ext}")))
            .find(|p| p.is_file())
            .unwrap_or_else(|| panic!("no video for {name}"));
        let scramble = fixture["scramble"].as_str().unwrap();
        // As the site's evaluation read them: on the processor, one frame at a time, the detector one frame in 3
        let Models { detector, reader, .. } = Models::load(&models, Some(Device::Cpu)).unwrap();
        let mut frames = FrameLoop::new(detector, reader, scramble, None, 3);
        let mut source = VideoReader::open(&video, &VideoOptions::default()).unwrap();
        while let Some(image) = source.next_image().unwrap() {
            frames.push(&image).unwrap();
        }
        let read = frames.finish();
        let site = fixture["frames"].as_array().unwrap();
        assert_eq!(read.frames.len(), site.len(), "{name}: frames");
        assert!(read.frames.iter().zip(site).all(|(a, b)| a.t == b["t"].as_f64().unwrap()), "{name}: times");
        let same = read
            .frames
            .iter()
            .zip(site)
            .filter(|(a, b)| a.faces.iter().map(|f| f.center as u64).collect::<Vec<_>>() == centres(&b["faces"]))
            .count();
        // The first second is the site's to the last digit shown: the same pixels through the same models
        for (i, (a, b)) in read.frames.iter().zip(site).take(30).enumerate() {
            if let (Some(turning), Some(want)) = (a.turning, b["turning"].as_f64()) {
                assert!((turning - want).abs() < 1e-4, "{name} frame {i}: turning {turning}, the site {want}");
            }
        }
        let result = analyze_readings(&read.frames, scramble, &AnalyzeOptions::default()).unwrap();
        let want = &fixture["result"];
        let solution = want["reconstruction"]["solution"].as_str().unwrap();
        println!(
            "{name}: {same}/{} frames with the site's faces; solved {} (site {}); agreement {:.3} (site {:.3}); same solution: {}",
            site.len(),
            result.solved,
            want["solved"],
            result.agreement,
            want["agreement"].as_f64().unwrap(),
            result.reconstruction.solution == solution
        );
        assert_eq!(result.solved, want["solved"].as_bool().unwrap(), "{name}: solved");
        // Later a box a few pixels off (native engine against WebAssembly) may shift some readings: most stay
        assert!(same * 10 >= site.len() * 8, "{name}: {same}/{} frames with the site's faces", site.len());
        if same == site.len() {
            identical += 1;
            assert_eq!(result.reconstruction.solution, solution, "{name}: the same readings, the same solution");
        }
    }
    assert!(identical >= 1, "no video read exactly as the site");
}
