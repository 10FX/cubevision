//! Frames out of a video file through ffmpeg; test videos are made by ffmpeg in a temporary folder

use std::path::{Path, PathBuf};
use std::process::Command;

use spike::models::{Device, Models, Precision};
use spike::pipeline::process_frame;
use spike::preprocess::Frame;
use spike::video::VideoReader;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("spike-video-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn ffmpeg(args: &[&str]) {
    let status = Command::new("ffmpeg").args(["-v", "error", "-y"]).args(args).status().unwrap();
    assert!(status.success(), "ffmpeg {args:?}");
}

/// One second of ffmpeg's test pattern, 320×240 at 30 fps, index at the start of the file
fn test_video(name: &str) -> PathBuf {
    let path = scratch(name);
    ffmpeg(&["-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path.to_str().unwrap()]);
    path
}

fn all_frames(path: &Path) -> Vec<Frame> {
    let mut reader = VideoReader::open(path, "ffmpeg").unwrap();
    let mut frames = Vec::new();
    while let Some(frame) = reader.next_frame().unwrap() {
        frames.push(frame);
    }
    frames
}

#[test]
fn reads_every_frame() {
    let path = test_video("plain.mp4");
    let reader = VideoReader::open(&path, "ffmpeg").unwrap();
    assert_eq!(reader.size(), (320, 240));
    assert_eq!(reader.fps(), 30.0);
    let frames = all_frames(&path);
    assert_eq!(frames.len(), 30);
    assert!(frames.iter().all(|f| f.data.len() == 320 * 240 * 3 && (f.width, f.height) == (320, 240)));
}

#[test]
fn rotated_video_swaps_size() {
    let input = test_video("upright.mp4");
    let out = scratch("rotated.mp4");
    ffmpeg(&["-display_rotation", "90", "-i", input.to_str().unwrap(), "-c", "copy", out.to_str().unwrap()]);
    assert_eq!(VideoReader::open(&out, "ffmpeg").unwrap().size(), (240, 320));
    let frames = all_frames(&out);
    assert_eq!(frames.len(), 30);
    assert!(frames.iter().all(|f| (f.width, f.height) == (240, 320) && f.data.len() == 240 * 320 * 3));
}

#[test]
fn truncated_video_ends_cleanly() {
    let path = test_video("whole.mp4");
    let bytes = std::fs::read(&path).unwrap();
    let cut = scratch("cut.mp4");
    std::fs::write(&cut, &bytes[..bytes.len() * 6 / 10]).unwrap();
    let frames = all_frames(&cut);
    assert!(!frames.is_empty() && frames.len() < 30, "{} frames", frames.len());
    assert!(frames.iter().all(|f| f.data.len() == 320 * 240 * 3));
}

#[test]
fn missing_ffmpeg_is_explained() {
    let path = test_video("any.mp4");
    let error = VideoReader::open(&path, "ffmpeg-absent").err().expect("no ffmpeg, no video");
    assert!(error.to_string().contains("ffmpeg"), "{error}");
}

#[test]
fn without_a_cube_the_reader_reads_the_centre() {
    // needs ORT_DYLIB_PATH and CUBE_MODELS, as tests/parity.rs
    let dir = PathBuf::from(std::env::var("CUBE_MODELS").expect("CUBE_MODELS names the models' folder"));
    let mut models = Models::load(&dir, Device::Cpu, Precision::Fp32).unwrap();
    let frame = Frame { data: vec![0; 640 * 480 * 3], width: 640, height: 480 };
    let mut last = None;
    let times = process_frame(&mut models, &frame, &mut last).unwrap();
    assert!(!times.found);
    assert!(last.is_none());
    assert!(times.reader_ms > 0.0 && times.detector_ms > 0.0);
}

#[test]
fn a_batch_finds_what_frame_by_frame_finds() {
    // needs ORT_DYLIB_PATH and CUBE_MODELS; the synthetic test video stands in for a solve
    let dir = PathBuf::from(std::env::var("CUBE_MODELS").expect("CUBE_MODELS names the models' folder"));
    let video = Path::new("/home/thomas/TIMER/ml2/evals/synthetic-v3/synth-23-000.webm");
    let mut reader = VideoReader::open(video, "ffmpeg").unwrap();
    let frames: Vec<Frame> = (0..20).map(|_| reader.next_frame().unwrap().unwrap()).collect();
    let mut models = Models::load(&dir, Device::Cpu, Precision::Fp32).unwrap();
    let (mut one_last, mut batch_last) = (None, None);
    let one: Vec<bool> = frames.iter().map(|f| process_frame(&mut models, f, &mut one_last).unwrap().found).collect();
    let batched: Vec<bool> = frames.chunks(8).flat_map(|c| spike::pipeline::process_batch(&mut models, c, &mut batch_last).unwrap()).map(|t| t.found).collect();
    assert_eq!(one, batched);
    let (a, b) = (one_last.unwrap(), batch_last.unwrap());
    assert!((a.x - b.x).abs() < 0.5 && (a.w - b.w).abs() < 0.5, "{a:?} vs {b:?}");
}
