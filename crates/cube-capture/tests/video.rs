//! Frames out of video files; the test videos are made by ffmpeg in a temporary folder

use std::path::{Path, PathBuf};
use std::process::Command;

use cube_capture::{Image, VideoOptions, VideoReader};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cube-capture-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn ffmpeg(args: &[&str]) {
    let status = Command::new("ffmpeg").args(["-v", "error", "-y"]).args(args).status().expect("ffmpeg is needed by these tests");
    assert!(status.success(), "ffmpeg {args:?}");
}

/// One second of ffmpeg's test pattern at 30 fps, index at the start of the file
fn test_video(name: &str, size: &str) -> PathBuf {
    let path = scratch(name);
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={size}:rate=30:duration=1"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        path.to_str().unwrap(),
    ]);
    path
}

fn all(path: &Path, options: &VideoOptions) -> Vec<Image> {
    let mut reader = VideoReader::open(path, options).unwrap();
    let mut images = Vec::new();
    while let Some(image) = reader.next_image().unwrap() {
        images.push(image);
    }
    images
}

#[test]
fn reads_every_frame() {
    let path = test_video("plain.mp4", "320x240");
    let reader = VideoReader::open(&path, &VideoOptions::default()).unwrap();
    let info = reader.info();
    assert_eq!((info.width, info.height, info.fps, info.codec.as_str(), info.rotation), (320, 240, 30.0, "h264", 0));
    assert!((info.duration.unwrap() - 1.0).abs() < 0.05, "{:?}", info.duration);
    let images = all(&path, &VideoOptions::default());
    assert_eq!(images.len(), 30);
    assert!(images.iter().all(|f| f.data.len() == 320 * 240 * 4 && (f.width, f.height) == (320, 240)));
    // RGBA, opaque; the test pattern is not one flat colour
    assert!(images[0].data.chunks(4).all(|px| px[3] == 255));
    assert!(images[0].data.chunks(4).any(|px| px[..3] != images[0].data[..3]));
    // The times are the frames' own, in ms
    let times: Vec<f64> = images.iter().map(|f| f.t).collect();
    assert_eq!(times[..4], [0.0, 33.0, 67.0, 100.0]);
    assert_eq!(times[29], 967.0);
}

#[test]
fn a_wide_video_is_scaled_to_1280() {
    let path = test_video("wide.mp4", "1920x1080");
    let reader = VideoReader::open(&path, &VideoOptions::default()).unwrap();
    assert_eq!((reader.info().width, reader.info().height), (1280, 720));
    let images = all(&path, &VideoOptions::default());
    assert_eq!(images.len(), 30);
    assert!(images.iter().all(|f| (f.width, f.height) == (1280, 720) && f.data.len() == 1280 * 720 * 4));
    // As filmed when asked
    let full = VideoOptions { max_width: None, ..VideoOptions::default() };
    assert_eq!(all(&path, &full).iter().map(|f| (f.width, f.height, f.data.len())).next(), Some((1920, 1080, 1920 * 1080 * 4)));
    // The height is even, as the site makes it: 853 rows become 854
    let odd = test_video("odd.mp4", "1920x1280");
    let narrow = VideoOptions { max_width: Some(1279), ..VideoOptions::default() };
    assert_eq!(all(&odd, &narrow).iter().map(|f| (f.width, f.height)).next(), Some((1279, 852)));
}

#[test]
fn rotated_video_swaps_size() {
    let input = test_video("upright.mp4", "320x240");
    let out = scratch("rotated.mp4");
    ffmpeg(&["-display_rotation", "90", "-i", input.to_str().unwrap(), "-c", "copy", out.to_str().unwrap()]);
    let reader = VideoReader::open(&out, &VideoOptions::default()).unwrap();
    assert_eq!((reader.info().width, reader.info().height), (240, 320));
    assert_eq!(reader.info().rotation.abs(), 90);
    let images = all(&out, &VideoOptions::default());
    assert_eq!(images.len(), 30);
    assert!(images.iter().all(|f| (f.width, f.height) == (240, 320) && f.data.len() == 240 * 320 * 4));
}

#[test]
fn truncated_video_ends_cleanly() {
    let path = test_video("whole.mp4", "320x240");
    let bytes = std::fs::read(&path).unwrap();
    let cut = scratch("cut.mp4");
    std::fs::write(&cut, &bytes[..bytes.len() * 6 / 10]).unwrap();
    let images = all(&cut, &VideoOptions::default());
    assert!(!images.is_empty() && images.len() < 30, "{} frames", images.len());
    assert!(images.iter().all(|f| f.data.len() == 320 * 240 * 4));
}

#[test]
fn missing_ffmpeg_is_explained() {
    let path = test_video("any.mp4", "320x240");
    let options = VideoOptions { ffmpeg: "ffmpeg-absent".into(), ..VideoOptions::default() };
    let error = format!("{:#}", VideoReader::open(&path, &options).err().expect("no ffmpeg, no video"));
    assert!(error.contains("ffmpeg") && error.contains("PATH"), "{error}");
}

#[test]
fn a_file_that_is_no_video_is_explained() {
    let text = scratch("notes.mp4");
    std::fs::write(&text, "not a video").unwrap();
    let error = format!("{:#}", VideoReader::open(&text, &VideoOptions::default()).err().expect("no video in it"));
    assert!(error.contains("notes.mp4"), "{error}");
    let missing = scratch("nowhere.mp4");
    let error = format!("{:#}", VideoReader::open(&missing, &VideoOptions::default()).err().expect("no file"));
    assert!(error.contains("nowhere.mp4"), "{error}");
    // Sound only: no video stream
    let sound = scratch("sound.m4a");
    ffmpeg(&["-f", "lavfi", "-i", "sine=duration=1", sound.to_str().unwrap()]);
    let error = format!("{:#}", VideoReader::open(&sound, &VideoOptions::default()).err().expect("no picture"));
    assert!(error.contains("no video"), "{error}");
}

#[test]
fn frame_times_follow_the_timestamps() {
    // A pause of five frames after every ten: the times are not those of a steady 30 fps
    let path = scratch("uneven.mp4");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=30:duration=1",
        "-vf",
        "setpts=(N+floor(N/10)*5)/30/TB",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "yuv420p",
        path.to_str().unwrap(),
    ]);
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0"])
        .arg(&path)
        .output()
        .unwrap();
    let want: Vec<f64> = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .filter_map(|l| l.trim_end_matches(',').parse::<f64>().ok())
        .map(|s| (s * 1000.0 + 0.5).floor())
        .collect();
    let times: Vec<f64> = all(&path, &VideoOptions::default()).iter().map(|f| f.t).collect();
    assert_eq!(times.len(), 30);
    assert_eq!(times, want);
    assert_eq!((times[9], times[10]), (300.0, 500.0), "{times:?}");
}

#[test]
fn a_reader_dropped_early_stops_ffmpeg() {
    let path = test_video("left.mp4", "1920x1080");
    let mut reader = VideoReader::open(&path, &VideoOptions::default()).unwrap();
    assert!(reader.next_image().unwrap().is_some());
    drop(reader); // must not hang on ffmpeg's full pipe
}

#[test]
fn a_title_that_looks_like_a_frame_time_is_not_one() {
    // ffmpeg prints the file's tags on the stream the frame times come by
    let path = scratch("titled.mp4");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=30:duration=1",
        "-pix_fmt",
        "yuv420p",
        "-metadata",
        "title=pts_time:99 hello",
        path.to_str().unwrap(),
    ]);
    let times: Vec<f64> = all(&path, &VideoOptions::default()).iter().map(|f| f.t).collect();
    assert_eq!(times.len(), 30);
    assert_eq!(times[..4], [0.0, 33.0, 67.0, 100.0], "{times:?}");
}

#[test]
fn a_video_is_read_from_where_asked_with_its_own_times() {
    let path = test_video("later.mp4", "320x240");
    let from = VideoOptions { start: 0.5, ..VideoOptions::default() };
    let images = all(&path, &from);
    // Half of the 30 frames, the first one at half a second of the video
    assert!((14..=16).contains(&images.len()), "{} frames", images.len());
    assert!((images[0].t - 500.0).abs() <= 34.0, "starts at {} ms", images[0].t);
    assert_eq!(images.last().unwrap().t, 967.0);
    assert!(images.windows(2).all(|w| w[1].t > w[0].t));
    // Past the end: no frame, said as such; a start that is no number is the beginning
    let past = VideoOptions { start: 30.0, ..VideoOptions::default() };
    let mut reader = VideoReader::open(&path, &past).unwrap();
    assert!(reader.next_image().is_err());
    assert_eq!(all(&path, &VideoOptions { start: f64::NAN, ..VideoOptions::default() }).len(), 30);
}
