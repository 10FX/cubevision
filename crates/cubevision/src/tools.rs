//! `cubevision bench` and `cubevision eval`: how long each step takes on a video, and a reconstruction scored
//! against the truth of a synthetic solve

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use cube_capture::{Image, VideoInfo, VideoOptions, VideoReader};
use cube_core::Cube;
use cube_decode::delayed::DelayedTracker;
use cube_decode::pipeline::{AnalyzeOptions, VisionResult, analyze_readings};
use cube_decode::state::state_of;
use cube_decode::truth::{Truth, score_against_truth};
use cube_infer::device::Device;
use cube_infer::follow::CubeFollower;
use cube_infer::geometry::square_crop;
use cube_infer::models::{BATCH, Models};
use cube_infer::output::{Reader, readings_from_output};
use cube_infer::{BoxF, Frame};
use rayon::prelude::*;
use serde_json::Value;

use crate::read::{FrameLoop, Read};

/// A video read in batches, as the video mode reads it: what it is, its frames read, where, in how many seconds
pub fn read_video(
    video: &Path,
    ffmpeg: &str,
    mut models: Models,
    scramble: &str,
    hold: Option<&str>,
) -> Result<(VideoInfo, Read, Device, f64)> {
    let mut source = VideoReader::open(video, &VideoOptions { ffmpeg: ffmpeg.to_owned(), ..VideoOptions::default() })?;
    let info = source.info().clone();
    models.warm(BATCH)?;
    let Models { detector, reader, device, .. } = models;
    let mut frames = FrameLoop::new(detector, reader, scramble, hold, 1);
    let reading = Instant::now();
    let mut batch = Vec::with_capacity(BATCH);
    while let Some(image) = source.next_image()? {
        batch.push(image);
        if batch.len() == BATCH {
            frames.push_batch(&batch)?;
            batch.clear();
        }
    }
    frames.push_batch(&batch)?;
    Ok((info, frames.finish(), device, reading.elapsed().as_secs_f64()))
}

/// The truth of a synthetic solve, as the renderer writes it next to the video (`<video>.truth.json`)
pub fn truth(text: &str) -> Result<Truth> {
    let json: Value = serde_json::from_str(text).context("not JSON")?;
    let string = |v: &Value, what: &str| v.as_str().map(str::to_owned).with_context(|| format!("no {what}"));
    let number = |v: &Value, what: &str| v.as_f64().filter(|x| x.is_finite()).with_context(|| format!("no {what}"));
    let moves = json["moves"]
        .as_array()
        .context("no moves")?
        .iter()
        .map(|m| {
            Ok((
                string(&m["move"], "move")?,
                number(&m["start"], "start of a move")?,
                number(&m["end"], "end of a move")?,
                string(&m["stage"], "stage of a move")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let stages = ["cross", "f2l", "oll", "pll"].map(|stage| number(&json["stages"][stage], stage));
    let [cross, f2l, oll, pll] = stages;
    Ok(Truth {
        scramble: string(&json["scramble"], "scramble")?,
        solution: string(&json["solution"], "solution")?,
        moves,
        stages: [cross?, f2l?, oll?, pll?],
    })
}

/// A video's reconstruction scored against its truth, printed
pub fn eval(video: &Path, truth_file: &Path, ffmpeg: &str, models: Models) -> Result<bool> {
    let text = std::fs::read_to_string(truth_file).with_context(|| format!("reading {}", truth_file.display()))?;
    let truth = truth(&text).with_context(|| format!("{} is not the truth of a solve", truth_file.display()))?;
    let (_, read, device, seconds) = read_video(video, ffmpeg, models, &truth.scramble, None)?;
    let Some(result) = analyze_readings(&read.frames, &truth.scramble, &AnalyzeOptions::default()) else {
        bail!("no face of a cube could be read in {}", video.display());
    };
    let score = score_against_truth(&result, &truth);
    println!("{}: read on {} in {seconds:.1} s, {} of {} frames read", video.display(), device.name(), result.read_frames, result.frames);
    println!("solved: {}; stickers read that fit the moves: {:.1} %", result.solved, result.agreement * 100.0);
    println!("quarter turns played: {}; missing, added or wrong: {}; exact: {:.1} %", score.quarters, score.distance, score.exact * 100.0);
    match score.time_median {
        Some(gap) => println!("median gap between a move found and the move played: {gap:.0} ms"),
        None => println!("no move found matches one played"),
    }
    for (stage, gap) in ["cross", "f2l", "oll", "pll"].iter().zip(score.stages) {
        match gap {
            Some(gap) => println!("  {stage:<5} ends {gap:+.0} ms from the truth"),
            None => println!("  {stage:<5} not found"),
        }
    }
    let (played, found) = score.rotations;
    println!("rotations of the whole cube: {played} played, {}", found.map_or("none read".to_owned(), |n| format!("{n} found")));
    Ok(result.solved)
}

/// One step's time over a video
struct Step {
    name: &'static str,
    seconds: f64,
}

/// How long each step takes on a video, printed as a table: the same video and device give the same table,
/// but for the machine's own jitter
pub fn bench(video: &Path, scramble: &str, ffmpeg: &str, mut models: Models) -> Result<()> {
    let mut source = VideoReader::open(video, &VideoOptions { ffmpeg: ffmpeg.to_owned(), ..VideoOptions::default() })?;
    let info = source.info().clone();
    models.warm(BATCH)?;
    let Models { detector, mut reader, device, .. } = models;
    let mut follower = CubeFollower::new(detector, 1);
    let (mut decode, mut detect, mut crop, mut read) = (0.0, 0.0, 0.0, 0.0);
    let mut frames = Vec::new();
    let mut found = 0usize;
    loop {
        let t = Instant::now();
        let mut images: Vec<Image> = Vec::with_capacity(BATCH);
        while images.len() < BATCH {
            match source.next_image()? {
                Some(image) => images.push(image),
                None => break,
            }
        }
        decode += t.elapsed().as_secs_f64();
        if images.is_empty() {
            break;
        }
        let views: Vec<Frame> = images.iter().map(|i| Frame { data: &i.data, width: i.width, height: i.height }).collect();
        let t = Instant::now();
        let boxes = follower.crop_batch(&views)?;
        detect += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let seen: Vec<(&Frame, &BoxF)> = views.iter().zip(&boxes).filter_map(|(frame, boxed)| boxed.as_ref().map(|b| (frame, b))).collect();
        let crops: Vec<Vec<u8>> = seen.par_iter().map(|(frame, boxed)| square_crop(frame, boxed, reader.size())).collect();
        crop += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let outputs = reader.read(&crops.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        read += t.elapsed().as_secs_f64();
        found += crops.len();
        let mut outputs = outputs.iter();
        for (image, boxed) in images.iter().zip(&boxes) {
            let out = boxed.as_ref().and_then(|_| outputs.next());
            frames.push(cube_decode::FrameReadings {
                t: image.t,
                faces: out.map(|o| readings_from_output(o, reader.ordered(), false)).unwrap_or_default(),
                orientation: out.and_then(|o| o.orientation64()),
                turning: out.map(|o| o.turning as f64),
                motion: cube_decode::Motion::NotMeasured,
            });
        }
    }
    if frames.is_empty() {
        bail!("{} holds no frame", video.display());
    }
    let t = Instant::now();
    let mut tracker = DelayedTracker::new(state_of(&Cube::solved().apply(scramble)));
    frames.iter().for_each(|f| drop(tracker.update(&f.faces)));
    let track = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let result: Option<VisionResult> = analyze_readings(&frames, scramble, &AnalyzeOptions::default());
    let search = t.elapsed().as_secs_f64();

    let n = frames.len() as f64;
    let steps = [
        Step { name: "capture (ffmpeg, decoding)", seconds: decode },
        Step { name: "detector (letterbox, model)", seconds: detect },
        Step { name: "crops", seconds: crop },
        Step { name: "reader (model)", seconds: read },
        Step { name: "live tracker", seconds: track },
        Step { name: "video decoder (search)", seconds: search },
    ];
    let reading: f64 = steps[..4].iter().map(|s| s.seconds).sum();
    println!(
        "{}: {}×{} at {:.2} fps, {} frames, {:.1} s of video",
        video.display(),
        info.width,
        info.height,
        info.fps,
        frames.len(),
        n / info.fps
    );
    println!("device: {}; batches of {BATCH}; cube found in {found} frames", device.name());
    println!("{:<30} {:>9} {:>12}", "step", "seconds", "ms a frame");
    for step in &steps {
        println!("{:<30} {:>9.2} {:>12.2}", step.name, step.seconds, step.seconds * 1000.0 / n);
    }
    println!(
        "{:<30} {:>9.2} {:>12.2}   {:.0} frames a second, {:.1} times the video's rate",
        "reading, in all",
        reading,
        reading * 1000.0 / n,
        n / reading,
        n / reading / info.fps
    );
    match result {
        Some(result) => println!("result: {} moves, solved: {}", result.reconstruction.moves, result.solved),
        None => println!("result: no face read"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_truth_file_is_read() {
        let text = r#"{"video":"s.mp4","scramble":"R U","solution":"x2 U' R'",
            "moves":[{"move":"x2","start":1038,"end":1444,"stage":"cross"},{"move":"U'","start":1500,"end":1600.5,"stage":"pll"}],
            "stages":{"cross":3416,"f2l":8308,"oll":10760,"pll":12445},"rotations":1}"#;
        let t = truth(text).unwrap();
        assert_eq!((t.scramble.as_str(), t.solution.as_str(), t.stages), ("R U", "x2 U' R'", [3416.0, 8308.0, 10760.0, 12445.0]));
        assert_eq!(t.moves, [("x2".to_owned(), 1038.0, 1444.0, "cross".to_owned()), ("U'".to_owned(), 1500.0, 1600.5, "pll".to_owned())]);
        // What is missing is named
        assert!(format!("{:#}", truth(&text.replace(r#""f2l":8308,"#, "")).unwrap_err()).contains("f2l"));
        assert!(format!("{:#}", truth(&text.replace(r#""scramble":"R U","#, "")).unwrap_err()).contains("scramble"));
        assert!(truth("[]").is_err() && truth("{").is_err());
    }
}
