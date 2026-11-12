//! A video read from end to end: each frame's cube found, cropped and read, in batches as the video mode will.
//!
//!   CUBE_MODELS=<models' folder> ORT_DYLIB_PATH=<ONNX Runtime's library> \
//!     cargo run --release -p cube-infer --example read_video -- solve.mp4 [--device cuda]
//!
//! Prints what was read and how fast. The moves come later, from cube-decode.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use cube_capture::{Image, VideoOptions, VideoReader};
use cube_infer::device::Device;
use cube_infer::follow::CubeFollower;
use cube_infer::geometry::square_crop;
use cube_infer::models::{BATCH, Models};
use cube_infer::output::{TURNING, readings_from_output};
use cube_infer::{BoxF, Frame};
use rayon::prelude::*;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut video, mut device) = (None, None);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => device = Some(args.next().context("--device takes a name")?.parse::<Device>()?),
            _ => video = Some(PathBuf::from(arg)),
        }
    }
    let Some(video) = video else { bail!("usage: read_video <video> [--device {}]", Device::names().replace(", ", "|")) };
    let dir = PathBuf::from(std::env::var_os("CUBE_MODELS").context("CUBE_MODELS names the models' folder")?);

    let loading = Instant::now();
    let mut models = Models::load(&dir, device)?;
    models.warm(BATCH)?;
    if let Some(why) = &models.fallback {
        eprintln!("note: {why}");
    }
    let Models { detector, mut reader, device, .. } = models;
    let loaded = loading.elapsed().as_secs_f64();
    let mut follower = CubeFollower::new(detector, 1);

    let mut source = VideoReader::open(&video, &VideoOptions::default())?;
    let info = source.info().clone();
    let (mut frames, mut found, mut read, mut turning, mut faces) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let reading = Instant::now();
    loop {
        let mut images: Vec<Image> = Vec::with_capacity(BATCH);
        while images.len() < BATCH {
            match source.next_image()? {
                Some(image) => images.push(image),
                None => break,
            }
        }
        if images.is_empty() {
            break;
        }
        let views: Vec<Frame> = images.iter().map(|i| Frame { data: &i.data, width: i.width, height: i.height }).collect();
        let boxes = follower.crop_batch(&views)?;
        // The cube, cropped at full resolution: its seams survive
        let seen: Vec<(&Frame, &BoxF)> = views.iter().zip(&boxes).filter_map(|(frame, boxed)| boxed.as_ref().map(|b| (frame, b))).collect();
        let crops: Vec<Vec<u8>> = seen.par_iter().map(|(frame, boxed)| square_crop(frame, boxed, reader.size())).collect();
        let outputs = reader.read(&crops.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        frames += images.len();
        found += crops.len();
        for out in &outputs {
            let readings = readings_from_output(out, reader.ordered(), false);
            turning += (out.turning > TURNING) as usize;
            read += !readings.is_empty() as usize;
            faces += readings.len();
        }
    }
    let seconds = reading.elapsed().as_secs_f64();
    let share = |n: usize| 100.0 * n as f64 / frames.max(1) as f64;
    println!("{}: {}×{} at {:.2} fps, {}", video.display(), info.width, info.height, info.fps, info.codec);
    println!("device: {} (models loaded and warmed in {loaded:.1} s)", device.name());
    println!(
        "frames: {frames} in {seconds:.2} s: {:.0} frames a second, {:.1} times the video's rate",
        frames as f64 / seconds,
        frames as f64 / seconds / info.fps
    );
    println!(
        "cube found: {found} ({:.1} %); mid-turn: {turning} ({:.1} %); read: {read} ({:.1} %), {:.2} faces a frame read",
        share(found),
        share(turning),
        share(read),
        faces as f64 / read.max(1) as f64
    );
    Ok(())
}
