//! A solve found in a video: its frames read with the models, then the moves decoded from the readings. What
//! `cubevision video` and the window's video mode both do.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use cube_capture::{VideoInfo, VideoOptions, VideoReader};
use cube_decode::pipeline::{AnalyzeOptions, VisionResult, analyze_readings};
use cube_infer::device::Device;
use cube_infer::models::{BATCH, Models};

use crate::read::FrameLoop;

pub struct Request {
    pub video: PathBuf,
    /// In WCA notation, checked already
    pub scramble: String,
    /// How the cube was held to scramble (a rotation from white on top, green in front)
    pub hold: &'static str,
    /// The cross's colour, when told
    pub cross: Option<u8>,
    pub ffmpeg: String,
    /// Read frame by frame, the detector looking at one frame in so many (None: batches, every frame)
    pub detect_every: Option<u32>,
    /// Also write the frames as they were read to this file (JSON)
    pub dump: Option<PathBuf>,
}

impl Request {
    /// A video and its scramble, the cube scrambled white on top and green in front
    pub fn new(video: PathBuf, scramble: String) -> Self {
        Request { video, scramble, hold: "", cross: None, ffmpeg: "ffmpeg".into(), detect_every: None, dump: None }
    }
}

/// How far the work is
#[derive(Clone, Debug, PartialEq)]
pub enum Progress {
    Loading,
    /// Frames read so far, and how many the file says it holds
    Reading {
        read: usize,
        all: Option<u64>,
    },
    /// The moves are being found: all the frames are read
    Decoding {
        read: usize,
        seconds: f64,
    },
}

pub struct Solve {
    pub info: VideoInfo,
    pub device: String,
    /// Why the models do not run where asked, when they fell back
    pub note: Option<String>,
    /// Frames where the cube was found
    pub found: usize,
    pub mirrored: bool,
    pub result: VisionResult,
    /// The scramble does not seem to be this video's: the moves are not to be given as a solve
    pub mismatch: bool,
    pub warning: Option<&'static str>,
    pub read_seconds: f64,
    pub decode_seconds: f64,
}

/// Reads the video and finds its moves. `progress` is told how far the work is, a few times a second while
/// frames are read; false from it stops the reading (None is returned).
pub fn solve(
    request: &Request,
    models: &Path,
    device: Option<Device>,
    progress: &mut dyn FnMut(Progress) -> bool,
) -> Result<Option<Solve>> {
    let mut source = VideoReader::open(&request.video, &VideoOptions { ffmpeg: request.ffmpeg.clone(), ..VideoOptions::default() })?;
    let info = source.info().clone();
    if !progress(Progress::Loading) {
        return Ok(None);
    }
    let mut models = Models::load(models, device)?;
    let batches = request.detect_every.is_none();
    if batches {
        models.warm(BATCH)?;
    }
    let Models { detector, reader, device, fallback, .. } = models;
    let mut frames = FrameLoop::new(detector, reader, &request.scramble, Some(request.hold), request.detect_every.unwrap_or(1).max(1));

    let reading = Instant::now();
    let mut batch = Vec::with_capacity(BATCH);
    let (mut told, mut count) = (Instant::now(), 0usize);
    while let Some(image) = source.next_image()? {
        count += 1;
        if told.elapsed().as_secs_f64() >= 0.2 {
            told = Instant::now();
            if !progress(Progress::Reading { read: count, all: info.frames }) {
                return Ok(None);
            }
        }
        if batches {
            batch.push(image);
            if batch.len() == BATCH {
                frames.push_batch(&batch)?;
                batch.clear();
            }
        } else {
            frames.push(&image)?;
        }
    }
    frames.push_batch(&batch)?;
    let read = frames.finish();
    let read_seconds = reading.elapsed().as_secs_f64();
    if read.frames.is_empty() {
        bail!("{} holds no frame", request.video.display());
    }
    if let Some(file) = &request.dump {
        std::fs::write(file, serde_json::to_string(&crate::json::frames(&read.frames))?)
            .with_context(|| format!("writing {}", file.display()))?;
    }
    if !progress(Progress::Decoding { read: count, seconds: read_seconds }) {
        return Ok(None);
    }
    let decoding = Instant::now();
    let options = AnalyzeOptions { hold: Some(request.hold.to_string()), cross: request.cross, ..AnalyzeOptions::default() };
    let Some(result) = analyze_readings(&read.frames, &request.scramble, &options) else {
        bail!(
            "no face of a cube could be read in {} ({} frames, the cube found in {})",
            request.video.display(),
            read.frames.len(),
            read.found
        );
    };
    let decode_seconds = decoding.elapsed().as_secs_f64();
    // A scramble that is not the video's: the readings fit no cube that follows from it
    let mismatch = crate::json::mismatch(&result);
    let warning = if mismatch {
        Some("the scramble does not seem to be this video's: the moves found explain too little of what was read, and are not given")
    } else if !result.solved {
        Some("the moves found do not end on a solved cube: some turns could not be followed (see gaps)")
    } else {
        None
    };
    Ok(Some(Solve {
        info,
        device: device.name().to_owned(),
        note: fallback,
        found: read.found,
        mirrored: read.mirrored,
        result,
        mismatch,
        warning,
        read_seconds,
        decode_seconds,
    }))
}
