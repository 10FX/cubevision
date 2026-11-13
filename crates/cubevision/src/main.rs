//! cubevision: a Rubik's Cube read from a video or a camera

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use cube_capture::{CameraRequest, VideoOptions, VideoReader};
use cube_decode::pipeline::{AnalyzeOptions, analyze_readings};
use cube_infer::device::Device;
use cube_infer::models::{BATCH, Models};
use cubevision::colours::{colour, hold};
use cubevision::live::{LiveOptions, Source};
use cubevision::read::FrameLoop;
use cubevision::replay_window;
use cubevision::window::Snapshot;
use serde_json::json;

#[derive(Parser)]
#[command(name = "cubevision", version, about = "Reads a Rubik's Cube from a camera or a video")]
struct Cli {
    /// Without a command: the live mode
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// The moves turned in front of the camera, as they are made, on a 3D cube (the default)
    Live(LiveArgs),
    /// The moves of a solve filmed, from its video and its scramble, played back on a 3D cube
    Video(VideoArgs),
    /// Tells, without a window, whether the models run on a device and how fast, and what the camera gives
    Check(CheckArgs),
    /// Downloads the released models into the models' folder, and checks them
    FetchModels(FetchArgs),
    /// Times each step on a video: capture, detector, crops, reader, live tracker, decoder
    Bench(BenchArgs),
    /// Scores a video's reconstruction against the truth of a synthetic solve
    Eval(EvalArgs),
}

#[derive(clap::Args, Default)]
struct Shared {
    /// Save a picture of the window after --after seconds and close it (to check it unattended)
    #[arg(long, value_name = "PNG", hide = true)]
    screenshot: Option<PathBuf>,
    #[arg(long, default_value_t = 6.0, hide = true)]
    after: f64,
    /// Where the models run: cpu, cuda, directml, openvino-cpu, openvino-gpu, openvino-npu (default: the first that works)
    #[arg(long)]
    device: Option<Device>,
    /// The folder of the models (default: `models` next to the program)
    #[arg(long, env = "CUBE_MODELS")]
    models: Option<PathBuf>,
}

#[derive(clap::Args)]
struct FetchArgs {
    /// The folder to fill (default: `models` next to the program)
    #[arg(long, env = "CUBE_MODELS")]
    models: Option<PathBuf>,
    /// Where the models' zip is
    #[arg(long, default_value = cubevision::fetch::URL)]
    url: String,
}

#[derive(clap::Args)]
struct BenchArgs {
    /// A video of a solve
    video: PathBuf,
    /// Its scramble
    #[arg(long)]
    scramble: String,
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: String,
    #[command(flatten)]
    shared: Shared,
}

#[derive(clap::Args)]
struct EvalArgs {
    /// A synthetic solve's video
    video: PathBuf,
    /// Its truth (default: the video's name with .truth.json)
    #[arg(long)]
    truth: Option<PathBuf>,
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: String,
    #[command(flatten)]
    shared: Shared,
}

#[derive(clap::Args)]
struct CheckArgs {
    /// Seconds the models are timed, and the camera read
    #[arg(long, default_value_t = 10.0)]
    seconds: f64,
    /// Also read the camera with the models, as the live mode does
    #[arg(long)]
    with_camera: bool,
    /// Which camera: 0 is the system's first
    #[arg(long, default_value_t = 0)]
    camera: u32,
    /// With the camera: save the crop where the reader saw the most of a cube to this PNG, to look at
    #[arg(long, value_name = "PNG")]
    save_crop: Option<PathBuf>,
    #[command(flatten)]
    shared: Shared,
}

#[derive(clap::Args, Default)]
struct LiveArgs {
    /// Which camera: 0 is the system's first
    #[arg(long, default_value_t = 0)]
    camera: u32,
    /// The size and rate asked of the camera; it gives the closest it has
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    #[arg(long, default_value_t = 30)]
    fps: u32,
    /// A video played at its own pace in place of the camera, to try the live mode without one
    #[arg(long, value_name = "VIDEO")]
    source: Option<PathBuf>,
    #[command(flatten)]
    shared: Shared,
}

#[derive(clap::Args)]
struct VideoArgs {
    /// The video of the solve
    video: PathBuf,
    /// The scramble, in WCA notation
    #[arg(long)]
    scramble: String,
    /// Colour on top when scrambling
    #[arg(long, default_value = "white")]
    top: String,
    /// Colour in front when scrambling
    #[arg(long, default_value = "green")]
    front: String,
    /// Colour of the cross (default: the one the solve shows)
    #[arg(long)]
    cross: Option<String>,
    /// Print the result as JSON, without opening a window
    #[arg(long)]
    json: bool,
    /// Print the result as text, without opening a window
    #[arg(long)]
    no_window: bool,
    /// Read frame by frame, the detector looking at one frame in so many once the cube is read (default: batches, every frame)
    #[arg(long, value_name = "FRAMES")]
    detect_every: Option<u32>,
    /// Also write the frames as they were read to this file (JSON), to study the decoding alone
    #[arg(long, value_name = "FILE")]
    dump: Option<PathBuf>,
    /// The ffmpeg to run
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: String,
    #[command(flatten)]
    shared: Shared,
}

/// The models' folder: the one named, else `models` next to the program
fn models_dir(named: &Option<PathBuf>) -> Result<PathBuf> {
    if let Some(dir) = named {
        return Ok(dir.clone());
    }
    let beside = std::env::current_exe().ok().and_then(|exe| exe.parent().map(|dir| dir.join("models")));
    beside.context("the models' folder is not known: name it with --models")
}

fn load(shared: &Shared) -> Result<Models> {
    let models = Models::load(&models_dir(&shared.models)?, shared.device)?;
    if let Some(why) = &models.fallback {
        eprintln!("note: {why}");
    }
    Ok(models)
}

fn video(args: &VideoArgs) -> Result<bool> {
    if args.scramble.split_whitespace().next().is_none() {
        bail!("the scramble is empty: the moves are found from the scrambled cube");
    }
    let hold = hold(colour(&args.top)?, colour(&args.front)?)?;
    let cross = args.cross.as_deref().map(colour).transpose()?;
    let mut source = VideoReader::open(&args.video, &VideoOptions { ffmpeg: args.ffmpeg.clone(), ..VideoOptions::default() })?;
    let info = source.info().clone();
    let mut models = load(&args.shared)?;
    let batches = args.detect_every.is_none();
    if batches {
        models.warm(BATCH)?;
    }
    let Models { detector, reader, device, .. } = models;
    let mut frames = FrameLoop::new(detector, reader, &args.scramble, Some(hold), args.detect_every.unwrap_or(1).max(1));

    let reading = Instant::now();
    let mut batch = Vec::with_capacity(BATCH);
    while let Some(image) = source.next_image()? {
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
        bail!("{} holds no frame", args.video.display());
    }

    if let Some(file) = &args.dump {
        std::fs::write(file, serde_json::to_string(&cubevision::json::frames(&read.frames))?)
            .with_context(|| format!("writing {}", file.display()))?;
    }

    let decoding = Instant::now();
    let options = AnalyzeOptions { hold: Some(hold.to_string()), cross, ..AnalyzeOptions::default() };
    let Some(result) = analyze_readings(&read.frames, &args.scramble, &options) else {
        bail!(
            "no face of a cube could be read in {} ({} frames, the cube found in {})",
            args.video.display(),
            read.frames.len(),
            read.found
        );
    };
    let decode_seconds = decoding.elapsed().as_secs_f64();

    // A scramble that is not the video's: the readings fit no cube that follows from it
    let mismatch = !result.solved && result.agreement < MISMATCH;
    let warning = if mismatch {
        Some(
            "the scramble does not seem to be this video's: the moves found explain too little of what was read, and are not shown as a solve",
        )
    } else if !result.solved {
        Some("the moves found do not end on a solved cube: some turns could not be followed (see gaps)")
    } else {
        None
    };
    if args.json {
        let mut out = json!({
            "video": { "file": args.video.display().to_string(), "width": info.width, "height": info.height, "fps": info.fps, "codec": info.codec, "mirrored": read.mirrored },
            "device": device.name(),
            "scramble": args.scramble,
            "cubeFound": read.found,
            "seconds": { "read": read_seconds, "decode": decode_seconds },
            "warning": warning,
        });
        let fields = out.as_object_mut().unwrap();
        for (key, value) in cubevision::json::result(&result).as_object().unwrap() {
            fields.insert(key.clone(), value.clone());
        }
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "{}: {}×{} at {:.2} fps{}",
            args.video.display(),
            info.width,
            info.height,
            info.fps,
            if read.mirrored { ", mirrored" } else { "" }
        );
        println!("read on {} in {read_seconds:.1} s, decoded in {decode_seconds:.1} s", device.name());
        println!("frames read: {} of {} (cube found in {})", result.read_frames, result.frames, read.found);
        if !mismatch {
            let r = &result.reconstruction;
            println!("solution ({} moves, {:.2} s, {:.2} turns a second):", r.moves, r.time / 1000.0, r.tps);
            for step in &r.splits {
                println!("  {:<6} {:>6.2} s  {}", format!("{:?}", step.phase).to_lowercase(), step.duration / 1000.0, step.alg);
            }
            println!("stickers read that fit the moves: {:.1} %", result.agreement * 100.0);
        }
    }
    if let Some(warning) = warning {
        eprintln!("warning: {warning}");
    }
    let solved = result.solved;
    if !args.json && !args.no_window && !mismatch {
        replay_window::run(replay_window::Shown {
            video: args.video.clone(),
            ffmpeg: args.ffmpeg.clone(),
            title: args.video.file_name().unwrap_or_default().to_string_lossy().into_owned(),
            scramble: args.scramble.clone(),
            device: device.name().to_owned(),
            mirrored: read.mirrored,
            warning: warning.map(str::to_owned),
            result,
            snapshot: Snapshot::new(args.shared.screenshot.clone(), args.shared.after),
        })?;
    }
    Ok(solved)
}

/// Below this share of the stickers read fitting the moves found, an unsolved result is taken for a scramble
/// that is not the video's. On synthetic solves read by the released models, a wrong scramble gives 0.27 to
/// 0.31, the right one on a solve the decoder could not finish 0.41 to 0.50.
const MISMATCH: f64 = 0.35;

fn live(args: &LiveArgs) -> Result<bool> {
    let source = match &args.source {
        Some(video) => Source::Video(video.clone()),
        None => Source::Camera(CameraRequest { index: args.camera, width: args.width, height: args.height, fps: args.fps }),
    };
    let snapshot = Snapshot::new(args.shared.screenshot.clone(), args.shared.after);
    cubevision::live::run(LiveOptions { source, models: models_dir(&args.shared.models)?, device: args.shared.device, snapshot })?;
    Ok(true)
}

fn run() -> Result<bool> {
    match Cli::parse().command {
        Some(Command::Video(args)) => video(&args),
        Some(Command::Live(args)) => live(&args),
        Some(Command::FetchModels(args)) => {
            cubevision::fetch::fetch(&models_dir(&args.models)?, &args.url)?;
            Ok(true)
        }
        Some(Command::Bench(args)) => {
            cubevision::tools::bench(&args.video, &args.scramble, &args.ffmpeg, load(&args.shared)?)?;
            Ok(true)
        }
        Some(Command::Eval(args)) => {
            let truth = args.truth.clone().unwrap_or_else(|| args.video.with_extension("truth.json"));
            cubevision::tools::eval(&args.video, &truth, &args.ffmpeg, load(&args.shared)?)
        }
        Some(Command::Check(args)) => {
            let models = cubevision::check::models(&models_dir(&args.shared.models)?, args.shared.device, args.seconds)?;
            if args.with_camera {
                cubevision::check::camera(
                    models,
                    &CameraRequest { index: args.camera, ..CameraRequest::default() },
                    args.seconds,
                    args.save_crop.as_deref(),
                )?;
            }
            Ok(true)
        }
        None => live(&LiveArgs { width: 1280, height: 720, fps: 30, ..LiveArgs::default() }),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        // Read, but not to a solved cube
        Ok(false) => ExitCode::from(2),
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
