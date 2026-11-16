//! cubevision: a Rubik's Cube read from a video or a camera

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cube_capture::CameraRequest;
use cube_infer::device::Device;
use cube_infer::models::Models;
use cubevision::app::{self, First, Start};
use cubevision::colours::{colour, hold};
use cubevision::live::{LiveOptions, Source};
use cubevision::settings::{self, Engine};
use cubevision::solve::{Progress, Request, solve};
use cubevision::window::Snapshot;
use serde_json::json;

#[derive(Parser)]
#[command(name = "cubevision", version, about = "Reads a Rubik's Cube from a camera or a video")]
struct Cli {
    /// Without a command: the window, on its first screen
    #[command(subcommand)]
    command: Option<Command>,
    /// The folder of the models (default: `models` next to the program)
    #[arg(long, env = "CUBE_MODELS")]
    models: Option<PathBuf>,
    #[arg(long, value_name = "PNG", hide = true)]
    screenshot: Option<PathBuf>,
    #[arg(long, default_value_t = 4.0, hide = true)]
    after: f64,
}

#[derive(Subcommand)]
enum Command {
    /// The moves turned in front of the camera, as they are made, on a 3D cube
    Live(LiveArgs),
    /// The moves of a solve filmed, from its video and its scramble, played back on a 3D cube
    Video(VideoArgs),
    /// Asks for moves from a solved cube and saves the camera's crops of it, labelled with the cube's state
    Label(LabelArgs),
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
struct LabelArgs {
    /// The session's folder (default: cubevision-labels/session-<time> in the current folder)
    #[arg(long)]
    out: Option<PathBuf>,
    /// Where and how it is filmed (the room, the light, the cube), kept with the images
    #[arg(long, default_value = "")]
    note: String,
    /// Moves that scramble in a sequence (as many undo them)
    #[arg(long, default_value_t = 20)]
    length: usize,
    /// The moves asked for come from it (default: the time)
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, hide = true)]
    auto: Option<f64>,
    #[command(flatten)]
    camera: LiveArgs,
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
    /// The scramble the cube starts from, held white on top and green in front (default: solved)
    #[arg(long, value_name = "SCRAMBLE", hide = true)]
    from: Option<String>,
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

/// The ONNX Runtime of the program's own `runtime` folder that runs this device, unless one is named already
/// (`ORT_DYLIB_PATH`): a command needs no script to set it
fn point_at_runtime(device: Option<Device>) {
    if std::env::var_os("ORT_DYLIB_PATH").is_some_and(|path| !path.is_empty()) {
        return;
    }
    let engine = match device {
        None => Engine::Auto,
        Some(Device::DirectMl) => Engine::DirectMl,
        Some(Device::Cuda) => Engine::Cuda,
        Some(Device::Cpu) => Engine::Cpu,
        Some(Device::OpenVinoGpu | Device::OpenVinoCpu | Device::OpenVinoNpu) => Engine::IntelGpu,
    };
    if let Some(beside) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(|dir| dir.to_path_buf())) {
        settings::point_at(engine, &beside);
    }
}

fn load(shared: &Shared) -> Result<Models> {
    point_at_runtime(shared.device);
    let models = Models::load(&models_dir(&shared.models)?, shared.device)?;
    if let Some(why) = &models.fallback {
        eprintln!("note: {why}");
    }
    Ok(models)
}

fn video(args: &VideoArgs) -> Result<bool> {
    cubevision::colours::scramble(&args.scramble)?;
    let request = Request {
        video: args.video.clone(),
        scramble: args.scramble.clone(),
        hold: hold(colour(&args.top)?, colour(&args.front)?)?,
        cross: args.cross.as_deref().map(colour).transpose()?,
        ffmpeg: args.ffmpeg.clone(),
        detect_every: args.detect_every,
        dump: args.dump.clone(),
    };
    point_at_runtime(args.shared.device);
    // A long video takes a while: how far the reading is, every second, on the error stream
    let mut told = Instant::now();
    let mut progress = |now: Progress| {
        match now {
            Progress::Reading { read, all } if told.elapsed().as_secs_f64() >= 1.0 => {
                told = Instant::now();
                match all {
                    Some(all) => eprintln!("reading: {read} of {all} frames"),
                    None => eprintln!("reading: {read} frames"),
                }
            }
            Progress::Decoding { read, seconds } => eprintln!("read {read} frames in {seconds:.1} s; finding the moves…"),
            _ => {}
        }
        true
    };
    let found =
        solve(&request, &models_dir(&args.shared.models)?, args.shared.device, &mut progress)?.context("the reading was stopped")?;
    if let Some(why) = &found.note {
        eprintln!("note: {why}");
    }
    let (info, result) = (&found.info, &found.result);
    if args.json {
        let mut out = json!({
            "video": { "file": args.video.display().to_string(), "width": info.width, "height": info.height, "fps": info.fps, "codec": info.codec, "mirrored": found.mirrored },
            "device": found.device,
            "scramble": args.scramble,
            "cubeFound": found.found,
            "seconds": { "read": found.read_seconds, "decode": found.decode_seconds },
            "warning": found.warning,
        });
        let fields = out.as_object_mut().unwrap();
        for (key, value) in cubevision::json::result(result, found.mismatch).as_object().unwrap() {
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
            if found.mirrored { ", mirrored" } else { "" }
        );
        println!("read on {} in {:.1} s, decoded in {:.1} s", found.device, found.read_seconds, found.decode_seconds);
        println!("frames read: {} of {} (cube found in {})", result.read_frames, result.frames, found.found);
        if !found.mismatch {
            let r = &result.reconstruction;
            println!("solution ({} moves, {:.2} s, {:.2} turns a second):", r.moves, r.time / 1000.0, r.tps);
            for step in &r.splits {
                println!("  {:<6} {:>6.2} s  {}", format!("{:?}", step.phase).to_lowercase(), step.duration / 1000.0, step.alg);
            }
            println!("stickers read that fit the moves: {:.1} %", result.agreement * 100.0);
        }
    }
    if let Some(warning) = found.warning {
        eprintln!("warning: {warning}");
    }
    let solved = result.solved;
    if !args.json && !args.no_window && !found.mismatch {
        let snapshot = Snapshot::new(args.shared.screenshot.clone(), args.shared.after);
        let shown = app::shown(&request, found, snapshot);
        app::run(Start {
            models: models_dir(&args.shared.models)?,
            first: Some(First::Replay(Box::new(shown))),
            snapshot: Snapshot::default(),
        })?;
    }
    Ok(solved)
}

fn live(args: &LiveArgs) -> Result<bool> {
    let source = match &args.source {
        Some(video) => Source::Video(video.clone()),
        None => Source::Camera(CameraRequest { index: args.camera, width: args.width, height: args.height, fps: args.fps }),
    };
    point_at_runtime(args.shared.device);
    let models = models_dir(&args.shared.models)?;
    let options = LiveOptions {
        source,
        models: models.clone(),
        device: args.shared.device,
        mirror: Arc::new(AtomicBool::new(true)),
        start: args.from.clone(),
        snapshot: Snapshot::new(args.shared.screenshot.clone(), args.shared.after),
    };
    app::run(Start { models, first: Some(First::Live(options)), snapshot: Snapshot::default() })?;
    Ok(true)
}

fn label(args: &LabelArgs) -> Result<bool> {
    let live = &args.camera;
    let source = match &live.source {
        Some(video) => Source::Video(video.clone()),
        None => Source::Camera(CameraRequest { index: live.camera, width: live.width, height: live.height, fps: live.fps }),
    };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let out = args.out.clone().unwrap_or_else(|| PathBuf::from("cubevision-labels").join(format!("session-{now}")));
    point_at_runtime(live.shared.device);
    let models = models_dir(&live.shared.models)?;
    let options = cubevision::label_window::LabelOptions {
        source,
        models: models.clone(),
        device: live.shared.device,
        out,
        seed: args.seed.unwrap_or(now),
        length: args.length,
        note: args.note.clone(),
        auto: args.auto,
        snapshot: Snapshot::new(live.shared.screenshot.clone(), live.shared.after),
    };
    app::run(Start { models, first: Some(First::Label(options)), snapshot: Snapshot::default() })?;
    Ok(true)
}

/// Started by a double click, the program owns a console of its own, empty: it is let go (a command typed in a
/// terminal shares the terminal's, and keeps it)
#[cfg(windows)]
fn leave_own_console() {
    unsafe extern "system" {
        fn GetConsoleProcessList(list: *mut u32, count: u32) -> u32;
        fn FreeConsole() -> i32;
    }
    let mut list = [0u32; 2];
    // SAFETY: the list is as long as told; FreeConsole takes nothing
    unsafe {
        if GetConsoleProcessList(list.as_mut_ptr(), 2) == 1 {
            FreeConsole();
        }
    }
}

#[cfg(not(windows))]
fn leave_own_console() {}

fn run() -> Result<bool> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Video(args)) => video(&args),
        Some(Command::Live(args)) => live(&args),
        Some(Command::Label(args)) => label(&args),
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
            point_at_runtime(args.shared.device);
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
        None => {
            leave_own_console();
            app::run(Start { models: models_dir(&cli.models)?, first: None, snapshot: Snapshot::new(cli.screenshot, cli.after) })?;
            Ok(true)
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        // Read, but not to a solved cube (2 is the command line's own, for a usage error)
        Ok(false) => ExitCode::from(3),
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
