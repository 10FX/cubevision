use std::path::PathBuf;

use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use spike::bench;
use spike::models::{Device, Models, Precision};
use spike::pipeline::{BATCH, FrameTimes, process_batch};
use spike::video::VideoReader;

#[derive(Parser)]
#[command(about = "Throwaway measurements: models, video and webcam from Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Detector + reader speed on a fixed input
    Infer {
        /// The models' folder
        #[arg(long, env = "CUBE_MODELS")]
        models: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: Device,
        /// fp16 on CUDA and DirectML, fp32 elsewhere, unless said
        #[arg(long)]
        precision: Option<Precision>,
        #[arg(long, default_value_t = 20.0)]
        seconds: f64,
    },
    /// A video file read end to end: decode, detector, crop, reader
    Video {
        file: PathBuf,
        #[arg(long, env = "CUBE_MODELS")]
        models: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: Device,
        #[arg(long)]
        precision: Option<Precision>,
        #[arg(long, default_value = "ffmpeg")]
        ffmpeg: String,
    },
    /// What the webcam delivers for some seconds; with --device, the models read it live
    Cam {
        #[arg(long, default_value_t = 0)]
        index: u32,
        #[arg(long, default_value_t = 1280)]
        width: u32,
        #[arg(long, default_value_t = 720)]
        height: u32,
        #[arg(long, default_value_t = 30)]
        fps: u32,
        #[arg(long, default_value_t = 30.0)]
        seconds: f64,
        /// Without a device, the camera alone is measured
        #[arg(long)]
        device: Option<Device>,
        #[arg(long, env = "CUBE_MODELS")]
        models: Option<PathBuf>,
        #[arg(long)]
        precision: Option<Precision>,
    },
}

/// Decoding runs in its own thread, a few batches ahead of the models
fn video(file: PathBuf, mut models: Models, ffmpeg: String) -> Result<serde_json::Value> {
    let mut reader = VideoReader::open(&file, &ffmpeg)?;
    let ((width, height), file_fps, codec) = (reader.size(), reader.fps(), reader.codec().to_string());
    let (send, frames) = std::sync::mpsc::sync_channel(2 * BATCH);
    let decoder = std::thread::spawn(move || -> Result<f64> {
        let mut decode_ms = 0.0;
        loop {
            let t = Instant::now();
            let frame = reader.next_frame()?;
            decode_ms += t.elapsed().as_secs_f64() * 1000.0;
            match frame {
                Some(frame) => if send.send(frame).is_err() { break },
                None => break,
            }
        }
        Ok(decode_ms)
    });
    let start = Instant::now();
    let (mut sum, mut count, mut found, mut last) = (FrameTimes::default(), 0usize, 0usize, None);
    let mut frames = frames.into_iter().peekable();
    while frames.peek().is_some() {
        let batch: Vec<_> = frames.by_ref().take(BATCH).collect();
        for t in process_batch(&mut models, &batch, &mut last)? {
            sum.letterbox_ms += t.letterbox_ms;
            sum.detector_ms += t.detector_ms;
            sum.crop_ms += t.crop_ms;
            sum.reader_ms += t.reader_ms;
            count += 1;
            found += t.found as usize;
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let decode_ms = decoder.join().map_err(|_| anyhow::anyhow!("the decoder thread panicked"))??;
    let n = count.max(1) as f64;
    Ok(serde_json::json!({
        "file": file.file_name().map(|n| n.to_string_lossy().into_owned()),
        "codec": codec, "width": width, "height": height, "file_fps": file_fps,
        "device": bench::device_name(models.device), "precision": bench::precision_name(models.precision),
        "frames": count,
        "processed_fps": count as f64 / wall,
        "found_share": found as f64 / n,
        "ms_mean_per_frame": {
            "decode": decode_ms / n, "letterbox": sum.letterbox_ms / n, "detector": sum.detector_ms / n,
            "crop": sum.crop_ms / n, "reader": sum.reader_ms / n,
        },
    }))
}

pub fn default_precision(device: Device) -> Precision {
    device.default_precision()
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Infer { models, device, precision, seconds } => {
            let precision = precision.unwrap_or(default_precision(device));
            let mut loaded = Models::load(&models, device, precision)?;
            let timings = bench::run(&mut loaded, seconds)?;
            println!("{}", serde_json::to_string_pretty(&timings.summary(device, precision))?);
        }
        Command::Video { file, models, device, precision, ffmpeg } => {
            let loaded = Models::load(&models, device, precision.unwrap_or(default_precision(device)))?;
            println!("{}", serde_json::to_string_pretty(&video(file, loaded, ffmpeg)?)?);
        }
        #[cfg(windows)]
        Command::Cam { index, width, height, fps, seconds, device, models, precision } => {
            let mut loaded = match device {
                Some(device) => {
                    let models = models.ok_or_else(|| anyhow::anyhow!("--device needs --models or CUBE_MODELS"))?;
                    Some(Models::load(&models, device, precision.unwrap_or(default_precision(device)))?)
                }
                None => None,
            };
            let request = spike::cam::Request { index, width, height, fps, seconds };
            println!("{}", serde_json::to_string_pretty(&spike::cam::run(&request, loaded.as_mut())?)?);
        }
        #[cfg(not(windows))]
        Command::Cam { .. } => anyhow::bail!("the camera is measured on Windows: WSL sees none"),
    }
    Ok(())
}
