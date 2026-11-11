use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use spike::bench;
use spike::models::{Device, Models, Precision};

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
        /// fp16 on a GPU and fp32 on the CPU unless said
        #[arg(long)]
        precision: Option<Precision>,
        #[arg(long, default_value_t = 20.0)]
        seconds: f64,
    },
}

pub fn default_precision(device: Device) -> Precision {
    if device == Device::Cpu { Precision::Fp32 } else { Precision::Fp16 }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Infer { models, device, precision, seconds } => {
            let precision = precision.unwrap_or(default_precision(device));
            let mut loaded = Models::load(&models, device, precision)?;
            let timings = bench::run(&mut loaded, seconds)?;
            println!("{}", serde_json::to_string_pretty(&timings.summary(device, precision))?);
        }
    }
    Ok(())
}
