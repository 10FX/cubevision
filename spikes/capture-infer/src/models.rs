//! The detector and the reader, on the device asked for and nowhere else

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use ort::environment::Environment;
use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::Session;
use ort::value::TensorRef;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
    DirectMl,
}

impl FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Ok(Device::Cuda),
            "directml" => Ok(Device::DirectMl),
            other => bail!("unknown device {other:?}: cpu, cuda or directml"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp16,
    Fp32,
}

impl FromStr for Precision {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "fp16" => Ok(Precision::Fp16),
            "fp32" => Ok(Precision::Fp32),
            other => bail!("unknown precision {other:?}: fp16 or fp32"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Detection {
    /// x0 y0 x1 y1, shares of the square
    pub boxed: [f32; 4],
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct Reading {
    pub present: Vec<f32>,
    pub stickers: Vec<f32>,
    pub turning: f32,
    pub orientation: Vec<f32>,
}

pub struct Models {
    detector: Session,
    reader: Session,
    pub device: Device,
    pub precision: Precision,
}

/// ONNX Runtime allows one environment per process
fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let path = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from).context("ORT_DYLIB_PATH must name the ONNX Runtime library")?;
    let env = ort::init_from(&path)
        .map_err(|e| anyhow!("cannot load ONNX Runtime from {}: {e}", path.display()))?
        .with_name("spike")
        .build()
        .map_err(|e| anyhow!("{e}"))?;
    Ok(ENV.get_or_init(|| env))
}

fn provider(device: Device) -> Option<ExecutionProviderDispatch> {
    match device {
        Device::Cpu => None,
        Device::Cuda => Some(ep::CUDA::default().build().error_on_failure()),
        Device::DirectMl => Some(ep::DirectML::default().build().error_on_failure()),
    }
}

fn session(path: &Path, device: Device) -> Result<Session> {
    let mut builder = Session::builder(environment()?).map_err(|e| anyhow!("{e}"))?;
    if let Some(provider) = provider(device) {
        builder = builder
            .with_execution_providers([provider])
            .map_err(|e| anyhow!("{device:?} is not available here: {e}"))?
            // On a GPU the CPU threads mostly wait: one, not spinning, leaves the cores to decoding
            .with_intra_threads(1)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_op_spinning(false)
            .map_err(|e| anyhow!("{e}"))?;
    }
    builder.commit_from_file(path).map_err(|e| anyhow!("cannot open {}: {e}", path.display()))
}

impl Models {
    pub fn load(dir: &Path, device: Device, precision: Precision) -> Result<Models> {
        let suffix = match precision {
            Precision::Fp16 => "fp16",
            Precision::Fp32 => "fp32",
        };
        Ok(Models {
            detector: session(&dir.join(format!("cube-detector.{suffix}.onnx")), device)?,
            reader: session(&dir.join(format!("cube-reader.{suffix}.onnx")), device)?,
            device,
            precision,
        })
    }

    /// One letterboxed 320 square, CHW
    pub fn detect(&mut self, tensor: &[f32]) -> Result<Detection> {
        Ok(self.detect_batch(tensor, 1)?.remove(0))
    }

    /// `batch` letterboxed 320 squares, CHW one after the other
    pub fn detect_batch(&mut self, tensors: &[f32], batch: usize) -> Result<Vec<Detection>> {
        let input = TensorRef::from_array_view(([batch, 3, 320, 320], tensors))?;
        let outputs = self.detector.run(ort::inputs!["image" => input])?;
        let (_, boxed) = outputs["box"].try_extract_tensor::<f32>()?;
        let (_, score) = outputs["score"].try_extract_tensor::<f32>()?;
        Ok((0..batch).map(|i| Detection { boxed: [boxed[4 * i], boxed[4 * i + 1], boxed[4 * i + 2], boxed[4 * i + 3]], score: score[i] }).collect())
    }

    /// `batch` 224 squares, CHW one after the other
    pub fn read(&mut self, tensors: &[f32], batch: usize) -> Result<Vec<Reading>> {
        let input = TensorRef::from_array_view(([batch, 3, 224, 224], tensors))?;
        let outputs = self.reader.run(ort::inputs!["image" => input])?;
        let per = |name: &str| -> Result<Vec<Vec<f32>>> {
            let (_, data) = outputs[name].try_extract_tensor::<f32>()?;
            Ok(data.chunks(data.len() / batch).map(<[f32]>::to_vec).collect())
        };
        let (present, stickers, turning, orientation) = (per("present")?, per("stickers")?, per("turning")?, per("orientation")?);
        Ok((0..batch)
            .map(|i| Reading {
                present: present[i].clone(),
                stickers: stickers[i].clone(),
                turning: turning[i][0],
                orientation: orientation[i].clone(),
            })
            .collect())
    }
}
