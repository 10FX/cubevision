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
    /// Intel's OpenVINO on the processor, the integrated or discrete GPU, or the NPU
    OpenVinoCpu,
    OpenVinoGpu,
    OpenVinoNpu,
}

const DEVICES: [(Device, &str); 6] = [
    (Device::Cpu, "cpu"),
    (Device::Cuda, "cuda"),
    (Device::DirectMl, "directml"),
    (Device::OpenVinoCpu, "openvino-cpu"),
    (Device::OpenVinoGpu, "openvino-gpu"),
    (Device::OpenVinoNpu, "openvino-npu"),
];

impl Device {
    pub fn name(self) -> &'static str {
        DEVICES.iter().find(|(d, _)| *d == self).map(|(_, n)| *n).unwrap()
    }

    /// fp16 models for CUDA and DirectML; fp32 elsewhere: OpenVINO runs a GPU or an NPU in fp16 by itself from
    /// the fp32 model (its GPU plugin does not compile every fp16 conversion of the reader)
    pub fn default_precision(self) -> Precision {
        if matches!(self, Device::Cuda | Device::DirectMl) { Precision::Fp16 } else { Precision::Fp32 }
    }
}

impl FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match DEVICES.iter().find(|(_, n)| *n == s) {
            Some((device, _)) => Ok(*device),
            None => bail!("unknown device {s:?}: {}", DEVICES.map(|(_, n)| n).join(", ")),
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
        Device::OpenVinoCpu | Device::OpenVinoGpu | Device::OpenVinoNpu => {
            let target = device.name().trim_start_matches("openvino-").to_uppercase();
            // Compiled models are kept: a GPU or an NPU takes seconds to compile them at each start otherwise
            let cache = std::env::temp_dir().join("cubevision-openvino");
            let openvino = ep::OpenVINO::default().with_device_type(target).with_cache_dir(cache.display());
            // An NPU runs fixed shapes only: compiled again for each batch size met
            let openvino = if device == Device::OpenVinoNpu { openvino.with_dynamic_shapes(false) } else { openvino };
            Some(openvino.build().error_on_failure())
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_device_is_named_as_it_is_asked_for() {
        for name in ["cpu", "cuda", "directml", "openvino-cpu", "openvino-gpu", "openvino-npu"] {
            assert_eq!(name.parse::<Device>().unwrap().name(), name);
        }
        assert!("openvino".parse::<Device>().is_err(), "OpenVINO needs its device: cpu, gpu or npu");
    }

    #[test]
    fn fp16_models_only_where_onnx_runtime_runs_them_as_given() {
        for device in [Device::Cpu, Device::OpenVinoCpu, Device::OpenVinoGpu, Device::OpenVinoNpu] {
            assert_eq!(device.default_precision(), Precision::Fp32);
        }
        for device in [Device::Cuda, Device::DirectMl] {
            assert_eq!(device.default_precision(), Precision::Fp16);
        }
    }
}
