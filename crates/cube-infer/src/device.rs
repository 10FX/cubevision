//! Where the models run: ONNX Runtime's execution provider, asked for by name or tried in order

use std::str::FromStr;

use anyhow::{Result, bail};

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

/// The devices tried, in this order, when none is asked for: the first that runs the models is kept. The
/// ONNX Runtime library loaded offers one GPU engine at most, so this finds it; the processor comes last.
pub const AUTO: [Device; 4] = [Device::Cuda, Device::DirectMl, Device::OpenVinoGpu, Device::Cpu];

impl Device {
    pub fn name(self) -> &'static str {
        DEVICES.iter().find(|(d, _)| *d == self).map(|(_, n)| *n).unwrap()
    }

    /// Every name `--device` takes
    pub fn names() -> String {
        DEVICES.map(|(_, n)| n).join(", ")
    }

    /// fp16 models for CUDA and DirectML; fp32 elsewhere: OpenVINO runs a GPU or an NPU in fp16 by itself from
    /// the fp32 model (its GPU plugin does not compile every fp16 conversion of the reader). The same network.
    pub fn precision(self) -> Precision {
        if matches!(self, Device::Cuda | Device::DirectMl) { Precision::Fp16 } else { Precision::Fp32 }
    }
}

impl FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match DEVICES.iter().find(|(_, n)| *n == s) {
            Some((device, _)) => Ok(*device),
            None => bail!("unknown device {s:?}: {}", Device::names()),
        }
    }
}

/// The weights' precision: the models' "full" variant is fp16, their "lite" one fp32
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp16,
    Fp32,
}
