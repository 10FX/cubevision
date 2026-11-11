//! How long each model takes per frame, after warming up

use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::{Value, json};

use crate::models::{Device, Models, Precision};
use crate::stats::percentile;

pub fn device_name(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda => "cuda",
        Device::DirectMl => "directml",
    }
}

pub fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::Fp16 => "fp16",
        Precision::Fp32 => "fp32",
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// Milliseconds per run
#[derive(Default)]
pub struct Timings {
    pub detector: Vec<f64>,
    pub reader: Vec<f64>,
    /// one batch of 8 squares per value
    pub reader_batch8: Vec<f64>,
    pub detector_batch8: Vec<f64>,
}

impl Timings {
    pub fn summary(&self, device: Device, precision: Precision) -> Value {
        let quantiles = |values: &[f64]| {
            let mut v = values.to_vec();
            json!({ "p50": percentile(&mut v, 50.0), "p95": percentile(&mut v, 95.0) })
        };
        let mut totals: Vec<f64> = self.detector.iter().zip(&self.reader).map(|(d, r)| d + r).collect();
        let mut batch = self.reader_batch8.clone();
        let mut detector_batch = self.detector_batch8.clone();
        json!({
            "device": device_name(device),
            "precision": precision_name(precision),
            "frames": self.detector.len(),
            "detector_ms": quantiles(&self.detector),
            "reader_ms": quantiles(&self.reader),
            "total_p95_ms": percentile(&mut totals, 95.0),
            "reader_batch8_ms_per_image": percentile(&mut batch, 50.0) / 8.0,
            "detector_batch8_ms_per_image": percentile(&mut detector_batch, 50.0) / 8.0,
        })
    }
}

/// A fixed square of `size`, the same as the parity test's
pub fn fixed_input(size: usize, batch: usize) -> Vec<f32> {
    let mut one = Vec::with_capacity(3 * size * size);
    for c in 0..3 {
        for y in 0..size {
            for i in 0..size {
                one.push(((i + y + c) % size) as f32 / (size - 1) as f32);
            }
        }
    }
    one.repeat(batch)
}

/// 20 warm-up frames, then detector + reader frame after frame for `seconds`, then 8-square batches
pub fn run(models: &mut Models, seconds: f64) -> Result<Timings> {
    let (square, crop, crops, squares) = (fixed_input(320, 1), fixed_input(224, 1), fixed_input(224, 8), fixed_input(320, 8));
    for _ in 0..20 {
        models.detect(&square)?;
        models.read(&crop, 1)?;
        models.read(&crops, 8)?;
        models.detect_batch(&squares, 8)?;
    }
    let mut timings = Timings::default();
    let end = Instant::now() + Duration::from_secs_f64(seconds);
    while Instant::now() < end {
        let t = Instant::now();
        models.detect(&square)?;
        timings.detector.push(ms(t));
        let t = Instant::now();
        models.read(&crop, 1)?;
        timings.reader.push(ms(t));
    }
    for _ in 0..50 {
        let t = Instant::now();
        models.read(&crops, 8)?;
        timings.reader_batch8.push(ms(t));
        let t = Instant::now();
        models.detect_batch(&squares, 8)?;
        timings.detector_batch8.push(ms(t));
    }
    Ok(timings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_adds_both_models_per_frame() {
        let timings = Timings { detector: vec![1.0, 2.0, 3.0, 4.0], reader: vec![10.0, 10.0, 10.0, 30.0], reader_batch8: vec![16.0, 24.0], detector_batch8: vec![40.0] };
        let s = timings.summary(Device::Cuda, Precision::Fp16);
        assert_eq!(s["device"], "cuda");
        assert_eq!(s["precision"], "fp16");
        assert_eq!(s["detector_ms"]["p50"], 2.0);
        assert_eq!(s["reader_ms"]["p95"], 30.0);
        assert_eq!(s["total_p95_ms"], 34.0);
        assert_eq!(s["reader_batch8_ms_per_image"], 2.0);
        assert_eq!(s["detector_batch8_ms_per_image"], 5.0);
    }
}
