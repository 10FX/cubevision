//! The Rust side gives the same outputs as ONNX Runtime from Python on the same input.
//! CUBE_MODELS names the models' folder; ORT_DYLIB_PATH the ONNX Runtime library.

use std::path::PathBuf;

use spike::models::{Device, Models, Precision};

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("CUBE_MODELS").expect("CUBE_MODELS names the models' folder"))
}

fn fixed_input(size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(3 * size * size);
    for c in 0..3 {
        for y in 0..size {
            for i in 0..size {
                out.push(((i + y + c) % size) as f32 / (size - 1) as f32);
            }
        }
    }
    out
}

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!("reference.json")).unwrap()
}

fn assert_close(name: &str, got: &[f32], reference: &serde_json::Value, tolerance: f32) {
    let want: Vec<f32> = reference[name].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    assert_eq!(got.len(), want.len(), "{name}: length");
    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(worst <= tolerance, "{name}: max gap {worst} > {tolerance}");
}

fn check(device: Device, precision: Precision, tolerance: f32) {
    let reference = reference();
    let mut models = Models::load(&models_dir(), device, precision).unwrap();
    let detection = models.detect(&fixed_input(320)).unwrap();
    assert_close("box", &detection.boxed, &reference, tolerance);
    assert_close("score", &[detection.score], &reference, tolerance);
    let reading = models.read(&fixed_input(224), 1).unwrap().remove(0);
    assert_close("present", &reading.present, &reference, tolerance);
    assert_close("stickers", &reading.stickers, &reference, tolerance);
    assert_close("turning", &[reading.turning], &reference, tolerance);
    assert_close("orientation", &reading.orientation, &reference, tolerance);
}

#[test]
fn cpu_fp32_matches_python() {
    check(Device::Cpu, Precision::Fp32, 1e-4);
}

#[test]
#[ignore = "needs a CUDA runtime"]
fn cuda_fp16_is_close_to_python() {
    check(Device::Cuda, Precision::Fp16, 1e-2);
}

#[test]
#[ignore = "needs ONNX Runtime for OpenVINO"]
fn openvino_cpu_matches_python() {
    check(Device::OpenVinoCpu, Precision::Fp32, 1e-3);
}

#[test]
#[ignore = "needs a CPU-only runtime"]
fn cuda_refused_without_cuda() {
    let error = Models::load(&models_dir(), Device::Cuda, Precision::Fp16).err().expect("CUDA must be refused, not replaced by the CPU");
    assert!(error.to_string().to_lowercase().contains("cuda"), "{error}");
}

#[test]
fn a_batch_gives_each_square_its_own_outputs() {
    let reference = reference();
    let mut models = Models::load(&models_dir(), Device::Cpu, Precision::Fp32).unwrap();
    let detections = models.detect_batch(&fixed_input(320).repeat(3), 3).unwrap();
    assert_eq!(detections.len(), 3);
    for detection in &detections {
        assert_close("box", &detection.boxed, &reference, 1e-4);
        assert_close("score", &[detection.score], &reference, 1e-4);
    }
}
