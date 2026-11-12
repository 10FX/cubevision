//! The two models on ONNX Runtime. The tests that run them are ignored by default: they need CUBE_MODELS (the
//! folder of the released models) and ORT_DYLIB_PATH (the ONNX Runtime library):
//!   cargo test -p cube-infer --release --test models -- --include-ignored

use std::path::PathBuf;

use cube_infer::device::{AUTO, Device, Precision};
use cube_infer::follow::{Detector, Look};
use cube_infer::models::{BATCH, Models, metadata};
use cube_infer::output::Reader;
use cube_infer::{BoxF, Frame};

#[test]
fn every_device_is_named_as_it_is_asked_for() {
    for name in ["cpu", "cuda", "directml", "openvino-cpu", "openvino-gpu", "openvino-npu"] {
        assert_eq!(name.parse::<Device>().unwrap().name(), name);
    }
    let error = "openvino".parse::<Device>().unwrap_err().to_string();
    assert!(error.contains("openvino-gpu") && error.contains("cuda"), "the devices are listed: {error}");
    assert_eq!(AUTO.last(), Some(&Device::Cpu), "the processor is tried last");
}

#[test]
fn fp16_only_on_cuda_and_directml() {
    for device in [Device::Cpu, Device::OpenVinoCpu, Device::OpenVinoGpu, Device::OpenVinoNpu] {
        assert_eq!(device.precision(), Precision::Fp32);
    }
    for device in [Device::Cuda, Device::DirectMl] {
        assert_eq!(device.precision(), Precision::Fp16);
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cube-infer-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const DETECTOR: &str = r#"{"name":"cube-detector","input":{"name":"image","size":320,"layout":"NCHW","range":[0,1],"letterbox":0.5},
    "threshold":0.5,"cropMargin":1.5,"variants":{"full":{"file":"cube-detector.fp16.onnx"},"lite":{"file":"cube-detector.fp32.onnx"}}}"#;
const READER: &str = r#"{"name":"cube-reader","input":{"name":"image"},"cells":"cube",
    "variants":{"full":{"file":"cube-reader.fp16.onnx","size":224},"lite":{"file":"cube-reader.fp32.onnx","size":224}}}"#;

#[test]
fn metadata_is_read() {
    let dir = scratch("meta");
    std::fs::write(dir.join("cube-detector.json"), DETECTOR).unwrap();
    std::fs::write(dir.join("cube-reader.json"), READER).unwrap();
    let meta = metadata(&dir).unwrap();
    assert_eq!(
        (meta.detector.input.as_str(), meta.detector.size, meta.detector.letterbox, meta.detector.threshold, meta.detector.crop_margin),
        ("image", 320, 0.5, 0.5, 1.5)
    );
    assert_eq!(meta.detector.file(Precision::Fp16), "cube-detector.fp16.onnx");
    assert_eq!(meta.detector.file(Precision::Fp32), "cube-detector.fp32.onnx");
    assert_eq!((meta.reader.input.as_str(), meta.reader.ordered), ("image", true));
    assert_eq!(meta.reader.variant(Precision::Fp16), ("cube-reader.fp16.onnx", 224));
    assert_eq!(meta.reader.variant(Precision::Fp32), ("cube-reader.fp32.onnx", 224));
    // A reader that does not say its cells come in the cube's order
    std::fs::write(dir.join("cube-reader.json"), READER.replace(r#""cells":"cube","#, "")).unwrap();
    assert!(!metadata(&dir).unwrap().reader.ordered);
}

#[test]
fn missing_models_point_to_fetch_models() {
    let dir = scratch("empty");
    let error = format!("{:#}", metadata(&dir).unwrap_err());
    assert!(error.contains("cubevision fetch-models") && error.contains(dir.to_str().unwrap()), "{error}");
    let error = format!("{:#}", Models::load(&dir, Some(Device::Cpu)).err().expect("no models, no reading"));
    assert!(error.contains("cubevision fetch-models"), "{error}");
    // The description without the weights
    std::fs::write(dir.join("cube-detector.json"), DETECTOR).unwrap();
    std::fs::write(dir.join("cube-reader.json"), READER).unwrap();
    let error = format!("{:#}", Models::load(&dir, Some(Device::Cpu)).err().expect("no weights, no reading"));
    assert!(error.contains("cube-detector.fp32.onnx") && error.contains("cubevision fetch-models"), "{error}");
}

#[test]
fn half_the_weights_missing_is_not_a_reason_to_read_slower() {
    // Without --device: the fp16 files a GPU takes are missing. Not "no GPU here, the processor then"
    let dir = scratch("half");
    std::fs::write(dir.join("cube-detector.json"), DETECTOR).unwrap();
    std::fs::write(dir.join("cube-reader.json"), READER).unwrap();
    std::fs::write(dir.join("cube-detector.fp32.onnx"), "").unwrap();
    std::fs::write(dir.join("cube-reader.fp32.onnx"), "").unwrap();
    let error = format!("{:#}", Models::load(&dir, None).err().expect("half the models"));
    assert!(error.contains("fp16.onnx") && error.contains("cubevision fetch-models"), "{error}");
}

#[test]
fn a_description_that_is_not_one_is_explained() {
    let dir = scratch("broken");
    std::fs::write(dir.join("cube-detector.json"), "{").unwrap();
    std::fs::write(dir.join("cube-reader.json"), READER).unwrap();
    assert!(format!("{:#}", metadata(&dir).unwrap_err()).contains("cube-detector.json"));
    std::fs::write(dir.join("cube-detector.json"), DETECTOR.replace(r#""size":320,"#, "")).unwrap();
    let error = format!("{:#}", metadata(&dir).unwrap_err());
    assert!(error.contains("cube-detector.json") && error.contains("size"), "{error}");
    // A size no model has would only fail later, in a run
    std::fs::write(dir.join("cube-detector.json"), DETECTOR.replace(r#""size":320,"#, r#""size":0,"#)).unwrap();
    assert!(metadata(&dir).is_err());
}

// ---- with the models ----

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

fn worst(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

/// The outputs of ONNX Runtime 1.31 in Python on the same input (spikes/capture-infer/tools/make_reference.py)
fn against_python(device: Device, tolerance: f32) {
    let reference: serde_json::Value = serde_json::from_str(include_str!("reference.json")).unwrap();
    let want = |name: &str| -> Vec<f32> { reference[name].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect() };
    let mut models = Models::load(&models_dir(), Some(device)).unwrap();
    assert_eq!((models.device, &models.fallback), (device, &None));
    let (boxed, score) = models.detector.run(&fixed_input(320), 1).unwrap().remove(0);
    let out = models.reader.run(&fixed_input(224), 1).unwrap().remove(0);
    for (name, got, want) in [
        ("box", &boxed[..], want("box")),
        ("score", &[score][..], want("score")),
        ("present", &out.present[..], want("present")),
        ("stickers", &out.stickers[..], want("stickers")),
        ("turning", &[out.turning][..], want("turning")),
        ("orientation", &out.orientation.expect("the reader gives the orientation")[..], want("orientation")),
    ] {
        let gap = worst(got, &want);
        assert!(gap <= tolerance, "{name}: max gap {gap} > {tolerance}");
    }
}

#[test]
#[ignore = "needs CUBE_MODELS and ORT_DYLIB_PATH"]
fn cpu_matches_python() {
    against_python(Device::Cpu, 1e-4);
}

#[test]
#[ignore = "needs CUBE_MODELS, ORT_DYLIB_PATH and a CUDA runtime"]
fn cuda_is_close_to_python() {
    against_python(Device::Cuda, 1e-2);
}

/// A crop with something in it: a gradient and a few coloured squares
fn picture(size: usize, seed: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(size * size * 4);
    for y in 0..size {
        for x in 0..size {
            let square = ((x / 37 + y / 41 + seed) % 6) as u8;
            data.extend([(x * 255 / size) as u8 ^ (square * 40), (y * 255 / size) as u8, square * 50, 255]);
        }
    }
    data
}

#[test]
#[ignore = "needs CUBE_MODELS and ORT_DYLIB_PATH"]
fn a_batch_reads_as_one_by_one() {
    let mut models = Models::load(&models_dir(), Some(Device::Cpu)).unwrap();
    let size = models.reader.size();
    assert_eq!((size, models.reader.ordered(), BATCH), (224, true, 8));
    // Nine crops: one full batch and one padded
    let crops: Vec<Vec<u8>> = (0..9).map(|seed| picture(size, seed)).collect();
    let views: Vec<&[u8]> = crops.iter().map(Vec::as_slice).collect();
    let together = models.reader.read(&views).unwrap();
    assert_eq!(together.len(), 9);
    for (i, crop) in views.iter().enumerate() {
        let alone = models.reader.read(&[crop]).unwrap().remove(0);
        assert!(worst(&alone.stickers, &together[i].stickers) <= 1e-4 && worst(&alone.present, &together[i].present) <= 1e-4, "crop {i}");
        assert!((alone.turning - together[i].turning).abs() <= 1e-4, "crop {i}: turning");
    }
    assert!(models.reader.read(&[]).unwrap().is_empty());
    // A crop of another size is refused, not read askew
    let error = models.reader.read(&[&crops[0][..100]]).unwrap_err().to_string();
    assert!(error.contains("224"), "{error}");

    // The detector too: nine looks at once as nine looks one by one
    let frames: Vec<Vec<u8>> = (0..9).map(|seed| picture(160, seed)).collect();
    let frames: Vec<Frame> = frames.iter().map(|data| Frame { data, width: 160, height: 160 }).collect();
    let looks: Vec<Look> = frames
        .iter()
        .enumerate()
        .map(|(i, frame)| Look { frame, region: (i % 2 == 1).then_some(BoxF { x: 20.0, y: 10.0, w: 90.0, h: 100.0 }) })
        .collect();
    let together = models.detector.look(&looks).unwrap();
    assert_eq!(together.len(), 9);
    for (i, look) in looks.iter().enumerate() {
        let alone = models.detector.look(&[*look]).unwrap().remove(0);
        assert_eq!(alone.is_some(), together[i].is_some(), "look {i}");
    }
}

#[test]
#[ignore = "needs CUBE_MODELS and ORT_DYLIB_PATH"]
fn a_black_frame_shows_no_cube() {
    let mut models = Models::load(&models_dir(), Some(Device::Cpu)).unwrap();
    let data = vec![0u8; 640 * 480 * 4];
    let frame = Frame { data: &data, width: 640, height: 480 };
    assert_eq!(models.detector.look(&[Look { frame: &frame, region: None }]).unwrap(), [None]);
    assert_eq!((models.detector.size(), models.detector.crop_margin()), (320, 1.5));
    assert!(models.detector.look(&[]).unwrap().is_empty());
}

#[test]
#[ignore = "needs CUBE_MODELS and ORT_DYLIB_PATH"]
fn an_unavailable_device_is_an_error() {
    // DirectML is Windows'; OpenVINO's NPU needs its own ONNX Runtime: asked for by name, never replaced in silence
    let device = if cfg!(windows) { Device::OpenVinoNpu } else { Device::DirectMl };
    let error = format!("{:#}", Models::load(&models_dir(), Some(device)).err().expect("not available here"));
    assert!(error.contains(device.name()) && error.contains("--device"), "{error}");
}

#[test]
#[ignore = "needs CUBE_MODELS and ORT_DYLIB_PATH"]
fn auto_picks_a_device() {
    let models = Models::load(&models_dir(), None).unwrap();
    assert!(AUTO.contains(&models.device));
    // On the processor only after the others failed, and it says so
    assert_eq!(models.fallback.is_some(), models.device == Device::Cpu);
}
