//! ONNX Runtime not found. Alone in its file: the library is looked for once per process, where the
//! environment says.

use cube_infer::device::Device;
use cube_infer::models::Models;

#[test]
fn a_missing_onnx_runtime_says_how_to_name_it() {
    let dir = std::env::temp_dir().join(format!("cube-infer-{}-library", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("cube-detector.json"),
        r#"{"input":{"name":"image","size":320,"letterbox":0.5},"threshold":0.5,"cropMargin":1.5,
            "variants":{"full":{"file":"d16.onnx"},"lite":{"file":"d32.onnx"}}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("cube-reader.json"),
        r#"{"input":{"name":"image"},"variants":{"full":{"file":"r16.onnx","size":224},"lite":{"file":"r32.onnx","size":224}}}"#,
    )
    .unwrap();
    for file in ["d16.onnx", "d32.onnx", "r16.onnx", "r32.onnx"] {
        std::fs::write(dir.join(file), "").unwrap();
    }
    // SAFETY: the only test of this process, nothing else reads the environment meanwhile
    unsafe { std::env::set_var("ORT_DYLIB_PATH", dir.join("no-such-library.so")) };
    for device in [Some(Device::Cpu), Some(Device::Cuda), None] {
        // The first line is all a program prints of an error: it must say what to do
        let error = Models::load(&dir, device).err().expect("no ONNX Runtime").to_string();
        assert!(error.contains("ORT_DYLIB_PATH") && error.contains("no-such-library.so"), "{device:?}: {error}");
        assert!(!error.contains("--device"), "{device:?}: another device would not help: {error}");
    }
}
