//! The models' zip unpacked and checked, without the network

use std::io::Write;
use std::path::PathBuf;

use cubevision::fetch::{unpack, verify};
use sha2::{Digest, Sha256};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cubevision-fetch-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn sha(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

fn zipped(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, data) in files {
        writer.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
        writer.write_all(data).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn descriptions(detector: &[u8], reader: &[u8]) -> (String, String) {
    (
        format!(
            r#"{{"variants":{{"full":{{"file":"cube-detector.fp16.onnx","sha256":"{}"}},"lite":{{"file":"cube-detector.fp32.onnx"}}}}}}"#,
            sha(detector)
        ),
        format!(r#"{{"variants":{{"full":{{"file":"cube-reader.fp16.onnx","size":224,"sha256":"{}"}}}}}}"#, sha(reader).to_uppercase()),
    )
}

#[test]
fn a_models_zip_is_unpacked_flat_and_checked() {
    let dir = scratch("good");
    let (d, r) = descriptions(b"detector weights", b"reader weights");
    let zip = zipped(&[
        ("cubevision-models-v7/cube-detector.json", d.as_bytes()),
        ("cubevision-models-v7/cube-reader.json", r.as_bytes()),
        ("cubevision-models-v7/cube-detector.fp16.onnx", b"detector weights"),
        ("cubevision-models-v7/cube-reader.fp16.onnx", b"reader weights"),
        ("cubevision-models-v7/LICENSE.txt", b"Apache-2.0"),
        // Not models: left out, wherever they would land
        ("cubevision-models-v7/run.bat", b"echo"),
        ("../../evil.onnx.exe", b"no"),
        ("cubevision-models-v7/.hidden.json", b"{}"),
    ]);
    let written = unpack(&zip, &dir).unwrap();
    assert_eq!(written, ["LICENSE.txt", "cube-detector.fp16.onnx", "cube-detector.json", "cube-reader.fp16.onnx", "cube-reader.json"]);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 5);
    assert_eq!(verify(&dir).unwrap(), ["cube-detector.fp16.onnx", "cube-reader.fp16.onnx"]);
}

#[test]
fn a_zip_cannot_write_outside_the_folder() {
    let dir = scratch("inside").join("models");
    let zip = zipped(&[("../escaped.onnx", b"weights"), ("a\\..\\..\\other.json", b"{}")]);
    assert_eq!(unpack(&zip, &dir).unwrap(), ["escaped.onnx", "other.json"]);
    assert!(dir.join("escaped.onnx").is_file() && !dir.parent().unwrap().join("escaped.onnx").exists());
}

#[test]
fn weights_that_are_not_the_ones_described_are_refused() {
    let dir = scratch("bad");
    let (d, r) = descriptions(b"detector weights", b"reader weights");
    let zip = zipped(&[
        ("cube-detector.json", d.as_bytes()),
        ("cube-reader.json", r.as_bytes()),
        ("cube-detector.fp16.onnx", b"detector weights"),
        ("cube-reader.fp16.onnx", b"cut short"),
    ]);
    unpack(&zip, &dir).unwrap();
    let error = verify(&dir).unwrap_err().to_string();
    assert!(error.contains("cube-reader.fp16.onnx") && error.contains("fetch the models again"), "{error}");
    // Weights missing, a description missing, a description that names a file elsewhere
    std::fs::remove_file(dir.join("cube-reader.fp16.onnx")).unwrap();
    assert!(format!("{:#}", verify(&dir).unwrap_err()).contains("is missing"));
    std::fs::write(dir.join("cube-reader.json"), r#"{"variants":{"full":{"file":"../secret","sha256":"00"}}}"#).unwrap();
    assert!(verify(&dir).unwrap_err().to_string().contains("outside the folder"));
    std::fs::remove_file(dir.join("cube-reader.json")).unwrap();
    assert!(format!("{:#}", verify(&dir).unwrap_err()).contains("cube-reader.json"));
    // Not a zip, an empty zip
    assert!(unpack(b"not a zip", &dir).unwrap_err().to_string().contains("not a zip"));
    assert!(unpack(&zipped(&[("readme.md", b"hello")]), &dir).unwrap_err().to_string().contains("no model"));
}
