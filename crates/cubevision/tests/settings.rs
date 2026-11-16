//! The settings kept between starts, and the engines an install offers

use std::path::{Path, PathBuf};

use cubevision::settings::{Engine, Settings, engines, first_engine, library, library_name};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cubevision-settings-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn runtime(beside: &Path, folder: &str) {
    let dir = beside.join("runtime").join(folder);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(library_name()), b"").unwrap();
}

#[test]
fn settings_come_back_as_saved() {
    let file = scratch("round").join("deep").join("settings.json");
    let settings = Settings {
        engine: Some(Engine::IntelGpu),
        camera: 2,
        mirror: false,
        scramble: "R U R' U'".into(),
        folder: Some(PathBuf::from("/videos")),
        note: "desk, lamp".into(),
    };
    settings.save(&file).unwrap();
    assert_eq!(Settings::load(&file), settings);
}

#[test]
fn a_missing_or_broken_file_gives_the_defaults() {
    let dir = scratch("broken");
    assert_eq!(Settings::load(&dir.join("none.json")), Settings::default());
    std::fs::write(dir.join("broken.json"), "{ not json").unwrap();
    assert_eq!(Settings::load(&dir.join("broken.json")), Settings::default());
    // An engine this version does not know: not chosen
    std::fs::write(dir.join("old.json"), r#"{"engine": "quantum", "camera": 1}"#).unwrap();
    let old = Settings::load(&dir.join("old.json"));
    assert_eq!((old.engine, old.camera, old.mirror), (None, 1, true));
}

#[test]
fn a_released_folder_offers_the_engines_it_has_a_runtime_for() {
    let beside = scratch("released");
    runtime(&beside, "directml");
    assert_eq!(engines(&beside, true), [Engine::Auto, Engine::DirectMl, Engine::Cpu]);
    runtime(&beside, "openvino");
    assert_eq!(engines(&beside, true), [Engine::Auto, Engine::DirectMl, Engine::IntelGpu, Engine::IntelCpu, Engine::Cpu]);
    // The processor alone runs on the OpenVINO build when there is one, the automatic choice on DirectML's
    assert!(library(Engine::Cpu, &beside).unwrap().ends_with(Path::new("openvino").join(library_name())));
    assert!(library(Engine::Auto, &beside).unwrap().ends_with(Path::new("directml").join(library_name())));
}

#[test]
fn a_build_from_the_sources_offers_what_the_system_may_have() {
    let beside = scratch("sources");
    assert!(engines(&beside, true).contains(&Engine::DirectMl));
    assert!(!engines(&beside, false).contains(&Engine::DirectMl));
    assert_eq!(library(Engine::Auto, &beside), None);
}

#[test]
fn intel_graphics_start_on_openvino() {
    let all = Engine::ALL;
    assert_eq!(first_engine(&all, true), Engine::IntelGpu);
    assert_eq!(first_engine(&all, false), Engine::DirectMl);
    assert_eq!(first_engine(&[Engine::Auto, Engine::DirectMl, Engine::Cpu], true), Engine::DirectMl);
    assert_eq!(first_engine(&[Engine::Auto, Engine::Cuda, Engine::Cpu], false), Engine::Auto);
    assert_eq!(first_engine(&[], false), Engine::Auto);
}

#[test]
fn engines_keep_their_names() {
    for engine in Engine::ALL {
        assert_eq!(Engine::of(engine.key()), Some(engine));
    }
    assert_eq!(Engine::of("npu"), None);
}
