//! What the window remembers from one start to the next (the engine, the camera, the last scramble), and the
//! engines an install can run the models on

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use cube_infer::device::Device;
use serde_json::{Value, json};

/// Where the models run, as the window offers it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// The first device that works
    Auto,
    DirectMl,
    IntelGpu,
    IntelCpu,
    Cuda,
    Cpu,
}

impl Engine {
    pub const ALL: [Engine; 6] = [Engine::Auto, Engine::DirectMl, Engine::IntelGpu, Engine::IntelCpu, Engine::Cuda, Engine::Cpu];

    /// Its name in the settings' file
    pub fn key(self) -> &'static str {
        match self {
            Engine::Auto => "auto",
            Engine::DirectMl => "directml",
            Engine::IntelGpu => "openvino-gpu",
            Engine::IntelCpu => "openvino-cpu",
            Engine::Cuda => "cuda",
            Engine::Cpu => "cpu",
        }
    }

    pub fn of(key: &str) -> Option<Engine> {
        Engine::ALL.into_iter().find(|engine| engine.key() == key)
    }

    /// Its name in the window
    pub fn label(self) -> &'static str {
        match self {
            Engine::Auto => "Automatic",
            Engine::DirectMl => "Graphics card (DirectML)",
            Engine::IntelGpu => "Intel graphics (OpenVINO)",
            Engine::IntelCpu => "Processor (OpenVINO)",
            Engine::Cuda => "NVIDIA graphics (CUDA)",
            Engine::Cpu => "Processor alone (slow)",
        }
    }

    /// The device asked of the models; None: the first that works
    pub fn device(self) -> Option<Device> {
        match self {
            Engine::Auto => None,
            Engine::DirectMl => Some(Device::DirectMl),
            Engine::IntelGpu => Some(Device::OpenVinoGpu),
            Engine::IntelCpu => Some(Device::OpenVinoCpu),
            Engine::Cuda => Some(Device::Cuda),
            Engine::Cpu => Some(Device::Cpu),
        }
    }

    /// The folders of `runtime` whose ONNX Runtime runs it, the preferred one first
    fn folders(self) -> &'static [&'static str] {
        match self {
            Engine::Auto => &["directml", "cuda", "openvino"],
            Engine::DirectMl => &["directml"],
            Engine::IntelGpu | Engine::IntelCpu => &["openvino"],
            Engine::Cuda => &["cuda"],
            Engine::Cpu => &["openvino", "directml", "cuda"],
        }
    }
}

/// The name of the ONNX Runtime library on this system
pub fn library_name() -> &'static str {
    if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// The ONNX Runtime an engine runs on, in the `runtime` folder beside the program (the released zip has one
/// folder per runtime); None: there is none there, the one the system names is used
pub fn library(engine: Engine, beside: &Path) -> Option<PathBuf> {
    engine.folders().iter().map(|folder| beside.join("runtime").join(folder).join(library_name())).find(|file| file.is_file())
}

/// The engines offered: with a `runtime` folder beside the program, those it holds a runtime for; without one
/// (a build run from its sources), those the system may have
pub fn engines(beside: &Path, windows: bool) -> Vec<Engine> {
    let released = beside.join("runtime").is_dir();
    Engine::ALL
        .into_iter()
        .filter(|&engine| if released { library(engine, beside).is_some() } else { engine != Engine::DirectMl || windows })
        .collect()
}

/// The engine of a first start: Intel graphics run the models about twice as fast with OpenVINO as with DirectML
pub fn first_engine(offered: &[Engine], intel_graphics: bool) -> Engine {
    let wanted = if intel_graphics { [Engine::IntelGpu, Engine::DirectMl] } else { [Engine::DirectMl, Engine::Auto] };
    wanted.into_iter().find(|engine| offered.contains(engine)).or(offered.first().copied()).unwrap_or(Engine::Auto)
}

/// Points this process at an engine's runtime, when the program has one beside it: to call before the models
/// are first loaded. The library it will load, if it is one of the program's.
pub fn point_at(engine: Engine, beside: &Path) -> Option<PathBuf> {
    let file = library(engine, beside)?;
    let folder = file.parent()?.to_path_buf();
    let mut path = vec![folder];
    path.extend(std::env::var_os("PATH").iter().flat_map(std::env::split_paths));
    // SAFETY: called from the window's thread before any model is loaded; the reading threads start after
    unsafe {
        std::env::set_var("ORT_DYLIB_PATH", &file);
        if let Ok(joined) = std::env::join_paths(path) {
            std::env::set_var("PATH", joined);
        }
    }
    Some(file)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// None: not chosen yet
    pub engine: Option<Engine>,
    pub camera: u32,
    /// The camera shown as a mirror
    pub mirror: bool,
    /// The last video's scramble and folder, the last labelling note
    pub scramble: String,
    pub folder: Option<PathBuf>,
    pub note: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { engine: None, camera: 0, mirror: true, scramble: String::new(), folder: None, note: String::new() }
    }
}

impl Settings {
    /// The settings' file: `CUBEVISION_SETTINGS` when set, else `cubevision/settings.json` in the user's
    /// configuration folder
    pub fn file() -> Option<PathBuf> {
        if let Some(file) = std::env::var_os("CUBEVISION_SETTINGS").filter(|f| !f.is_empty()) {
            return Some(PathBuf::from(file));
        }
        let base = if cfg!(windows) {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("XDG_CONFIG_HOME")
                .filter(|dir| !dir.is_empty())
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        };
        Some(base?.join("cubevision").join("settings.json"))
    }

    /// The settings a file holds; what it lacks, or a file that cannot be read, gives the defaults
    pub fn load(file: &Path) -> Settings {
        let read = std::fs::read_to_string(file).ok().and_then(|text| serde_json::from_str::<Value>(&text).ok());
        let Some(value) = read else { return Settings::default() };
        let base = Settings::default();
        let text = |key: &str| value[key].as_str().map(str::to_owned);
        Settings {
            engine: value["engine"].as_str().and_then(Engine::of),
            camera: value["camera"].as_u64().map_or(base.camera, |index| index as u32),
            mirror: value["mirror"].as_bool().unwrap_or(base.mirror),
            scramble: text("scramble").unwrap_or_default(),
            folder: text("folder").map(PathBuf::from),
            note: text("note").unwrap_or_default(),
        }
    }

    pub fn save(&self, file: &Path) -> Result<()> {
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let value = json!({
            "engine": self.engine.map(Engine::key),
            "camera": self.camera,
            "mirror": self.mirror,
            "scramble": self.scramble,
            "folder": self.folder.as_ref().map(|folder| folder.display().to_string()),
            "note": self.note,
        });
        std::fs::write(file, serde_json::to_string_pretty(&value)?).with_context(|| format!("writing {}", file.display()))
    }
}
