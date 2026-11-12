//! The detector and the reader on ONNX Runtime, on the device asked for and nowhere else.
//!
//! The models' folder holds `cube-detector.json` and `cube-reader.json` (what the training exported: input,
//! sizes, thresholds, files) next to the weights. ONNX Runtime is loaded at run time: `ORT_DYLIB_PATH` names its
//! library, or it sits next to the executable.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use ort::environment::Environment;
use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::Session;
use ort::value::TensorRef;
use rayon::prelude::*;
use serde_json::Value;

use crate::device::{AUTO, Device, Precision};
use crate::follow::{Detection, Detector, Look};
use crate::geometry::{Letterbox, check, letterbox, tensor, unletterbox};
use crate::output::{Reader, ReaderOutput};

/// Images run together. A GPU session that changes input shape from one run to the next pays about 100 ms, so
/// batches are always padded to this size, and single images run in a session of their own.
pub const BATCH: usize = 8;

const FETCH: &str = "run `cubevision fetch-models`, or name the folder that holds them";

/// `cube-detector.json`
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorMeta {
    /// The input's name
    pub input: String,
    /// Side of the square it looks at
    pub size: usize,
    /// The grey of the bands around the frame
    pub letterbox: f32,
    /// Score from which a cube is there
    pub threshold: f32,
    /// The reader's square: the box's longer side times this
    pub crop_margin: f64,
    full: String,
    lite: String,
}

impl DetectorMeta {
    pub fn file(&self, precision: Precision) -> &str {
        if precision == Precision::Fp16 { &self.full } else { &self.lite }
    }
}

/// `cube-reader.json`
#[derive(Clone, Debug, PartialEq)]
pub struct ReaderMeta {
    pub input: String,
    /// Its stickers come in the cube's order, as the centres seen turn it
    pub ordered: bool,
    full: (String, usize),
    lite: (String, usize),
}

impl ReaderMeta {
    /// The weights' file and the side of the square it reads
    pub fn variant(&self, precision: Precision) -> (&str, usize) {
        let (file, size) = if precision == Precision::Fp16 { &self.full } else { &self.lite };
        (file, *size)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub detector: DetectorMeta,
    pub reader: ReaderMeta,
}

fn description(dir: &Path, name: &str) -> Result<Value> {
    let path = dir.join(name);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!("no models in {} ({name} is missing): {FETCH}", dir.display()),
        Err(e) => bail!("cannot read {}: {e}", path.display()),
    };
    serde_json::from_str(&text).with_context(|| format!("{} is not a model's description", path.display()))
}

fn text(v: &Value, what: &str) -> Result<String> {
    v.as_str().map(str::to_owned).with_context(|| format!("no {what}"))
}

fn number(v: &Value, what: &str) -> Result<f64> {
    v.as_f64().filter(|x| x.is_finite()).with_context(|| format!("no {what}"))
}

/// A square's side: the models take a few hundred pixels
fn side(v: &Value, what: &str) -> Result<usize> {
    v.as_u64().filter(|s| (1..=4096).contains(s)).map(|s| s as usize).with_context(|| format!("no {what} (a size in pixels)"))
}

/// What the models' folder says of them
pub fn metadata(dir: &Path) -> Result<Metadata> {
    let d = description(dir, "cube-detector.json")?;
    let detector = (|| -> Result<DetectorMeta> {
        Ok(DetectorMeta {
            input: text(&d["input"]["name"], "input.name")?,
            size: side(&d["input"]["size"], "input.size")?,
            letterbox: number(&d["input"]["letterbox"], "input.letterbox")? as f32,
            threshold: number(&d["threshold"], "threshold")? as f32,
            crop_margin: number(&d["cropMargin"], "cropMargin")?,
            full: text(&d["variants"]["full"]["file"], "variants.full.file")?,
            lite: text(&d["variants"]["lite"]["file"], "variants.lite.file")?,
        })
    })()
    .with_context(|| format!("{} does not describe a detector", dir.join("cube-detector.json").display()))?;
    let r = description(dir, "cube-reader.json")?;
    let reader = (|| -> Result<ReaderMeta> {
        let variant = |name: &str| -> Result<(String, usize)> {
            let v = &r["variants"][name];
            Ok((text(&v["file"], &format!("variants.{name}.file"))?, side(&v["size"], &format!("variants.{name}.size"))?))
        };
        Ok(ReaderMeta {
            input: text(&r["input"]["name"], "input.name")?,
            ordered: r["cells"] == "cube",
            full: variant("full")?,
            lite: variant("lite")?,
        })
    })()
    .with_context(|| format!("{} does not describe a reader", dir.join("cube-reader.json").display()))?;
    Ok(Metadata { detector, reader })
}

/// The ONNX Runtime library: the one `ORT_DYLIB_PATH` names, else the one next to the executable
pub fn library() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH").filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    let name = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    let beside = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf));
    for dir in beside.iter().flat_map(|dir| [dir.clone(), dir.join("ort")]) {
        if dir.join(name).is_file() {
            return Ok(dir.join(name));
        }
    }
    bail!("ONNX Runtime not found: set ORT_DYLIB_PATH to its library ({name}), or put it next to the executable")
}

/// ONNX Runtime allows one environment per process
fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let path = library()?;
    let env = ort::init_from(&path)
        .map_err(|e| {
            anyhow!(
                "cannot load ONNX Runtime from {} (ORT_DYLIB_PATH names its library, or it sits next to the executable): {e}",
                path.display()
            )
        })?
        .with_name("cubevision")
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

fn open(path: &Path, device: Device) -> Result<Session> {
    let mut builder = Session::builder(environment()?).map_err(|e| anyhow!("{e}"))?;
    if let Some(provider) = provider(device) {
        builder = builder
            .with_execution_providers([provider])
            .map_err(|e| anyhow!("{e}"))?
            // On a GPU the processor's threads mostly wait: one, not spinning, leaves the cores to decoding
            .with_intra_threads(1)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_op_spinning(false)
            .map_err(|e| anyhow!("{e}"))?;
    }
    builder.commit_from_file(path).map_err(|e| anyhow!("cannot open {}: {e}", path.display()))
}

/// One model, in a session for single images and another for batches, each opened when first needed
struct Shaped {
    path: PathBuf,
    device: Device,
    input: String,
    one: Option<Session>,
    many: Option<Session>,
}

impl Shaped {
    fn new(path: PathBuf, device: Device, input: &str) -> Result<Shaped> {
        if !path.is_file() {
            bail!("{} is missing: {FETCH}", path.display());
        }
        Ok(Shaped { path, device, input: input.to_owned(), one: None, many: None })
    }

    fn session(&mut self, n: usize) -> Result<&mut Session> {
        let slot = if n == 1 { &mut self.one } else { &mut self.many };
        if slot.is_none() {
            *slot = Some(open(&self.path, self.device)?);
        }
        Ok(slot.as_mut().unwrap())
    }
}

/// The trained cube detector: where the whole cube is in a frame
pub struct OnnxDetector {
    meta: DetectorMeta,
    model: Shaped,
}

impl OnnxDetector {
    /// Side of the square it looks at
    pub fn size(&self) -> usize {
        self.meta.size
    }

    /// The model itself on `n` letterboxed squares (planes R, G, B in 0..1, one image after the other): for
    /// each, its box (x0 y0 x1 y1, shares of the square) and its score
    pub fn run(&mut self, tensors: &[f32], n: usize) -> Result<Vec<([f32; 4], f32)>> {
        let size = self.meta.size;
        if n == 0 || tensors.len() != n * 3 * size * size {
            bail!("{} values where the detector takes {n} squares of {size}×{size}", tensors.len());
        }
        let name = self.model.input.clone();
        let input = TensorRef::from_array_view(([n, 3, size, size], tensors))?;
        let outputs = self.model.session(n)?.run(ort::inputs![name => input])?;
        let (_, boxed) = outputs["box"].try_extract_tensor::<f32>()?;
        let (_, score) = outputs["score"].try_extract_tensor::<f32>()?;
        if boxed.len() < 4 * n || score.len() < n {
            bail!("the detector gave {} box values and {} scores for {n} squares", boxed.len(), score.len());
        }
        Ok((0..n).map(|i| ([boxed[4 * i], boxed[4 * i + 1], boxed[4 * i + 2], boxed[4 * i + 3]], score[i])).collect())
    }

    fn found(&self, (boxed, score): ([f32; 4], f32), lb: &Letterbox) -> Option<Detection> {
        // A score that is no number is no cube, and a box no frame can hold neither
        let detection = Detection { boxed: unletterbox(boxed, self.meta.size, lb), score };
        (score >= self.meta.threshold && detection.boxed.is_real()).then_some(detection)
    }
}

impl Detector for OnnxDetector {
    fn crop_margin(&self) -> f64 {
        self.meta.crop_margin
    }

    fn look(&mut self, looks: &[Look]) -> Result<Vec<Option<Detection>>> {
        for look in looks {
            check(look.frame)?;
        }
        let (size, pad) = (self.meta.size, self.meta.letterbox);
        if let [look] = looks {
            let (square, lb) = letterbox(look.frame, size, pad, look.region.as_ref());
            let found = self.run(&square, 1)?.remove(0);
            return Ok(vec![self.found(found, &lb)]);
        }
        let plane = 3 * size * size;
        let mut out = Vec::with_capacity(looks.len());
        for chunk in looks.chunks(BATCH) {
            let boxed: Vec<(Vec<f32>, Letterbox)> =
                chunk.par_iter().map(|look| letterbox(look.frame, size, pad, look.region.as_ref())).collect();
            let mut tensors = Vec::with_capacity(BATCH * plane);
            boxed.iter().for_each(|(square, _)| tensors.extend_from_slice(square));
            tensors.resize(BATCH * plane, pad);
            let found = self.run(&tensors, BATCH)?;
            out.extend(found.into_iter().zip(&boxed).map(|(found, (_, lb))| self.found(found, lb)));
        }
        Ok(out)
    }
}

/// The trained reader: which faces a crop of the cube shows, and their stickers
pub struct OnnxReader {
    meta: ReaderMeta,
    size: usize,
    model: Shaped,
}

impl OnnxReader {
    /// The model itself on `n` squares (planes R, G, B in 0..1, one image after the other)
    pub fn run(&mut self, tensors: &[f32], n: usize) -> Result<Vec<ReaderOutput>> {
        let size = self.size;
        if n == 0 || tensors.len() != n * 3 * size * size {
            bail!("{} values where the reader takes {n} squares of {size}×{size}", tensors.len());
        }
        let name = self.model.input.clone();
        let input = TensorRef::from_array_view(([n, 3, size, size], tensors))?;
        let outputs = self.model.session(n)?.run(ort::inputs![name => input])?;
        let (_, present) = outputs["present"].try_extract_tensor::<f32>()?;
        let (_, stickers) = outputs["stickers"].try_extract_tensor::<f32>()?;
        let (_, turning) = outputs["turning"].try_extract_tensor::<f32>()?;
        let orientation = match outputs.get("orientation") {
            Some(value) => Some(value.try_extract_tensor::<f32>()?.1).filter(|data| data.len() == 9 * n),
            None => None,
        };
        if present.len() != 6 * n || stickers.len() % n != 0 || turning.len() < n {
            bail!(
                "the reader gave {} face values, {} sticker values and {} turning values for {n} squares",
                present.len(),
                stickers.len(),
                turning.len()
            );
        }
        let per = stickers.len() / n;
        Ok((0..n)
            .map(|i| ReaderOutput {
                present: present[6 * i..6 * i + 6].try_into().unwrap(),
                stickers: stickers[per * i..per * (i + 1)].to_vec(),
                turning: turning[i * (turning.len() / n)],
                orientation: orientation.map(|data| data[9 * i..9 * i + 9].try_into().unwrap()),
            })
            .collect())
    }
}

impl Reader for OnnxReader {
    /// Side of the square it reads
    fn size(&self) -> usize {
        self.size
    }

    /// Its stickers come in the cube's order
    fn ordered(&self) -> bool {
        self.meta.ordered
    }

    /// Crops read: RGBA squares of `size()`, as `square_crop` cuts them. One crop runs alone; several run in
    /// batches (far fewer runs' overhead per crop on a GPU), with the same outcome.
    fn read(&mut self, crops: &[&[u8]]) -> Result<Vec<ReaderOutput>> {
        let size = self.size;
        let plane = 3 * size * size;
        if let Some(crop) = crops.iter().find(|crop| crop.len() != size * size * 4) {
            bail!("a crop of {} bytes where the reader takes {size}×{size} RGBA", crop.len());
        }
        if let [crop] = crops {
            let mut square = vec![0.0; plane];
            tensor(crop, &mut square);
            return self.run(&square, 1);
        }
        let mut out = Vec::with_capacity(crops.len());
        for chunk in crops.chunks(BATCH) {
            let mut tensors = vec![0.0; BATCH * plane];
            tensors.par_chunks_mut(plane).zip(chunk).for_each(|(square, crop)| tensor(crop, square));
            out.extend(self.run(&tensors, BATCH)?.into_iter().take(chunk.len()));
        }
        Ok(out)
    }
}

/// Both models, on one device
pub struct Models {
    pub detector: OnnxDetector,
    pub reader: OnnxReader,
    /// Where they run
    pub device: Device,
    /// Set when no device was asked for and no GPU engine could run the models: why they run on the processor
    pub fallback: Option<String>,
}

/// An error short enough to show: its end, where the cause is (the start of ONNX Runtime's is where in its
/// sources it happened)
fn brief(error: &anyhow::Error) -> String {
    let text = format!("{error:#}").split_whitespace().collect::<Vec<_>>().join(" ");
    let skip = text.chars().count().saturating_sub(200);
    if skip == 0 { text } else { format!("…{}", text.chars().skip(skip).collect::<String>()) }
}

impl Models {
    /// The models of `dir` on `device`, or on the first device of `AUTO` that runs them when none is asked for.
    /// A device asked for by name is that one or an error: never another in silence.
    pub fn load(dir: &Path, device: Option<Device>) -> Result<Models> {
        let meta = metadata(dir)?;
        // The weights first: without them no engine matters
        let weights = |precision| -> Result<(Shaped, Shaped, usize)> {
            let (file, size) = meta.reader.variant(precision);
            // The device is set when the sessions open
            Ok((
                Shaped::new(dir.join(meta.detector.file(precision)), Device::Cpu, &meta.detector.input)?,
                Shaped::new(dir.join(file), Device::Cpu, &meta.reader.input)?,
                size,
            ))
        };
        let on = |device: Device| -> Result<Models> {
            let (mut detector, mut reader, size) = weights(device.precision())?;
            environment()?;
            (detector.device, reader.device) = (device, device);
            let mut models = Models {
                detector: OnnxDetector { meta: meta.detector.clone(), model: detector },
                reader: OnnxReader { meta: meta.reader.clone(), size, model: reader },
                device,
                fallback: None,
            };
            // A first run, blank: it proves the device runs the models, and pays their warming up here
            models.warm(1)?;
            Ok(models)
        };
        if let Some(device) = device {
            // What no other device would mend is said as it is: weights missing, ONNX Runtime not found
            weights(device.precision())?;
            environment()?;
            return on(device).with_context(|| match device {
                Device::Cpu => {
                    "the processor cannot run the models: are their files whole? (`cubevision fetch-models` gets them again)".to_string()
                }
                _ => format!("{} cannot run the models here (another device: --device {})", device.name(), Device::names()),
            });
        }
        // Both precisions: a GPU's weights missing must not look like a machine without a GPU
        weights(Precision::Fp16)?;
        weights(Precision::Fp32)?;
        environment()?;
        let mut failures = Vec::new();
        for device in AUTO {
            match on(device) {
                Ok(mut models) => {
                    if device == Device::Cpu && !failures.is_empty() {
                        models.fallback = Some(format!(
                            "no GPU engine runs the models here, they run on the processor, slower ({})",
                            failures.join("; ")
                        ));
                    }
                    return Ok(models);
                }
                Err(e) => failures.push(format!("{}: {}", device.name(), brief(&e))),
            }
        }
        bail!("no device runs the models here ({})", failures.join("; "))
    }

    /// A blank run of both models for `n` images at once (1, or `BATCH` for a video): opens the sessions of that
    /// shape and pays their first run, far slower than the next ones on a GPU
    pub fn warm(&mut self, n: usize) -> Result<()> {
        let n = if n == 1 { 1 } else { BATCH };
        let size = self.detector.meta.size;
        self.detector.run(&vec![self.detector.meta.letterbox; n * 3 * size * size], n)?;
        let size = self.reader.size;
        self.reader.run(&vec![0.0; n * 3 * size * size], n)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_error_is_shown_by_its_end() {
        assert_eq!(brief(&anyhow!("short\n  and  plain")), "short and plain");
        let long = anyhow!("{} libcublasLt.so.13: cannot open shared object file", "/onnxruntime/core/providers/x.cc:42 ".repeat(20));
        let shown = brief(&long);
        assert!(
            shown.starts_with('…') && shown.ends_with("libcublasLt.so.13: cannot open shared object file") && shown.chars().count() == 201,
            "{shown}"
        );
    }
}
