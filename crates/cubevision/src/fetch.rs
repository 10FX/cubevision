//! `cubevision fetch-models`: the released models downloaded from GitHub, unpacked and checked against the
//! checksums their own description gives

use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

/// The models' zip of the latest release
pub const URL: &str = "https://github.com/10FX/cubevision/releases/latest/download/cubevision-models-v7.zip";

/// Largest zip and largest file taken: several times the released ones, far under a zip bomb
const MOST: u64 = 400 << 20;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The files a models' folder holds: the two descriptions, the weights, the licence
fn wanted(name: &str) -> bool {
    name.ends_with(".onnx") || name.ends_with(".json") || name == "LICENSE.txt"
}

/// A models' zip unpacked into `dir`: its files by their own names (whatever folder the zip keeps them in),
/// nothing else, nothing outside `dir`. The names written.
pub fn unpack(zip: &[u8], dir: &Path) -> Result<Vec<String>> {
    let mut archive = zip::ZipArchive::new(Cursor::new(zip)).context("the download is not a zip")?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut written = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        // The last part of the name only: a zip cannot write outside the folder
        let name = entry.name().rsplit(['/', '\\']).next().unwrap_or_default().to_owned();
        if !wanted(&name) || name.starts_with('.') {
            continue;
        }
        if entry.size() > MOST {
            bail!("{name} in the zip is {} bytes: not a model", entry.size());
        }
        let mut data = Vec::with_capacity(entry.size() as usize);
        entry.by_ref().take(MOST).read_to_end(&mut data)?;
        std::fs::write(dir.join(&name), &data).with_context(|| format!("writing {}", dir.join(&name).display()))?;
        written.push(name);
    }
    if written.is_empty() {
        bail!("the zip holds no model");
    }
    written.sort();
    Ok(written)
}

/// Whether the weights in `dir` are those their descriptions tell: each variant's file against its sha256
/// (a description without one is taken as it is). The files checked.
pub fn verify(dir: &Path) -> Result<Vec<String>> {
    let mut checked = Vec::new();
    for description in ["cube-detector.json", "cube-reader.json"] {
        let path = dir.join(description);
        let text = std::fs::read_to_string(&path).with_context(|| format!("{} is missing", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("{} is not a model's description", path.display()))?;
        for variant in json["variants"].as_object().into_iter().flat_map(|v| v.values()) {
            let (Some(file), Some(want)) = (variant["file"].as_str(), variant["sha256"].as_str()) else { continue };
            // A description may name any file: only one of this folder is read
            if file.contains(['/', '\\']) {
                bail!("{description} names a file outside the folder: {file}");
            }
            let data = std::fs::read(dir.join(file)).with_context(|| format!("{} is missing", dir.join(file).display()))?;
            let got = hex(&Sha256::digest(&data));
            if !got.eq_ignore_ascii_case(want) {
                bail!("{file} is not the file its description tells (sha256 {got}, expected {want}): fetch the models again");
            }
            checked.push(file.to_owned());
        }
    }
    Ok(checked)
}

/// The released models downloaded into `dir`, unless they are there already and whole
pub fn fetch(dir: &Path, url: &str) -> Result<()> {
    if let Ok(checked) = verify(dir)
        && !checked.is_empty()
    {
        println!("the models are already in {} ({} files checked)", dir.display(), checked.len());
        return Ok(());
    }
    println!("downloading {url}");
    let response = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let mut zip = Vec::new();
    response.into_reader().take(MOST).read_to_end(&mut zip).context("the download stopped")?;
    println!("{:.1} MB", zip.len() as f64 / (1 << 20) as f64);
    let written = unpack(&zip, dir)?;
    let checked = verify(dir)?;
    println!("{} files in {}, {} checked against their checksums: {}", written.len(), dir.display(), checked.len(), written.join(", "));
    Ok(())
}
