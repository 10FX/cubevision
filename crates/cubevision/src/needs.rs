//! What a mode needs besides the program itself, and how to get it: ffmpeg to read a video, the models

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Result, bail};
use cube_capture::quiet;

fn runs(program: &str) -> bool {
    quiet(Command::new(program).arg("-version"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Whether ffmpeg and its ffprobe can be run: a video is read through them
pub fn ffmpeg_ready() -> bool {
    runs("ffmpeg") && runs("ffprobe")
}

/// The folder winget links the programs it installs in: on the PATH of programs started afterwards only
fn winget_links() -> Option<PathBuf> {
    let links = PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("Microsoft").join("WinGet").join("Links");
    links.is_dir().then_some(links)
}

/// `dir` joined to this process's PATH, unless it is there
fn join_path(dir: &Path) {
    let mut path: Vec<PathBuf> = std::env::var_os("PATH").iter().flat_map(std::env::split_paths).collect();
    if path.iter().any(|known| known == dir) {
        return;
    }
    path.push(dir.to_path_buf());
    if let Ok(joined) = std::env::join_paths(path) {
        // SAFETY: called from the window's thread; the programs started later read the PATH they are given
        unsafe { std::env::set_var("PATH", joined) };
    }
}

/// ffmpeg found where winget puts it, when it is not on the PATH yet (installed since this session began):
/// whether it can be run now
pub fn find_ffmpeg() -> bool {
    if ffmpeg_ready() {
        return true;
    }
    match winget_links() {
        Some(links) => {
            join_path(&links);
            ffmpeg_ready()
        }
        None => false,
    }
}

/// Whether this system can install ffmpeg for the user (Windows, with winget)
pub fn can_install_ffmpeg() -> bool {
    cfg!(windows)
        && quiet(Command::new("winget").arg("--version")).stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// What to type to install it oneself
pub fn ffmpeg_command() -> &'static str {
    if cfg!(windows) {
        "winget install Gyan.FFmpeg"
    } else if cfg!(target_os = "macos") {
        "brew install ffmpeg"
    } else {
        "sudo apt install ffmpeg"
    }
}

/// ffmpeg installed with winget (a minute or two: about 100 MB to download). `find_ffmpeg` then tells whether
/// it runs.
pub fn install_ffmpeg() -> Result<()> {
    let run = quiet(Command::new("winget").args([
        "install",
        "--exact",
        "--id",
        "Gyan.FFmpeg",
        "--accept-source-agreements",
        "--accept-package-agreements",
        "--disable-interactivity",
    ]))
    .stdin(Stdio::null())
    .output();
    match run {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let said = String::from_utf8_lossy(&out.stdout);
            let last = said.lines().rev().find(|line| !line.trim().is_empty()).unwrap_or("no message").trim().to_owned();
            bail!("winget could not install ffmpeg: {last}")
        }
        Err(e) => bail!("winget could not be run: {e}"),
    }
}
