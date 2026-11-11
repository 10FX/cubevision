//! Frames out of a video file: ffprobe for its size, rate and rotation, ffmpeg piping raw RGB

use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::preprocess::Frame;

pub struct VideoReader {
    child: Child,
    stdout: ChildStdout,
    width: u32,
    height: u32,
    fps: f64,
    codec: String,
}

/// ffprobe next to ffmpeg: the same path with "ffmpeg" in the file name changed
fn ffprobe_for(ffmpeg: &str) -> PathBuf {
    let path = Path::new(ffmpeg);
    let name = path.file_name().map(|n| n.to_string_lossy().replace("ffmpeg", "ffprobe")).unwrap_or_else(|| "ffprobe".into());
    path.with_file_name(name)
}

fn rate(text: &str) -> Option<f64> {
    let (num, den) = text.split_once('/')?;
    let (num, den): (f64, f64) = (num.parse().ok()?, den.parse().ok()?);
    (num > 0.0 && den > 0.0).then_some(num / den)
}

impl VideoReader {
    pub fn open(path: &Path, ffmpeg: &str) -> Result<VideoReader> {
        let ffprobe = ffprobe_for(ffmpeg);
        let probe = Command::new(&ffprobe)
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate:stream_side_data=rotation", "-of", "json"])
            .arg(path)
            .output()
            .with_context(|| format!("cannot run {} (ffmpeg's ffprobe): is ffmpeg installed and in the PATH?", ffprobe.display()))?;
        if !probe.status.success() {
            bail!("ffprobe cannot read {}: {}", path.display(), String::from_utf8_lossy(&probe.stderr).trim());
        }
        let info: Value = serde_json::from_slice(&probe.stdout)?;
        let stream = info["streams"].get(0).with_context(|| format!("{} has no video stream", path.display()))?;
        let (mut width, mut height) = (stream["width"].as_u64().unwrap_or(0) as u32, stream["height"].as_u64().unwrap_or(0) as u32);
        if width == 0 || height == 0 {
            bail!("{} has no frame size", path.display());
        }
        let rotation = stream["side_data_list"].as_array().into_iter().flatten().find_map(|d| d["rotation"].as_f64()).unwrap_or(0.0);
        if (rotation.abs() as i64) % 180 == 90 {
            std::mem::swap(&mut width, &mut height); // ffmpeg turns the frames upright
        }
        let fps = stream["avg_frame_rate"].as_str().and_then(rate).or_else(|| stream["r_frame_rate"].as_str().and_then(rate)).unwrap_or(30.0);
        let codec = stream["codec_name"].as_str().unwrap_or("?").to_string();
        let mut child = Command::new(ffmpeg)
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(path)
            .args(["-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("cannot run {ffmpeg}: is ffmpeg installed and in the PATH?"))?;
        let stdout = child.stdout.take().context("ffmpeg's output")?;
        Ok(VideoReader { child, stdout, width, height, fps, codec })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }

    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The next whole frame; None at the end, a last incomplete frame included
    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        let mut data = vec![0; self.width as usize * self.height as usize * 3];
        match self.stdout.read_exact(&mut data) {
            Ok(()) => Ok(Some(Frame { data, width: self.width, height: self.height })),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
