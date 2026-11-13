//! Frames out of a video file: ffprobe for its size, rate and rotation, ffmpeg piping raw RGBA and telling when
//! each frame is shown. Needs ffmpeg 5.1 or later, with its ffprobe next to it.

use std::io::{BufRead, BufReader, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::Image;

/// How a video is read
#[derive(Clone, Debug, PartialEq)]
pub struct VideoOptions {
    /// The ffmpeg to run: a name looked up in the PATH, or a path; ffprobe is taken next to it
    pub ffmpeg: String,
    /// Wider videos are shrunk to this width, as ChronoCube reads them (1280); None: as filmed
    pub max_width: Option<u32>,
    /// Where to start in the video, seconds: the frames before are left out, the times stay the video's
    pub start: f64,
}

impl Default for VideoOptions {
    fn default() -> Self {
        VideoOptions { ffmpeg: "ffmpeg".into(), max_width: Some(1280), start: 0.0 }
    }
}

/// What a video is, as its frames are handed over
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    /// Size of the frames read: upright, and shrunk when the video is wider than asked
    pub width: usize,
    pub height: usize,
    /// Frames a second, on average
    pub fps: f64,
    pub codec: String,
    /// The rotation the file asks for, degrees (the frames come upright)
    pub rotation: i32,
    /// Seconds, when the file tells
    pub duration: Option<f64>,
    /// Frames, when the file tells
    pub frames: Option<u64>,
}

pub struct VideoReader {
    child: Child,
    stdout: ChildStdout,
    info: VideoInfo,
    path: PathBuf,
    /// When each frame is shown (seconds), as ffmpeg's showinfo tells, in order
    times: Receiver<f64>,
    /// Once ffmpeg stopped telling: the times come from the frame rate
    untimed: bool,
    /// Where the reading started, seconds: ffmpeg counts the times from there
    start: f64,
    errors: Arc<Mutex<Vec<String>>>,
    read: u64,
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

/// A number ffprobe writes as text ("1.000000")
fn seconds(v: &Value) -> Option<f64> {
    v.as_str().and_then(|s| s.parse::<f64>().ok()).filter(|s| s.is_finite() && *s > 0.0)
}

/// JavaScript's Math.round: halves go up
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

/// The size the frames are read at: at most `max_width` wide, the height even, as ChronoCube asks ffmpeg
fn scaled(width: u32, height: u32, max_width: Option<u32>) -> (u32, u32) {
    match max_width {
        Some(max) => {
            let w = width.min(max.max(2));
            (w, (js_round(height as f64 * w as f64 / width as f64 / 2.0) * 2.0).max(2.0) as u32)
        }
        None => (width, height),
    }
}

impl VideoReader {
    pub fn open(path: &Path, options: &VideoOptions) -> Result<VideoReader> {
        let ffmpeg = &options.ffmpeg;
        let start = if options.start.is_finite() { options.start.max(0.0) } else { 0.0 };
        let ffprobe = ffprobe_for(ffmpeg);
        let probe = Command::new(&ffprobe)
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
            .arg("stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration,nb_frames:stream_side_data=rotation:format=duration")
            .args(["-of", "json"])
            .arg(path)
            .stdin(Stdio::null())
            .output()
            .with_context(|| format!("cannot run {} (ffmpeg's ffprobe): is ffmpeg installed and in the PATH?", ffprobe.display()))?;
        if !probe.status.success() {
            bail!("{} cannot be read as a video: {}", path.display(), String::from_utf8_lossy(&probe.stderr).trim());
        }
        let described: Value =
            serde_json::from_slice(&probe.stdout).with_context(|| format!("ffprobe's answer about {}", path.display()))?;
        let stream = described["streams"].get(0).with_context(|| format!("{} has no video stream", path.display()))?;
        let (mut width, mut height) = (stream["width"].as_u64().unwrap_or(0) as u32, stream["height"].as_u64().unwrap_or(0) as u32);
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            bail!("{} has no usable frame size ({width}×{height})", path.display());
        }
        let rotation = stream["side_data_list"].as_array().into_iter().flatten().find_map(|d| d["rotation"].as_f64()).unwrap_or(0.0) as i32;
        if rotation.abs() % 180 == 90 {
            std::mem::swap(&mut width, &mut height); // ffmpeg turns the frames upright
        }
        let (out_width, out_height) = scaled(width, height, options.max_width);
        let fps =
            stream["avg_frame_rate"].as_str().and_then(rate).or_else(|| stream["r_frame_rate"].as_str().and_then(rate)).unwrap_or(30.0);
        let info = VideoInfo {
            width: out_width as usize,
            height: out_height as usize,
            fps,
            codec: stream["codec_name"].as_str().unwrap_or("?").to_string(),
            rotation,
            duration: seconds(&stream["duration"]).or_else(|| seconds(&described["format"]["duration"])),
            frames: stream["nb_frames"].as_str().and_then(|n| n.parse().ok()),
        };
        // showinfo tells each frame's time on the error stream, one line a frame, in the order of the frames
        let filter = if options.max_width.is_some() { format!("scale={out_width}:{out_height},showinfo") } else { "showinfo".to_string() };
        let mut child = Command::new(ffmpeg)
            .args(["-nostdin", "-nostats", "-loglevel", "level+info"])
            // Before the input: ffmpeg jumps to the key frame before and decodes up to the time asked
            .args(if start > 0.0 { vec!["-ss".to_string(), format!("{start:.3}")] } else { Vec::new() })
            .arg("-i")
            .arg(path)
            .args(["-map", "0:v:0", "-vf", &filter, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("cannot run {ffmpeg}: is ffmpeg installed and in the PATH?"))?;
        let stdout = child.stdout.take().context("ffmpeg's output")?;
        let stderr = child.stderr.take().context("ffmpeg's messages")?;
        let (send, times) = channel();
        let errors = Arc::new(Mutex::new(Vec::new()));
        let noted = errors.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).split(b'\n').map_while(|line| line.ok()) {
                let line = String::from_utf8_lossy(&line);
                // showinfo's own lines only ("[Parsed_showinfo_1 @ …] n: 0 pts: 0 pts_time:0 …"): a tag of the file
                // may hold the same words
                if let Some((_, after)) =
                    line.split_once("pts_time:").filter(|(before, _)| before.contains("Parsed_showinfo") && before.contains(" n:"))
                {
                    // A frame without a time of its own still takes its place in the order
                    let _ = send.send(after.split_whitespace().next().and_then(|t| t.parse::<f64>().ok()).unwrap_or(f64::NAN));
                } else if line.contains("[error]") || line.contains("[fatal]") {
                    let mut noted = noted.lock().unwrap();
                    if noted.len() < 5 {
                        noted.push(line.trim().to_string());
                    }
                }
            }
        });
        Ok(VideoReader { child, stdout, info, path: path.to_path_buf(), times, untimed: false, start, errors, read: 0 })
    }

    pub fn info(&self) -> &VideoInfo {
        &self.info
    }

    /// When the frame just read is shown, ms: its own timestamp, or its rank at the video's rate when ffmpeg
    /// gives none
    fn time(&mut self) -> f64 {
        let told = if self.untimed { None } else { self.times.recv_timeout(Duration::from_secs(5)).ok() };
        match told.filter(|t| t.is_finite()) {
            Some(seconds) => js_round((self.start + seconds) * 1000.0),
            None => {
                self.untimed |= told.is_none();
                js_round(self.start * 1000.0 + self.read as f64 * 1000.0 / self.info.fps)
            }
        }
    }

    /// The next whole frame; None at the end (a last incomplete frame is left out)
    pub fn next_image(&mut self) -> Result<Option<Image>> {
        let (width, height) = (self.info.width, self.info.height);
        let mut data = vec![0; width * height * 4];
        match self.stdout.read_exact(&mut data) {
            Ok(()) => {
                let t = self.time();
                self.read += 1;
                Ok(Some(Image { data, width, height, t }))
            }
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                if self.read == 0 {
                    let _ = self.child.wait();
                    // The messages come from another thread: leave it the time to note the last ones
                    std::thread::sleep(Duration::from_millis(50));
                    let errors = self.errors.lock().unwrap().join("; ");
                    bail!(
                        "no frame could be read from {}{}",
                        self.path.display(),
                        if errors.is_empty() { String::new() } else { format!(": {errors}") }
                    );
                }
                Ok(None)
            }
            Err(e) => Err(e).with_context(|| format!("reading the frames of {}", self.path.display())),
        }
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_read_at_most_as_wide_as_asked_with_an_even_height() {
        assert_eq!(scaled(1920, 1080, Some(1280)), (1280, 720));
        assert_eq!(scaled(640, 480, Some(1280)), (640, 480));
        assert_eq!(scaled(1080, 1920, Some(1280)), (1080, 1920));
        // 853.3 rows: 854, as Math.round(h / 2) * 2
        assert_eq!(scaled(1920, 1280, Some(1280)), (1280, 854));
        assert_eq!(scaled(641, 479, Some(1280)), (641, 480));
        assert_eq!(scaled(1920, 1081, None), (1920, 1081));
        assert_eq!(scaled(4000, 2, Some(0)), (2, 2));
    }

    #[test]
    fn ffprobe_sits_next_to_ffmpeg() {
        assert_eq!(ffprobe_for("ffmpeg"), PathBuf::from("ffprobe"));
        assert_eq!(ffprobe_for("/opt/ff/ffmpeg"), PathBuf::from("/opt/ff/ffprobe"));
        assert_eq!(ffprobe_for("C:/tools/ffmpeg.exe"), PathBuf::from("C:/tools/ffprobe.exe"));
    }

    #[test]
    fn a_rate_is_a_fraction() {
        assert_eq!(rate("30/1"), Some(30.0));
        assert_eq!(rate("30000/1001").map(|r| (r * 100.0).round()), Some(2997.0));
        assert_eq!(rate("0/0"), None);
        assert_eq!(rate("30"), None);
    }
}
