//! The camera, in a thread of its own that keeps its latest frame only.
//!
//! Windows: Media Foundation through nokhwa, in MJPEG. Linux: ffmpeg reading the V4L2 device. Other systems:
//! not built yet.

use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use crate::{Image, Latest};

/// The camera asked for; it gives the closest format it has
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraRequest {
    /// Which camera: 0 is the system's first
    pub index: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Default for CameraRequest {
    fn default() -> Self {
        CameraRequest { index: 0, width: 1280, height: 720, fps: 30 }
    }
}

/// What the camera opened gives
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraFormat {
    pub name: String,
    pub width: usize,
    pub height: usize,
    /// Frames a second it says it gives; in poor light many cameras give fewer
    pub fps: u32,
}

/// How long a camera may take to open and give its first frame
const OPENING: Duration = Duration::from_secs(15);

pub struct Camera {
    latest: Latest<Image>,
    format: CameraFormat,
    stop: Arc<AtomicBool>,
    /// The program that reads the camera, where one does (ffmpeg on Linux): ended with the camera
    child: Option<Arc<Mutex<Child>>>,
}

impl Camera {
    /// Opens the camera and waits for its first frame
    pub fn open(request: &CameraRequest) -> Result<Camera> {
        let latest = Latest::new();
        let stop = Arc::new(AtomicBool::new(false));
        let (format, child) = platform::start(request, latest.clone(), stop.clone())?;
        // From here on, leaving (an error included) lets the camera go
        let camera = Camera { latest, format, stop, child };
        // The first frame proves the camera delivers: opening alone does not (another app may hold it)
        let waiting = Instant::now();
        while camera.latest.dropped() == 0 && !camera.latest.has_value() {
            if let Some(error) = camera.latest.error() {
                bail!("{error}");
            }
            if waiting.elapsed() > OPENING {
                bail!("camera {} opened but gave no frame within {} s", request.index, OPENING.as_secs());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(camera)
    }

    pub fn format(&self) -> &CameraFormat {
        &self.format
    }

    fn checked(&self, image: Option<Image>) -> Result<Option<Image>> {
        match (image, self.latest.error()) {
            (None, Some(error)) => bail!("{error}"),
            (image, _) => Ok(image),
        }
    }

    /// The most recent frame, if a new one came since the last call (those in between are dropped); an error
    /// once the camera stopped
    pub fn latest(&self) -> Result<Option<Image>> {
        let image = self.latest.take();
        self.checked(image)
    }

    /// The most recent frame as soon as there is a new one, waiting at most `timeout`
    pub fn wait(&self, timeout: Duration) -> Result<Option<Image>> {
        let image = self.latest.wait(timeout);
        self.checked(image)
    }

    /// Frames the camera gave that were never read: the reader was slower than the camera
    pub fn dropped(&self) -> u64 {
        self.latest.dropped()
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Its thread may be blocked waiting for a frame that never comes: the reader is ended from here
        if let Some(child) = &self.child {
            let mut child = child.lock().unwrap();
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// RGB bytes as RGBA, opaque
#[cfg_attr(not(windows), allow(dead_code))]
fn rgba(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.chunks_exact(3) {
        out.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    out
}

#[cfg(windows)]
mod platform {
    use std::sync::mpsc::RecvTimeoutError;

    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraFormat as Format, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};

    use super::*;

    /// The format the camera runs: nokhwa 0.10.11 on Media Foundation reads its frame rate's denominator (30/1
    /// gives 1), so the rate comes from the format it was asked to set
    fn rate(reported: Format, set: Option<Format>) -> u32 {
        set.map_or(reported.frame_rate(), |set| set.frame_rate())
    }

    pub fn start(
        request: &CameraRequest,
        latest: Latest<Image>,
        stop: Arc<AtomicBool>,
    ) -> Result<(CameraFormat, Option<Arc<Mutex<Child>>>)> {
        let (opened, opening) = std::sync::mpsc::channel();
        let (index, width, height, fps) = (request.index, request.width, request.height, request.fps);
        // Not joined: a camera that stops delivering blocks in frame() and must not hang the program
        std::thread::spawn(move || {
            let wanted = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(Format::new(
                Resolution::new(width, height),
                FrameFormat::MJPEG,
                fps,
            )));
            let camera = nokhwa::Camera::new(CameraIndex::Index(index), wanted)
                .map_err(|e| format!("camera {index} could not be opened: {e}"))
                .and_then(|mut camera| {
                    camera.open_stream().map_err(|e| format!("camera {index} could not start, is another app using it? {e}"))?;
                    Ok(camera)
                });
            let mut camera = match camera {
                Ok(mut camera) => {
                    let set = camera.compatible_camera_formats().ok().and_then(|formats| wanted.fulfill(&formats));
                    let format = camera.camera_format();
                    let _ = opened.send(Ok(CameraFormat {
                        name: camera.info().human_name(),
                        width: format.width() as usize,
                        height: format.height() as usize,
                        fps: rate(format, set),
                    }));
                    camera
                }
                Err(e) => {
                    let _ = opened.send(Err(e));
                    return;
                }
            };
            let start = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                let frame = camera.frame().map_err(|e| e.to_string()).and_then(|buffer| {
                    let t = start.elapsed().as_secs_f64() * 1000.0;
                    let image = buffer.decode_image::<RgbFormat>().map_err(|e| e.to_string())?;
                    Ok(Image { width: image.width() as usize, height: image.height() as usize, data: rgba(image.as_raw()), t })
                });
                match frame {
                    Ok(image) => latest.put(image),
                    Err(e) => {
                        latest.fail(format!("camera {index} stopped delivering: {e}"));
                        return;
                    }
                }
            }
            let _ = camera.stop_stream();
        });
        match opening.recv_timeout(OPENING) {
            Ok(Ok(format)) => Ok((format, None)),
            Ok(Err(e)) => bail!("{e}"),
            Err(RecvTimeoutError::Timeout) => bail!("camera {} did not open within {} s", request.index, OPENING.as_secs()),
            Err(RecvTimeoutError::Disconnected) => bail!("camera {} could not be opened", request.index),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_frame_rate_is_the_one_set_not_its_denominator() {
            let size = Resolution::new(1280, 720);
            let reported = Format::new(size, FrameFormat::MJPEG, 1); // nokhwa 0.10.11 reads 30/1 as 1
            assert_eq!(rate(reported, Some(Format::new(size, FrameFormat::MJPEG, 30))), 30);
            assert_eq!(rate(reported, None), 1);
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::io::{BufRead, BufReader, Read};
    use std::path::Path;
    use std::process::{Command, Stdio};

    use super::*;

    /// What ffmpeg is asked: the V4L2 device in MJPEG at the size and rate wanted, its frames as raw RGBA of
    /// exactly that size (a device that gives another size is scaled to it)
    pub(super) fn arguments(request: &CameraRequest) -> Vec<String> {
        let CameraRequest { index, width, height, fps } = request;
        [
            "-nostdin",
            "-v",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &fps.to_string(),
            "-i",
            &format!("/dev/video{index}"),
            "-vf",
            &format!("scale={width}:{height}"),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ]
        .map(String::from)
        .to_vec()
    }

    /// The program that reads the camera, started: its frames (raw RGBA of the size given, on its output) kept
    /// as they come, its messages read as they come too (a pipe left full would stop it), the last ones kept to
    /// say why it ended
    pub(super) fn pump(
        mut command: Command,
        index: u32,
        width: usize,
        height: usize,
        latest: Latest<Image>,
        stop: Arc<AtomicBool>,
    ) -> Result<Arc<Mutex<Child>>> {
        let mut child = match command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
            Ok(child) => child,
            Err(e) => bail!("camera {index}: cannot run ffmpeg, which reads it ({e}): is ffmpeg installed and in the PATH?"),
        };
        let mut stdout = child.stdout.take().expect("piped");
        let stderr = child.stderr.take().expect("piped");
        let child = Arc::new(Mutex::new(child));
        let said = Arc::new(Mutex::new(String::new()));
        let noted = said.clone();
        let messages = std::thread::spawn(move || {
            for line in BufReader::new(stderr).split(b'\n').map_while(|line| line.ok()) {
                let line = String::from_utf8_lossy(&line).trim().chars().take(300).collect::<String>();
                if !line.is_empty() {
                    *noted.lock().unwrap() = line;
                }
            }
        });
        std::thread::spawn(move || {
            let start = Instant::now();
            let mut data = vec![0; width * height * 4];
            while !stop.load(Ordering::Relaxed) {
                if stdout.read_exact(&mut data).is_err() {
                    // Its last words are read to the end before they are quoted
                    let _ = messages.join();
                    let said = said.lock().unwrap();
                    let said = if said.is_empty() { "no message" } else { said.as_str() };
                    latest.fail(format!("camera {index} stopped delivering, is another app using it? ({said})"));
                    break;
                }
                latest.put(Image { data: data.clone(), width, height, t: start.elapsed().as_secs_f64() * 1000.0 });
            }
        });
        Ok(child)
    }

    pub fn start(
        request: &CameraRequest,
        latest: Latest<Image>,
        stop: Arc<AtomicBool>,
    ) -> Result<(CameraFormat, Option<Arc<Mutex<Child>>>)> {
        let index = request.index;
        let device = format!("/dev/video{index}");
        if request.width == 0 || request.height == 0 || request.width > 16384 || request.height > 16384 {
            bail!("camera {index}: no camera gives {}×{} frames", request.width, request.height);
        }
        if !Path::new(&device).exists() {
            bail!("camera {index} not found: there is no {device} (under WSL, Windows keeps the cameras)");
        }
        let mut command = Command::new("ffmpeg");
        command.args(arguments(request));
        let (width, height) = (request.width as usize, request.height as usize);
        let child = pump(command, index, width, height, latest, stop)?;
        Ok((CameraFormat { name: device, width, height, fps: request.fps }, Some(child)))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn shell(script: &str) -> Command {
            let mut command = Command::new("sh");
            command.args(["-c", script]);
            command
        }

        fn alive(pid: u32) -> bool {
            Path::new(&format!("/proc/{pid}/stat")).exists()
                && !std::fs::read_to_string(format!("/proc/{pid}/stat"))
                    .is_ok_and(|s| s.rsplit(") ").next().is_some_and(|rest| rest.starts_with('Z')))
        }

        #[test]
        fn a_talkative_reader_does_not_stop_the_frames() {
            // More messages than a pipe holds, then a frame: the messages must be read meanwhile
            let (latest, stop) = (Latest::new(), Arc::new(AtomicBool::new(false)));
            let source = shell("head -c 300000 /dev/zero | tr '\\0' 'x' >&2; head -c 64 /dev/zero; sleep 30");
            let child = pump(source, 0, 4, 4, latest.clone(), stop).unwrap();
            let image = latest.wait(Duration::from_secs(10)).expect("the frame comes");
            assert_eq!((image.width, image.height, image.data.len()), (4, 4, 64));
            let _ = child.lock().unwrap().kill();
        }

        #[test]
        fn a_camera_let_go_stops_its_reader() {
            // No frame ever: the thread is blocked reading, the camera is dropped all the same
            let (latest, stop) = (Latest::new(), Arc::new(AtomicBool::new(false)));
            let child = pump(shell("exec sleep 30"), 0, 4, 4, latest.clone(), stop.clone()).unwrap();
            let pid = child.lock().unwrap().id();
            assert!(alive(pid));
            drop(Camera { latest, format: CameraFormat { name: "test".into(), width: 4, height: 4, fps: 30 }, stop, child: Some(child) });
            let gone = Instant::now();
            while alive(pid) && gone.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(!alive(pid), "the reader still runs");
        }

        #[test]
        fn a_reader_that_ends_says_its_last_words() {
            let (latest, stop) = (Latest::<Image>::new(), Arc::new(AtomicBool::new(false)));
            let _child =
                pump(shell("echo 'first' >&2; echo 'Device or resource busy' >&2; exit 1"), 3, 4, 4, latest.clone(), stop).unwrap();
            assert!(latest.wait(Duration::from_secs(10)).is_none());
            let error = latest.error().expect("it failed");
            assert!(error.contains("camera 3") && error.contains("Device or resource busy"), "{error}");
        }

        #[test]
        fn ffmpeg_is_asked_for_the_device_in_the_size_wanted() {
            let asked = arguments(&CameraRequest { index: 2, width: 640, height: 480, fps: 60 }).join(" ");
            assert!(asked.contains("-f v4l2") && asked.contains("-video_size 640x480") && asked.contains("-framerate 60"), "{asked}");
            assert!(asked.contains("-i /dev/video2") && asked.contains("scale=640:480") && asked.ends_with("-pix_fmt rgba -"), "{asked}");
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod platform {
    use super::*;

    pub fn start(
        request: &CameraRequest,
        _latest: Latest<Image>,
        _stop: Arc<AtomicBool>,
    ) -> Result<(CameraFormat, Option<Arc<Mutex<Child>>>)> {
        bail!("camera {}: reading a camera is not built for this system yet (Windows and Linux are)", request.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_becomes_opaque_rgba() {
        assert_eq!(rgba(&[1, 2, 3, 4, 5, 6]), [1, 2, 3, 255, 4, 5, 6, 255]);
    }
}
