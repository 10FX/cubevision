//! The camera, in a thread of its own that keeps its latest frame only.
//!
//! Windows: Media Foundation through nokhwa, in MJPEG. Linux: ffmpeg reading the V4L2 device. Other systems:
//! not built yet.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
}

impl Camera {
    /// Opens the camera and waits for its first frame
    pub fn open(request: &CameraRequest) -> Result<Camera> {
        let latest = Latest::new();
        let stop = Arc::new(AtomicBool::new(false));
        let format = platform::start(request, latest.clone(), stop.clone())?;
        let camera = Camera { latest, format, stop };
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

    pub fn start(request: &CameraRequest, latest: Latest<Image>, stop: Arc<AtomicBool>) -> Result<CameraFormat> {
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
            Ok(Ok(format)) => Ok(format),
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
    use std::io::Read;
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

    pub fn start(request: &CameraRequest, latest: Latest<Image>, stop: Arc<AtomicBool>) -> Result<CameraFormat> {
        let index = request.index;
        let device = format!("/dev/video{index}");
        if request.width == 0 || request.height == 0 || request.width > 16384 || request.height > 16384 {
            bail!("camera {index}: no camera gives {}×{} frames", request.width, request.height);
        }
        if !Path::new(&device).exists() {
            bail!("camera {index} not found: there is no {device} (under WSL, Windows keeps the cameras)");
        }
        let mut child = match Command::new("ffmpeg")
            .args(arguments(request))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => bail!("camera {index}: cannot run ffmpeg, which reads it ({e}): is ffmpeg installed and in the PATH?"),
        };
        let (width, height) = (request.width as usize, request.height as usize);
        let mut stdout = child.stdout.take().expect("piped");
        let mut stderr = child.stderr.take().expect("piped");
        std::thread::spawn(move || {
            let start = Instant::now();
            let mut data = vec![0; width * height * 4];
            while !stop.load(Ordering::Relaxed) {
                if stdout.read_exact(&mut data).is_err() {
                    let mut said = String::new();
                    let _ = stderr.read_to_string(&mut said);
                    let said = said.lines().last().unwrap_or("no message").trim().to_string();
                    latest.fail(format!("camera {index} stopped delivering, is another app using it? ({said})"));
                    break;
                }
                latest.put(Image { data: data.clone(), width, height, t: start.elapsed().as_secs_f64() * 1000.0 });
            }
            let _ = child.kill();
            let _ = child.wait();
        });
        Ok(CameraFormat { name: device, width, height, fps: request.fps })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

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

    pub fn start(request: &CameraRequest, _latest: Latest<Image>, _stop: Arc<AtomicBool>) -> Result<CameraFormat> {
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
