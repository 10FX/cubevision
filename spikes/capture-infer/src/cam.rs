//! The webcam measured on Windows: what it delivers, alone and with the models behind it

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution};
use serde_json::{Value, json};

use crate::models::Models;
use crate::pipeline::process_frame;
use crate::preprocess::Frame;
use crate::stats::percentile;

pub struct Request {
    pub index: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub seconds: f64,
}

/// What the capture thread shares: the latest frame only, and when each one arrived
#[derive(Default)]
struct Shared {
    latest: Option<Frame>,
    arrivals: Vec<Instant>,
    /// time blocked waiting for the camera, then decoding its MJPEG, per frame
    wait_ms: Vec<f64>,
    decode_ms: Vec<f64>,
    error: Option<String>,
}

const OPENING: Duration = Duration::from_secs(15);

/// The format the camera runs: nokhwa 0.10.11 on Media Foundation reads its frame rate's
/// denominator (30/1 gives 1), so the rate comes from the format it was asked to set
fn actual_format(reported: CameraFormat, set: Option<CameraFormat>) -> CameraFormat {
    match set {
        Some(set) => CameraFormat::new(reported.resolution(), reported.format(), set.frame_rate()),
        None => reported,
    }
}

/// Opens the camera in its own thread and keeps the latest frame, like the live mode; with models,
/// reads each new frame as soon as the previous reading ends (frames that arrive meanwhile are dropped)
pub fn run(request: &Request, mut models: Option<&mut Models>) -> Result<Value> {
    let shared = Arc::new(Mutex::new(Shared::default()));
    let stop = Arc::new(AtomicBool::new(false));
    let (opened, opening) = std::sync::mpsc::channel();
    {
        let (shared, stop) = (shared.clone(), stop.clone());
        let (index, width, height, fps) = (request.index, request.width, request.height, request.fps);
        // Not joined: a camera that stops delivering blocks in frame() and must not hang the measure
        thread::spawn(move || {
            let wanted = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(Resolution::new(width, height), FrameFormat::MJPEG, fps)));
            let camera = Camera::new(CameraIndex::Index(index), wanted)
                .map_err(|e| format!("camera {index} could not be opened: {e}"))
                .and_then(|mut camera| {
                    camera.open_stream().map_err(|e| format!("camera {index} could not start, is another app using it? {e}"))?;
                    Ok(camera)
                });
            let mut camera = match camera {
                Ok(mut camera) => {
                    let set = camera.compatible_camera_formats().ok().and_then(|formats| wanted.fulfill(&formats));
                    let _ = opened.send(Ok((camera.info().human_name(), actual_format(camera.camera_format(), set))));
                    camera
                }
                Err(e) => {
                    let _ = opened.send(Err(e));
                    return;
                }
            };
            while !stop.load(Ordering::Relaxed) {
                let asked = Instant::now();
                let frame = camera.frame().map_err(|e| e.to_string()).and_then(|buffer| {
                    let arrived = Instant::now();
                    let image = buffer.decode_image::<RgbFormat>().map_err(|e| e.to_string())?;
                    Ok((arrived, Frame { width: image.width(), height: image.height(), data: image.into_raw() }))
                });
                let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
                let mut shared = shared.lock().unwrap();
                match frame {
                    Ok((arrived, frame)) => {
                        shared.wait_ms.push((arrived - asked).as_secs_f64() * 1000.0);
                        shared.decode_ms.push(ms(arrived));
                        shared.arrivals.push(arrived);
                        shared.latest = Some(frame);
                    }
                    Err(e) => {
                        shared.error = Some(format!("camera {index} stopped delivering: {e}"));
                        return;
                    }
                }
            }
            let _ = camera.stop_stream();
        });
    }
    let (name, format) = match opening.recv_timeout(OPENING) {
        Ok(result) => result.map_err(|e| anyhow!(e))?,
        Err(RecvTimeoutError::Timeout) => bail!("camera {} did not open within {} s", request.index, OPENING.as_secs()),
        Err(RecvTimeoutError::Disconnected) => bail!("camera {} thread ended while opening", request.index),
    };

    // The measure starts at the first frame: opening and the first exposure are not delivery
    let waiting = Instant::now();
    let start = loop {
        {
            let shared = shared.lock().unwrap();
            if let Some(error) = &shared.error {
                bail!("{error}");
            }
            if let Some(&first) = shared.arrivals.first() {
                break first;
            }
        }
        if waiting.elapsed() > OPENING {
            bail!("camera {} opened but delivered no frame within {} s", request.index, OPENING.as_secs());
        }
        thread::sleep(Duration::from_millis(5));
    };
    let end = start + Duration::from_secs_f64(request.seconds);

    let (mut totals, mut found, mut last) = (Vec::new(), 0usize, None);
    while Instant::now() < end {
        let frame = shared.lock().unwrap().latest.take();
        match (frame, models.as_deref_mut()) {
            (Some(frame), Some(models)) => {
                let t = Instant::now();
                let times = process_frame(models, &frame, &mut last)?;
                totals.push(t.elapsed().as_secs_f64() * 1000.0);
                found += times.found as usize;
            }
            _ => thread::sleep(Duration::from_millis(2)),
        }
    }
    stop.store(true, Ordering::Relaxed);

    let mut shared = shared.lock().unwrap();
    if let Some(error) = &shared.error {
        bail!("{error}");
    }
    let arrivals: Vec<Instant> = shared.arrivals.iter().copied().filter(|&t| t < end).collect();
    let mut gaps: Vec<f64> = arrivals.windows(2).map(|w| (w[1] - w[0]).as_secs_f64() * 1000.0).collect();
    let max_gap = gaps.iter().copied().fold(0.0, f64::max);
    let frames = arrivals.len();
    let inference = models.map(|_| {
        json!({
            "frames_read": totals.len(),
            "found_share": found as f64 / totals.len().max(1) as f64,
            "total_p50_ms": percentile(&mut totals.clone(), 50.0),
            "total_p95_ms": percentile(&mut totals, 95.0),
        })
    });
    Ok(json!({
        "camera": name,
        "requested": { "w": request.width, "h": request.height, "fps": request.fps },
        "actual": { "w": format.width(), "h": format.height(), "fps": format.frame_rate(), "format": format.format().to_string() },
        "seconds": request.seconds,
        "frames": frames,
        "delivered_fps": frames as f64 / request.seconds,
        "delivered_share": frames as f64 / (format.frame_rate() as f64 * request.seconds),
        "frame_gap_ms": { "p50": percentile(&mut gaps, 50.0), "p95": percentile(&mut gaps, 95.0), "max": max_gap },
        "capture_ms": {
            "wait_p50": percentile(&mut shared.wait_ms, 50.0),
            "decode_p50": percentile(&mut shared.decode_ms.clone(), 50.0),
            "decode_p95": percentile(&mut shared.decode_ms, 95.0),
        },
        "inference": inference,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_frame_rate_is_the_one_set_not_its_denominator() {
        let size = Resolution::new(1280, 720);
        let reported = CameraFormat::new(size, FrameFormat::MJPEG, 1); // nokhwa 0.10.11 reads 30/1 as 1
        let set = CameraFormat::new(size, FrameFormat::MJPEG, 30);
        assert_eq!(actual_format(reported, Some(set)).frame_rate(), 30);
        assert_eq!(actual_format(reported, None).frame_rate(), 1);
    }

    #[test]
    fn missing_camera_is_explained() {
        let error = run(&Request { index: 99, width: 1280, height: 720, fps: 30, seconds: 1.0 }, None).unwrap_err();
        assert!(format!("{error:#}").contains("camera"), "{error:#}");
    }
}
