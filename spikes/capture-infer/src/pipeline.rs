//! One frame as the site reads it: letterbox, detector, the reader's square around the cube, reader

use std::time::Instant;

use anyhow::Result;
use rayon::prelude::*;

use crate::models::Models;
use crate::models::Detection;
use crate::preprocess::{BoxF, Frame, Letterbox, letterbox, letterbox_region, reader_crop, square_crop, unletterbox};

pub const DETECTOR_SIZE: u32 = 320;
pub const READER_SIZE: u32 = 224;
pub const LETTERBOX_PAD: f32 = 0.5;
pub const THRESHOLD: f32 = 0.5;
pub const CROP_MARGIN: f32 = 1.5;
/// A cube whose longer side is under this share of the frame's is looked at again, closer
pub const SMALL_CUBE: f32 = 0.25;
/// The closer look: a square of the box's longer side times this
pub const ZOOM: f32 = 4.0;

pub fn is_small(width: u32, height: u32, boxed: &BoxF) -> bool {
    boxed.w.max(boxed.h) < SMALL_CUBE * width.max(height) as f32
}

pub fn zoom_region(near: &BoxF) -> BoxF {
    let side = near.w.max(near.h) * ZOOM;
    BoxF { x: near.x + near.w / 2.0 - side / 2.0, y: near.y + near.h / 2.0 - side / 2.0, w: side, h: side }
}

/// The detector's box in frame pixels, when it is sure enough
fn found(detection: Detection, lb: &Letterbox) -> Option<BoxF> {
    (detection.score >= THRESHOLD).then(|| unletterbox(detection.boxed, DETECTOR_SIZE, lb))
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameTimes {
    pub letterbox_ms: f64,
    pub detector_ms: f64,
    pub crop_ms: f64,
    pub reader_ms: f64,
    pub found: bool,
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// A cube looked for as the site does; without one found, the reader reads around the last box, or the frame's central square at first
pub fn process_frame(models: &mut Models, frame: &Frame, last: &mut Option<BoxF>) -> Result<FrameTimes> {
    let mut times = FrameTimes::default();
    // The site's detectWithZoom: the whole frame, then closer when the cube (or the last box) is small
    let t = Instant::now();
    let (square, lb) = letterbox(frame, DETECTOR_SIZE, LETTERBOX_PAD);
    times.letterbox_ms = ms(t);
    let t = Instant::now();
    let mut boxed = found(models.detect(&square)?, &lb);
    times.detector_ms = ms(t);
    if let Some(near) = boxed.or(*last).filter(|near| is_small(frame.width, frame.height, near)) {
        let t = Instant::now();
        let (square, lb) = letterbox_region(frame, DETECTOR_SIZE, LETTERBOX_PAD, &zoom_region(&near));
        times.letterbox_ms += ms(t);
        let t = Instant::now();
        boxed = found(models.detect(&square)?, &lb).or(boxed);
        times.detector_ms += ms(t);
    }
    if boxed.is_some() {
        *last = boxed;
        times.found = true;
    }

    let t = Instant::now();
    let crop = square_crop(frame, &reading_region(frame, last), READER_SIZE);
    times.crop_ms = ms(t);

    let t = Instant::now();
    models.read(&crop, 1)?;
    times.reader_ms = ms(t);
    Ok(times)
}

/// The reader's square: around the last box, or the frame's central square
fn reading_region(frame: &Frame, last: &Option<BoxF>) -> BoxF {
    match last {
        Some(boxed) => reader_crop(boxed, CROP_MARGIN),
        None => {
            let side = frame.width.min(frame.height) as f32;
            BoxF { x: (frame.width as f32 - side) / 2.0, y: (frame.height as f32 - side) / 2.0, w: side, h: side }
        }
    }
}

/// Batches run padded to this size. A CUDA session that changes input shape from one run to the next
/// pays about 100 ms; one shape costs nothing (measured on the webm: 107 ms against 12 ms a look).
pub const BATCH: usize = 8;

fn padded(n: usize) -> usize {
    n.div_ceil(BATCH) * BATCH
}

/// Detector looks at several regions, letterboxed in parallel and run as one batch
fn look(models: &mut Models, looks: &[(&Frame, Option<BoxF>)], times: &mut FrameTimes) -> Result<Vec<Option<BoxF>>> {
    if looks.is_empty() {
        return Ok(Vec::new());
    }
    let t = Instant::now();
    let boxed: Vec<(Vec<f32>, Letterbox)> = looks
        .par_iter()
        .map(|(frame, region)| match region {
            Some(region) => letterbox_region(frame, DETECTOR_SIZE, LETTERBOX_PAD, region),
            None => letterbox(frame, DETECTOR_SIZE, LETTERBOX_PAD),
        })
        .collect();
    let mut tensors: Vec<f32> = boxed.iter().flat_map(|(tensor, _)| tensor.iter().copied()).collect();
    let plane = (3 * DETECTOR_SIZE * DETECTOR_SIZE) as usize;
    tensors.resize(padded(looks.len()) * plane, LETTERBOX_PAD);
    times.letterbox_ms += ms(t);
    let t = Instant::now();
    let detections = models.detect_batch(&tensors, padded(looks.len()))?;
    times.detector_ms += ms(t);
    Ok(detections.into_iter().zip(&boxed).map(|(d, (_, lb))| found(d, lb)).collect())
}

/// Frames read together, with the same outcome as `process_frame` one after the other: whole-frame
/// looks in one batch, closer looks in batches, crops in parallel, one reader batch. The times are
/// the batch's, shared evenly among its frames.
pub fn process_batch(models: &mut Models, frames: &[Frame], last: &mut Option<BoxF>) -> Result<Vec<FrameTimes>> {
    let n = frames.len();
    let mut times = FrameTimes::default();
    let first = look(models, &frames.iter().map(|f| (f, None)).collect::<Vec<_>>(), &mut times)?;

    // Closer looks around a small cube found: they depend on nothing before
    let small = |i: usize, near: &BoxF| is_small(frames[i].width, frames[i].height, near);
    let around_found: Vec<usize> = (0..n).filter(|&i| first[i].is_some_and(|b| small(i, &b))).collect();
    let zoomed = look(models, &around_found.iter().map(|&i| (&frames[i], Some(zoom_region(&first[i].unwrap())))).collect::<Vec<_>>(), &mut times)?;
    let mut finals: Vec<Option<BoxF>> = first.clone();
    for (&i, z) in around_found.iter().zip(zoomed) {
        finals[i] = z.or(first[i]);
    }

    // Frames where nothing was found look around the last box, which a closer look may change:
    // the following misses are looked at together, as if none changed it, and redone after one that does
    let mut current = *last;
    let mut i = 0;
    while i < n {
        if first[i].is_some() {
            current = finals[i];
            i += 1;
            continue;
        }
        let Some(near) = current.filter(|near| small(i, near)) else {
            i += 1;
            continue;
        };
        let run: Vec<usize> = (i..n).take_while(|&j| first[j].is_none()).collect();
        let region = zoom_region(&near);
        let results = look(models, &run.iter().map(|&j| (&frames[j], Some(region))).collect::<Vec<_>>(), &mut times)?;
        for (&j, result) in run.iter().zip(results) {
            i = j + 1;
            if result.is_some() {
                finals[j] = result;
                current = result;
                break;
            }
        }
    }

    let mut regions = Vec::with_capacity(n);
    let mut found = Vec::with_capacity(n);
    for (frame, boxed) in frames.iter().zip(&finals) {
        if boxed.is_some() {
            *last = *boxed;
        }
        found.push(boxed.is_some());
        regions.push(reading_region(frame, last));
    }
    let t = Instant::now();
    let mut crops: Vec<f32> = frames.par_iter().zip(&regions).flat_map_iter(|(frame, region)| square_crop(frame, region, READER_SIZE)).collect();
    crops.resize(padded(n) * (3 * READER_SIZE * READER_SIZE) as usize, 0.0);
    times.crop_ms += ms(t);
    let t = Instant::now();
    models.read(&crops, padded(n))?;
    times.reader_ms += ms(t);

    let share = |v: f64| v / n as f64;
    Ok(found
        .into_iter()
        .map(|found| FrameTimes { letterbox_ms: share(times.letterbox_ms), detector_ms: share(times.detector_ms), crop_ms: share(times.crop_ms), reader_ms: share(times.reader_ms), found })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_are_always_a_multiple_of_eight() {
        assert_eq!([0, 1, 2, 8, 9].map(padded), [0, 8, 8, 8, 16]);
    }

    #[test]
    fn a_cube_under_a_quarter_of_the_frame_is_small() {
        assert!(is_small(1920, 1080, &BoxF { x: 0., y: 0., w: 470., h: 100. }));
        assert!(!is_small(1920, 1080, &BoxF { x: 0., y: 0., w: 100., h: 480. }));
    }

    #[test]
    fn the_zoom_is_four_times_the_box_around_its_centre() {
        let z = zoom_region(&BoxF { x: 100., y: 100., w: 50., h: 40. });
        assert_eq!(z, BoxF { x: 25., y: 20., w: 200., h: 200. });
    }
}
