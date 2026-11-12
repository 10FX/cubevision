//! Where the cube is, frame after frame (`detector.ts`): the detector's box, a second and closer look at a small
//! cube, the box smoothed and kept a few frames when the cube is lost, and the square the reader reads.

use anyhow::Result;

use crate::Frame;
use crate::geometry::{BoxF, reader_crop};

/// The cube found in a frame
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    /// In frame pixels
    pub boxed: BoxF,
    pub score: f32,
}

/// A frame, or a region of it, to look at
#[derive(Clone, Copy, Debug)]
pub struct Look<'a> {
    pub frame: &'a Frame<'a>,
    pub region: Option<BoxF>,
}

/// What finds the cube: the trained detector, or a stand-in in tests
pub trait Detector {
    /// The reader's square: the box's longer side times this
    fn crop_margin(&self) -> f64;
    /// One look at each frame (or region of it), all in one run, without the closer look: the cube when sure enough
    fn look(&mut self, looks: &[Look]) -> Result<Vec<Option<Detection>>>;
}

/// Below this share of the frame's longer side, a cube is looked at a second time, closer
pub const SMALL_CUBE: f64 = 0.25;
/// The closer look: a square of the box's longer side times this, around the box
pub const ZOOM: f64 = 4.0;

/// A cube too small in the frame for the detector to read it well at once
pub fn is_small(frame: &Frame, boxed: &BoxF) -> bool {
    boxed.w.max(boxed.h) < SMALL_CUBE * frame.width.max(frame.height) as f64
}

/// The closer look's square around a small cube
pub fn zoom_region(near: &BoxF) -> BoxF {
    let side = near.w.max(near.h) * ZOOM;
    BoxF { x: near.x + near.w / 2.0 - side / 2.0, y: near.y + near.h / 2.0 - side / 2.0, w: side, h: side }
}

fn one(detector: &mut (impl Detector + ?Sized), frame: &Frame, region: Option<BoxF>) -> Result<Option<Detection>> {
    Ok(detector.look(&[Look { frame, region }])?.pop().flatten())
}

/// One look at the whole frame, and a second, closer one when the cube found there, or the box followed so far
/// (`hint`) if none was, is small in the frame: the detector reads a cube of a few dozen pixels poorly once the
/// frame is shrunk to its input
pub fn detect_with_zoom(detector: &mut (impl Detector + ?Sized), frame: &Frame, hint: Option<&BoxF>) -> Result<Option<Detection>> {
    let found = one(detector, frame, None)?;
    match found.map(|d| d.boxed).or(hint.copied()) {
        Some(near) if is_small(frame, &near) => Ok(one(detector, frame, Some(zoom_region(&near)))?.or(found)),
        _ => Ok(found),
    }
}

fn iou(a: &BoxF, b: &BoxF) -> f64 {
    let ix = ((a.x + a.w).min(b.x + b.w) - a.x.max(b.x)).max(0.0);
    let iy = ((a.y + a.h).min(b.y + b.h) - a.y.max(b.y)).max(0.0);
    let union = a.w * a.h + b.w * b.h - ix * iy;
    (ix * iy) / if union == 0.0 { 1.0 } else { union }
}

/// The box followed from frame to frame: smoothed, kept a few frames when the cube is lost
#[derive(Clone, Debug)]
pub struct BoxTracker {
    boxed: Option<BoxF>,
    missed: u32,
    keep: u32,
    follow: f64,
}

impl Default for BoxTracker {
    fn default() -> Self {
        BoxTracker::new(8, 0.6)
    }
}

impl BoxTracker {
    /// `keep`: frames the box is kept without a cube; `follow`: share of the way to the new box at each frame
    pub fn new(keep: u32, follow: f64) -> Self {
        BoxTracker { boxed: None, missed: 0, keep, follow }
    }

    /// The box to read in this frame, given where the cube was found in it. A box no frame can hold (not
    /// finite, empty) counts as no cube: the site would follow it and never come back.
    pub fn update(&mut self, found: Option<BoxF>) -> Option<BoxF> {
        let Some(found) = found.filter(BoxF::is_real) else {
            if self.boxed.is_some() {
                self.missed += 1;
                if self.missed <= self.keep {
                    return self.boxed;
                }
            }
            self.boxed = None;
            return None;
        };
        self.missed = 0;
        // A jump (another place, a new cube): follow it at once; else smooth the jitter
        self.boxed = Some(match self.boxed {
            Some(last) if iou(&last, &found) >= 0.3 => {
                let ease = |from: f64, to: f64| from + (to - from) * self.follow;
                BoxF { x: ease(last.x, found.x), y: ease(last.y, found.y), w: ease(last.w, found.w), h: ease(last.h, found.h) }
            }
            _ => found,
        });
        self.boxed
    }

    /// The box followed so far (a hint for the next frame)
    pub fn current(&self) -> Option<BoxF> {
        self.boxed
    }

    pub fn reset(&mut self) {
        self.boxed = None;
        self.missed = 0;
    }
}

/// The square the reader reads, frame after frame: the detector looks at every frame until the reader reads the
/// cube, then only every `every` frames while it goes on reading it in the box followed (the cube moves little
/// from one frame to the next, and the detector costs as much as the reader)
pub struct CubeFollower<D: Detector> {
    pub boxes: BoxTracker,
    detector: D,
    every: u32,
    since: u32,
    reading: bool,
}

impl<D: Detector> CubeFollower<D> {
    /// `every`: 3 live, as the site; 1 for a video, where every frame is looked at
    pub fn new(detector: D, every: u32) -> Self {
        CubeFollower { boxes: BoxTracker::default(), detector, every, since: 0, reading: false }
    }

    pub fn detector(&self) -> &D {
        &self.detector
    }

    pub fn detector_mut(&mut self) -> &mut D {
        &mut self.detector
    }

    fn square(&self, boxed: Option<BoxF>) -> Option<BoxF> {
        boxed.map(|b| reader_crop(&b, self.detector.crop_margin()))
    }

    /// The reader's square in this frame (None: no cube)
    pub fn crop(&mut self, frame: &Frame) -> Result<Option<BoxF>> {
        let followed = self.reading && self.boxes.current().is_some() && {
            self.since += 1;
            self.since < self.every
        };
        let found = if followed {
            self.boxes.current()
        } else {
            self.since = 0;
            let hint = self.boxes.current();
            let detection = detect_with_zoom(&mut self.detector, frame, hint.as_ref())?;
            self.boxes.update(detection.map(|d| d.boxed))
        };
        self.reading = false;
        Ok(self.square(found))
    }

    /// The reader's squares in several frames in a row, as `crop` frame after frame would find them, the
    /// detector run on them all at once: every frame looked at (`every` 1), then the small cubes closer, at
    /// once too, but where the cube was lost (the closer look then goes where the box followed so far is)
    pub fn crop_batch(&mut self, frames: &[Frame]) -> Result<Vec<Option<BoxF>>> {
        if self.every != 1 {
            return frames.iter().map(|frame| self.crop(frame)).collect();
        }
        let whole: Vec<Look> = frames.iter().map(|frame| Look { frame, region: None }).collect();
        let found = if whole.is_empty() { Vec::new() } else { self.detector.look(&whole)? };
        let ahead: Vec<Look> = frames
            .iter()
            .zip(&found)
            .filter_map(|(frame, d)| d.filter(|d| is_small(frame, &d.boxed)).map(|d| Look { frame, region: Some(zoom_region(&d.boxed)) }))
            .collect();
        let mut closer = if ahead.is_empty() { Vec::new() } else { self.detector.look(&ahead)? }.into_iter();
        let mut out = Vec::with_capacity(frames.len());
        for (frame, first) in frames.iter().zip(found) {
            let detection = match first {
                Some(d) if is_small(frame, &d.boxed) => closer.next().flatten().or(first),
                Some(_) => first,
                None => match self.boxes.current() {
                    Some(hint) if is_small(frame, &hint) => one(&mut self.detector, frame, Some(zoom_region(&hint)))?,
                    _ => None,
                },
            };
            let boxed = self.boxes.update(detection.map(|d| d.boxed));
            out.push(self.square(boxed));
        }
        self.since = 0;
        self.reading = false;
        Ok(out)
    }

    /// Whether the reader read the cube in the last square (a layer mid-turn counts: the cube is there)
    pub fn read(&mut self, cube: bool) {
        self.reading = cube;
    }
}
