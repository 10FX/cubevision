//! Frames read one after the other, as ChronoCube reads a video: the cube found and cropped, the crop read
//! (flipped too while it is not known whether the video is mirrored), how much it moved since the frame before.
//!
//! Two ways, with the same outcome on the same boxes: in batches (`push_batch`, the site's video worker: the
//! detector looks at every frame), or frame by frame (`push`, the site's evaluation and its live mode: once
//! the reader reads the cube, the detector looks every few frames only).

use std::collections::HashMap;

use anyhow::Result;
use cube_capture::Image;
use cube_decode::mirror::{MirrorCheck, flip_frame};
use cube_decode::motion::MotionMeter;
use cube_decode::pipeline::MOTION;
use cube_decode::{FaceReading, FrameReadings, Motion};
use cube_infer::follow::{CubeFollower, Detector};
use cube_infer::geometry::square_crop;
use cube_infer::output::{Reader, ReaderOutput, TURNING, readings_from_output};
use cube_infer::{BoxF, Frame};
use rayon::prelude::*;

/// What a video's frames gave
#[derive(Clone, Debug, PartialEq)]
pub struct Read {
    pub frames: Vec<FrameReadings>,
    /// The video was filmed mirrored (a selfie camera): its frames were read flipped
    pub mirrored: bool,
    /// Frames where the cube was found
    pub found: usize,
}

/// The last frame read, for whoever shows it
#[derive(Clone, Debug, PartialEq)]
pub struct Seen {
    /// The reader's square in the frame (None: no cube)
    pub boxed: Option<BoxF>,
    pub faces: Vec<FaceReading>,
    /// The reader says a layer is mid-turn, and the crop did move
    pub turning: bool,
}

pub struct FrameLoop<D: Detector, R: Reader> {
    follower: CubeFollower<D>,
    reader: R,
    mirror: MirrorCheck,
    /// The first frames flipped, while it is not known whether the video is mirrored: the readings kept if it is
    flipped: HashMap<usize, FrameReadings>,
    meter: MotionMeter,
    read: Vec<FrameReadings>,
    found: usize,
}

fn view(image: &Image) -> Frame<'_> {
    Frame { data: &image.data, width: image.width, height: image.height }
}

impl<D: Detector, R: Reader> FrameLoop<D, R> {
    /// `every`: the detector looks at one frame in so many while the reader reads the cube (`push`); 1 for a
    /// video read in batches. `hold`: how the cube was held to scramble, when known.
    pub fn new(detector: D, reader: R, scramble: &str, hold: Option<&str>, every: u32) -> Self {
        FrameLoop {
            follower: CubeFollower::new(detector, every),
            reader,
            mirror: MirrorCheck::new(scramble, hold),
            flipped: HashMap::new(),
            meter: MotionMeter::default(),
            read: Vec::new(),
            found: 0,
        }
    }

    pub fn reader(&self) -> &R {
        &self.reader
    }

    pub fn frames(&self) -> &[FrameReadings] {
        &self.read
    }

    fn crop(&self, image: &Image, boxed: &Option<BoxF>) -> Option<Vec<u8>> {
        boxed.as_ref().map(|b| square_crop(&view(image), b, self.reader.size()))
    }

    /// One frame's outputs taken in: what the frame reads, and the flipped reading while the mirror is in doubt
    fn take(&mut self, t: f64, crop: Option<&[u8]>, out: Option<&ReaderOutput>, other: Option<&ReaderOutput>) -> (Vec<FaceReading>, bool) {
        let size = self.reader.size();
        let motion = self.meter.measure(crop.map(|data| Frame { data, width: size, height: size }).as_ref(), t);
        let still = motion.is_some_and(|m| m <= MOTION);
        let ordered = self.reader.ordered();
        let faces = out.map(|out| readings_from_output(out, ordered, still)).unwrap_or_default();
        let motion = motion.map_or(Motion::Unknown, Motion::Change);
        let reading = |faces: &[FaceReading], out: Option<&ReaderOutput>| FrameReadings {
            t,
            faces: faces.to_vec(),
            orientation: out.and_then(ReaderOutput::orientation64),
            turning: out.map(|o| o.turning as f64),
            motion,
        };
        self.read.push(reading(&faces, out));
        self.found += crop.is_some() as usize;
        if let (Some(other), None) = (other, self.mirror.mirrored) {
            let other_faces = readings_from_output(other, ordered, still);
            self.flipped.insert(self.read.len() - 1, reading(&other_faces, Some(other)));
            if self.mirror.add(&faces, &other_faces) == Some(true) {
                for (i, flipped) in self.flipped.drain() {
                    self.read[i] = flipped;
                }
            }
        }
        (faces, out.is_some_and(|o| o.turning > TURNING) && !still)
    }

    /// Frames read together: the cube found in them all at once, then all the crops read at once
    pub fn push_batch(&mut self, images: &[Image]) -> Result<Vec<Seen>> {
        let views: Vec<Frame> = images.iter().map(view).collect();
        let boxes = self.follower.crop_batch(&views)?;
        // The cube, cropped at full resolution: its seams survive
        let size = self.reader.size();
        let crops: Vec<Option<Vec<u8>>> =
            views.par_iter().zip(&boxes).map(|(frame, boxed)| boxed.as_ref().map(|b| square_crop(frame, b, size))).collect();
        // While it is not known whether the video is mirrored, each crop is read flipped too
        let unsure = self.mirror.mirrored.is_none();
        let mirrored = self.mirror.mirrored == Some(true);
        let flip = |crop: &Vec<u8>| flip_frame(&Frame { data: crop, width: size, height: size });
        let shown: Vec<&Vec<u8>> = crops.iter().flatten().collect();
        let mut inputs: Vec<Vec<u8>> = Vec::new();
        if mirrored || unsure {
            inputs = shown.par_iter().map(|crop| flip(crop)).collect();
        }
        let straight: Vec<&[u8]> =
            if mirrored { inputs.iter().map(Vec::as_slice).collect() } else { shown.iter().map(|c| c.as_slice()).collect() };
        let both: Vec<&[u8]> = if unsure { straight.iter().copied().chain(inputs.iter().map(Vec::as_slice)).collect() } else { straight };
        let outs = self.reader.read(&both)?;
        let mut seen = Vec::with_capacity(images.len());
        let mut k = 0;
        for ((image, crop), boxed) in images.iter().zip(&crops).zip(boxes) {
            let at = crop.as_ref().map(|_| {
                k += 1;
                k - 1
            });
            let straight = at.map(|at| &outs[at]);
            let other = at.filter(|_| unsure).map(|at| &outs[shown.len() + at]);
            // Found mirrored within this batch: the flipped reading is the one
            let out = if unsure && self.mirror.mirrored == Some(true) { other } else { straight };
            let (faces, turning) = self.take(image.t, crop.as_deref(), out, other);
            seen.push(Seen { boxed, faces, turning });
        }
        Ok(seen)
    }

    /// One frame read on its own
    pub fn push(&mut self, image: &Image) -> Result<Seen> {
        let boxed = self.follower.crop(&view(image))?;
        let crop = self.crop(image, &boxed);
        let size = self.reader.size();
        let flip = |crop: &Vec<u8>| flip_frame(&Frame { data: crop, width: size, height: size });
        let out = match &crop {
            Some(crop) if self.mirror.mirrored == Some(true) => Some(self.reader.read(&[&flip(crop)])?.remove(0)),
            Some(crop) => Some(self.reader.read(&[crop])?.remove(0)),
            None => None,
        };
        let other = match &crop {
            Some(crop) if self.mirror.mirrored.is_none() => Some(self.reader.read(&[&flip(crop)])?.remove(0)),
            _ => None,
        };
        let (faces, turning) = self.take(image.t, crop.as_deref(), out.as_ref(), other.as_ref());
        if let Some(out) = &out {
            // A layer mid-turn counts: the cube is there
            self.follower.read(!faces.is_empty() || out.turning > TURNING);
        }
        Ok(Seen { boxed, faces, turning })
    }

    pub fn finish(self) -> Read {
        Read { frames: self.read, mirrored: self.mirror.mirrored == Some(true), found: self.found }
    }
}
