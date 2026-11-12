//! The frame loop with stand-ins for the two models

use anyhow::Result;
use cube_capture::Image;
use cube_core::Cube;
use cube_decode::Motion;
use cube_infer::follow::{Detection, Detector, Look};
use cube_infer::output::{Reader, ReaderOutput};
use cube_infer::{BoxF, Frame};
use cubevision::read::FrameLoop;

const SCRAMBLE: &str = "R U2 F' L D2 B R' U F2 L'";

/// Finds a cube in the middle of every frame whose first byte is not 0
struct Middle;

impl Detector for Middle {
    fn crop_margin(&self) -> f64 {
        1.5
    }

    fn look(&mut self, looks: &[Look]) -> Result<Vec<Option<Detection>>> {
        Ok(looks
            .iter()
            .map(|look| {
                let Frame { data, width, height } = *look.frame;
                (data[0] != 0).then(|| Detection {
                    boxed: BoxF { x: width as f64 / 4.0, y: height as f64 / 4.0, w: width as f64 / 2.0, h: height as f64 / 2.0 },
                    score: 0.9,
                })
            })
            .collect())
    }
}

/// Reads the faces U, R and F of the scrambled cube, surely: as they are when the crop's brighter half is where
/// the camera put it (the left; the right when the video is mirrored, `mirror`), as a mirror shows them
/// (columns swapped) otherwise.
struct Sure {
    mirror: bool,
    reads: usize,
}

impl Reader for Sure {
    fn size(&self) -> usize {
        16
    }

    fn ordered(&self) -> bool {
        true
    }

    fn read(&mut self, crops: &[&[u8]]) -> Result<Vec<ReaderOutput>> {
        self.reads += crops.len();
        let state = Cube::solved().apply(SCRAMBLE).colors();
        Ok(crops
            .iter()
            .map(|crop| {
                assert_eq!(crop.len(), 16 * 16 * 4);
                let swapped = (crop[0] > crop[15 * 4]) == self.mirror;
                let mut stickers = vec![0.0; 6 * 9 * 7];
                for face in 0..3 {
                    for cell in 0..9 {
                        let from = if swapped { cell / 3 * 3 + 2 - cell % 3 } else { cell };
                        stickers[(face * 9 + cell) * 7 + state[face * 9 + from] as usize] = 1.0;
                    }
                }
                ReaderOutput {
                    present: [0.9, 0.9, 0.9, 0.0, 0.0, 0.0],
                    stickers,
                    turning: 0.1,
                    orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                }
            })
            .collect())
    }
}

/// Frames brighter on the left, each a little different from the one before; `cube`: whether a cube is found
fn images(n: usize, cube: bool) -> Vec<Image> {
    (0..n)
        .map(|i| {
            let (width, height) = (64, 48);
            let mut data = Vec::with_capacity(width * height * 4);
            for y in 0..height {
                for x in 0..width {
                    let v = if x < width / 2 { 200 } else { 40 } + ((x * 7 + y * 3 + i * 11) % 23) as u8;
                    data.extend([v, v / 2, 255 - v, 255]);
                }
            }
            data[0] = cube as u8 * 200;
            Image { data, width, height, t: i as f64 * 33.0 }
        })
        .collect()
}

#[test]
fn batches_and_single_frames_read_alike() {
    let images = images(21, true);
    let mut batched = FrameLoop::new(Middle, Sure { mirror: false, reads: 0 }, SCRAMBLE, None, 1);
    let mut seen = Vec::new();
    for chunk in images.chunks(8) {
        seen.extend(batched.push_batch(chunk).unwrap());
    }
    let mut single = FrameLoop::new(Middle, Sure { mirror: false, reads: 0 }, SCRAMBLE, None, 1);
    let alone: Vec<_> = images.iter().map(|image| single.push(image).unwrap()).collect();
    assert_eq!(seen, alone);
    let (batched, single) = (batched.finish(), single.finish());
    assert_eq!(batched, single);
    assert_eq!((batched.frames.len(), batched.found, batched.mirrored), (21, 21, false));
    assert_eq!(batched.frames.iter().map(|f| f.t).collect::<Vec<_>>(), images.iter().map(|i| i.t).collect::<Vec<_>>());
    for (i, f) in batched.frames.iter().enumerate() {
        assert!(
            f.faces.len() == 3 && f.orientation.is_some() && f.turning.is_some(),
            "frame {i}: {} faces, {:?}, {:?}",
            f.faces.len(),
            f.orientation,
            f.turning
        );
    }
    // The first frame has none before it to move from
    assert_eq!(batched.frames[0].motion, Motion::Unknown);
    assert!(matches!(batched.frames[1].motion, Motion::Change(_)));
    assert!(seen.iter().all(|s| s.boxed.is_some() && s.faces.len() == 3 && !s.turning));
    assert!(FrameLoop::new(Middle, Sure { mirror: false, reads: 0 }, SCRAMBLE, None, 1).push_batch(&[]).unwrap().is_empty());
}

#[test]
fn a_frame_without_a_cube_reads_nothing() {
    let mut frames = FrameLoop::new(Middle, Sure { mirror: false, reads: 0 }, SCRAMBLE, None, 1);
    let seen = frames.push_batch(&images(5, false)).unwrap();
    assert!(seen.iter().all(|s| s.boxed.is_none() && s.faces.is_empty()));
    let read = frames.finish();
    assert_eq!((read.frames.len(), read.found), (5, 0));
    assert!(
        read.frames.iter().all(|f| f.faces.is_empty() && f.turning.is_none() && f.orientation.is_none() && f.motion == Motion::Unknown)
    );
}

#[test]
fn a_mirrored_video_is_read_flipped() {
    for batches in [true, false] {
        let images = images(60, true);
        let mut frames = FrameLoop::new(Middle, Sure { mirror: true, reads: 0 }, SCRAMBLE, None, 1);
        if batches {
            images.chunks(8).for_each(|chunk| drop(frames.push_batch(chunk).unwrap()));
        } else {
            images.iter().for_each(|image| drop(frames.push(image).unwrap()));
        }
        let read = frames.finish();
        assert!(read.mirrored, "batches {batches}");
        // Every frame, the first ones included, holds the cube as it is: the flipped reading
        let state = Cube::solved().apply(SCRAMBLE).colors();
        for (i, frame) in read.frames.iter().enumerate() {
            for face in &frame.faces {
                for (cell, p) in face.stickers.iter().enumerate() {
                    let colour = state[face.center as usize * 9 + cell] as usize;
                    assert!(p.unwrap()[colour] > 0.8, "batches {batches}, frame {i}, face {}, cell {cell}", face.center);
                }
            }
        }
    }
}

#[test]
fn once_the_mirror_is_known_each_crop_is_read_once() {
    let images = images(80, true);
    let mut frames = FrameLoop::new(Middle, Sure { mirror: false, reads: 0 }, SCRAMBLE, None, 1);
    images.chunks(8).for_each(|chunk| drop(frames.push_batch(chunk).unwrap()));
    let reads = frames.reader().reads;
    assert!((81..160).contains(&reads), "{reads} crops read for 80 frames");
    assert!(!frames.finish().mirrored);
}
