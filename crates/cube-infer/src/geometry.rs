//! The frame into the models' squares (`detector.ts`: letterbox, unletterbox, readerCrop; `roi.ts`: squareCrop;
//! `model.ts`: toTensorData). Frames are RGBA, four bytes a pixel, row by row.

use std::fmt;

use crate::Frame;

/// A box in frame pixels (it may pass the frame's edges)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxF {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl BoxF {
    /// A box with a place and a size: finite, wider and taller than nothing
    pub fn is_real(&self) -> bool {
        [self.x, self.y, self.w, self.h].iter().all(|v| v.is_finite()) && self.w > 0.0 && self.h > 0.0
    }
}

/// Where a frame, or a region of it, sits in the detector's square
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub scale: f64,
    pub ox: i64,
    pub oy: i64,
    /// The region's corner in the frame
    pub rx: f64,
    pub ry: f64,
}

/// A frame that cannot be read: no pixel, or fewer bytes than its size says
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameError(String);

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FrameError {}

/// Whether a frame holds the pixels its size says: what `letterbox` and `square_crop` need
pub fn check(frame: &Frame) -> Result<(), FrameError> {
    let Frame { data, width, height } = *frame;
    if width == 0 || height == 0 {
        return Err(FrameError(format!("a {width}×{height} frame has no pixel")));
    }
    match width.checked_mul(height).and_then(|n| n.checked_mul(4)) {
        Some(n) if data.len() >= n => Ok(()),
        _ => Err(FrameError(format!("frame data ({} bytes) shorter than a {width}×{height} RGBA frame", data.len()))),
    }
}

fn checked(frame: &Frame) {
    if let Err(e) = check(frame) {
        panic!("{e}");
    }
}

/// JavaScript's Math.round: halves go up
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

/// The frame (or a region of it, which may pass its edges) fitted into a `size` square, pixels averaged, the
/// bands and what lies past the frame filled with `pad`: three planes (R, G, B) in 0..1, and how to map back.
///
/// A region no frame can hold (not finite, or empty) gives a square of `pad`; so does one so large that a pixel
/// of the square would cover more than the whole frame (the site reads past the frame there).
///
/// Panics when the frame is not one (`check`).
pub fn letterbox(frame: &Frame, size: usize, pad: f32, region: Option<&BoxF>) -> (Vec<f32>, Letterbox) {
    checked(frame);
    let (width, height) = (frame.width as i64, frame.height as i64);
    let region = region.copied().unwrap_or(BoxF { x: 0.0, y: 0.0, w: width as f64, h: height as f64 });
    let plane = size * size;
    let mut out = vec![pad; 3 * plane];
    let scale = size as f64 / region.w.max(region.h);
    let lb = Letterbox { scale, ox: 0, oy: 0, rx: region.x, ry: region.y };
    let span = js_round(1.0 / scale).max(1.0);
    if !region.is_real() || !scale.is_finite() || span > width as f64 || span > height as f64 {
        return (out, lb);
    }
    let w = js_round(region.w * scale).max(1.0) as i64;
    let h = js_round(region.h * scale).max(1.0) as i64;
    let size = size as i64;
    let (ox, oy) = ((size - w).div_euclid(2), (size - h).div_euclid(2));
    let half = span / 2.0;
    let span = span as i64;
    let n = (span * span * 255) as f64;
    // The columns of the frame under each column of the square, once
    let columns: Vec<Option<usize>> = (0..w)
        .map(|x| {
            let left = js_round(region.x + (x as f64 + 0.5) / scale - half);
            (left + span as f64 > 0.0 && left < width as f64).then(|| (width - span).min((left as i64).max(0)) as usize)
        })
        .collect();
    for y in 0..h {
        let top = js_round(region.y + (y as f64 + 0.5) / scale - half);
        if top + span as f64 <= 0.0 || top >= height as f64 {
            continue;
        }
        let sy = (height - span).min((top as i64).max(0));
        for (x, sx) in columns.iter().enumerate() {
            let Some(sx) = *sx else { continue };
            let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
            for dy in 0..span {
                let row = ((sy + dy) * width) as usize + sx;
                for px in frame.data[row * 4..(row + span as usize) * 4].chunks_exact(4) {
                    r += px[0] as u64;
                    g += px[1] as u64;
                    b += px[2] as u64;
                }
            }
            let o = ((oy + y) * size + ox) as usize + x;
            out[o] = (r as f64 / n) as f32;
            out[plane + o] = (g as f64 / n) as f32;
            out[2 * plane + o] = (b as f64 / n) as f32;
        }
    }
    (out, Letterbox { ox, oy, ..lb })
}

/// A box of the square (shares, x0 y0 x1 y1) back in frame pixels
pub fn unletterbox(boxed: [f32; 4], size: usize, lb: &Letterbox) -> BoxF {
    let size = size as f64;
    let fx = |v: f32| (v as f64 * size - lb.ox as f64) / lb.scale + lb.rx;
    let fy = |v: f32| (v as f64 * size - lb.oy as f64) / lb.scale + lb.ry;
    BoxF { x: fx(boxed[0]), y: fy(boxed[1]), w: fx(boxed[2]) - fx(boxed[0]), h: fy(boxed[3]) - fy(boxed[1]) }
}

/// The reader's crop: a square of the box's longer side times `margin` around its centre
pub fn reader_crop(boxed: &BoxF, margin: f64) -> BoxF {
    let side = boxed.w.max(boxed.h) * margin;
    BoxF { x: boxed.x + boxed.w / 2.0 - side / 2.0, y: boxed.y + boxed.h / 2.0 - side / 2.0, w: side, h: side }
}

/// A box so much larger than the frame that averaging it is all edge: read as black
const FAR_TOO_LARGE: f64 = 64.0;

/// The square around the box resampled to `size`×`size` RGBA, as the reader was trained: an area average when
/// shrinking, a box cut by the frame's edge completed by repeating the edge pixels.
///
/// A box that is not finite, or more than 64 times the frame, gives a black square.
///
/// Panics when the frame is not one (`check`).
pub fn square_crop(frame: &Frame, boxed: &BoxF, size: usize) -> Vec<u8> {
    checked(frame);
    let (width, height) = (frame.width as i64, frame.height as i64);
    let mut out = vec![0u8; size * size * 4];
    out.chunks_exact_mut(4).for_each(|px| px[3] = 255);
    let side = boxed.w.max(boxed.h);
    let finite = [boxed.x, boxed.y, boxed.w, boxed.h].iter().all(|v| v.is_finite());
    if !finite || side.abs() > FAR_TOO_LARGE * width.max(height) as f64 {
        return out;
    }
    let left = boxed.x + boxed.w / 2.0 - side / 2.0;
    let top = boxed.y + boxed.h / 2.0 - side / 2.0;
    let scale = side / size as f64;
    // Source pixels averaged per output pixel along each axis (at least one)
    let span = js_round(scale).max(1.0) as i64;
    let half = span as f64 / 2.0;
    let n = (span * span) as f64;
    let columns: Vec<usize> = (0..size)
        .flat_map(|x| {
            let sx = left + (x as f64 + 0.5) * scale - half;
            (0..span).map(move |dx| (js_round(sx + dx as f64) as i64).clamp(0, width - 1) as usize * 4)
        })
        .collect();
    for y in 0..size {
        let sy = top + (y as f64 + 0.5) * scale - half;
        let rows: Vec<usize> = (0..span).map(|dy| ((js_round(sy + dy as f64) as i64).clamp(0, height - 1) * width) as usize * 4).collect();
        for (x, across) in columns.chunks_exact(span as usize).enumerate() {
            let mut sum = [0u64; 3];
            for row in &rows {
                for column in across {
                    let i = row + column;
                    sum[0] += frame.data[i] as u64;
                    sum[1] += frame.data[i + 1] as u64;
                    sum[2] += frame.data[i + 2] as u64;
                }
            }
            let o = (y * size + x) * 4;
            for c in 0..3 {
                // As a Uint8ClampedArray takes a number: to the nearest, halves to the even one
                out[o + c] = (sum[c] as f64 / n).round_ties_even() as u8;
            }
        }
    }
    out
}

/// An RGBA image as three planes (R, G, B) in 0..1, the models' input (they normalize themselves).
/// `out` holds three values a pixel.
pub fn tensor(rgba: &[u8], out: &mut [f32]) {
    let plane = out.len() / 3;
    for (i, px) in rgba.chunks_exact(4).take(plane).enumerate() {
        out[i] = (px[0] as f64 / 255.0) as f32;
        out[plane + i] = (px[1] as f64 / 255.0) as f32;
        out[2 * plane + i] = (px[2] as f64 / 255.0) as f32;
    }
}
