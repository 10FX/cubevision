//! How much the cube's crop changed since the frame before (ChronoCube's vision/motion.ts): a turn, the hands or the
//! cube moving all change it, a cube held still barely (the noise of the camera). The decoder takes no turn between
//! two readings when nothing moved between them: a reading that changed then was misread.

/// An image: RGBA bytes, row by row
#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
}

/// Side of the grey thumbnail the crops are compared at
const SIDE: usize = 24;

/// The crop as a grey SIDE × SIDE thumbnail (the mean of each cell)
pub fn thumbnail(f: &Frame) -> [f32; SIDE * SIDE] {
    let mut out = [0f32; SIDE * SIDE];
    let mut counts = [0u32; SIDE * SIDE];
    // A pixel in two, both ways: enough for a mean
    for y in (0..f.height).step_by(2) {
        let row = (SIDE - 1).min(y * SIDE / f.height) * SIDE;
        for x in (0..f.width).step_by(2) {
            let i = (y * f.width + x) * 4;
            let cell = row + (SIDE - 1).min(x * SIDE / f.width);
            let grey = 0.299 * f.data[i] as f64 + 0.587 * f.data[i + 1] as f64 + 0.114 * f.data[i + 2] as f64;
            // Summed in f32, as ChronoCube's Float32Array
            out[cell] = (out[cell] as f64 + grey) as f32;
            counts[cell] += 1;
        }
    }
    for (o, &count) in out.iter_mut().zip(&counts) {
        *o = (*o as f64 / (count.max(1) as f64 * 255.0)) as f32;
    }
    out
}

/// Motion is told per this many ms: at 60 fps a frame changes half as much from the one before as at 30
const PER_MS: f64 = 1000.0 / 30.0;

/// The change from one crop to the next: the mean difference of their thumbnails (0–1) per 1/30 s, None without both
#[derive(Clone, Debug, Default)]
pub struct MotionMeter {
    last: Option<([f32; SIDE * SIDE], f64)>,
}

impl MotionMeter {
    /// The crop of the cube in a frame (None: the cube was not found) and the frame's time (ms)
    pub fn measure(&mut self, crop: Option<&Frame>, t: f64) -> Option<f64> {
        let Some(crop) = crop else {
            self.last = None;
            return None;
        };
        let thumb = thumbnail(crop);
        let (last, last_t) = self.last.replace((thumb, t))?;
        let sum: f64 = thumb.iter().zip(&last).map(|(&a, &b)| (a as f64 - b as f64).abs()).sum();
        Some(sum / thumb.len() as f64 * (PER_MS / (t - last_t).max(1.0)))
    }
}
