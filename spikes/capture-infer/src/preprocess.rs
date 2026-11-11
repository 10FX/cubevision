//! The frame into the models' squares, as `src/lib/vision/detector.ts` and `roi.ts` do on the site

/// An RGB image, 3 bytes per pixel, row by row
#[derive(Clone, Debug)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// A box in frame pixels
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxF {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Where the frame sits in the detector's square
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub ox: i32,
    pub oy: i32,
}

/// JavaScript's Math.round: halves go up
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

/// The whole frame in a `size` square, bands filled with `pad`: CHW planes in 0..1
pub fn letterbox(frame: &Frame, size: u32, pad: f32) -> (Vec<f32>, Letterbox) {
    let (width, height) = (frame.width as i64, frame.height as i64);
    let scale = size as f64 / frame.width.max(frame.height) as f64;
    let w = js_round(width as f64 * scale).max(1.0) as i64;
    let h = js_round(height as f64 * scale).max(1.0) as i64;
    let size = size as i64;
    let (ox, oy) = ((size - w).div_euclid(2), (size - h).div_euclid(2));
    let plane = (size * size) as usize;
    let mut out = vec![pad; 3 * plane];
    let span = js_round(1.0 / scale).max(1.0) as i64;
    let n = (span * span * 255) as f32;
    for y in 0..h {
        let top = js_round((y as f64 + 0.5) / scale - span as f64 / 2.0) as i64;
        if top + span <= 0 || top >= height {
            continue;
        }
        let sy = (height - span).min(top.max(0));
        for x in 0..w {
            let left = js_round((x as f64 + 0.5) / scale - span as f64 / 2.0) as i64;
            if left + span <= 0 || left >= width {
                continue;
            }
            let sx = (width - span).min(left.max(0));
            let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
            for dy in 0..span {
                let row = ((sy + dy) * width + sx) as usize * 3;
                for px in frame.data[row..row + span as usize * 3].chunks_exact(3) {
                    r += px[0] as u32;
                    g += px[1] as u32;
                    b += px[2] as u32;
                }
            }
            let o = ((oy + y) * size + ox + x) as usize;
            out[o] = r as f32 / n;
            out[plane + o] = g as f32 / n;
            out[2 * plane + o] = b as f32 / n;
        }
    }
    (out, Letterbox { scale: scale as f32, ox: ox as i32, oy: oy as i32 })
}

/// A box of the square (shares, x0 y0 x1 y1) back in frame pixels
pub fn unletterbox(boxed: [f32; 4], size: u32, lb: &Letterbox) -> BoxF {
    let size = size as f32;
    let fx = |v: f32| (v * size - lb.ox as f32) / lb.scale;
    let fy = |v: f32| (v * size - lb.oy as f32) / lb.scale;
    BoxF { x: fx(boxed[0]), y: fy(boxed[1]), w: fx(boxed[2]) - fx(boxed[0]), h: fy(boxed[3]) - fy(boxed[1]) }
}

/// The reader's crop: a square of the box's longer side times `margin` around its centre
pub fn reader_crop(boxed: &BoxF, margin: f32) -> BoxF {
    let side = boxed.w.max(boxed.h) * margin;
    BoxF { x: boxed.x + boxed.w / 2.0 - side / 2.0, y: boxed.y + boxed.h / 2.0 - side / 2.0, w: side, h: side }
}

/// The square around `boxed` in a `size` square, pixels averaged, edges clamped: CHW planes in 0..1,
/// rounded to bytes first as the site's canvas does
pub fn square_crop(frame: &Frame, boxed: &BoxF, size: u32) -> Vec<f32> {
    let side = boxed.w.max(boxed.h) as f64;
    let left = boxed.x as f64 + boxed.w as f64 / 2.0 - side / 2.0;
    let top = boxed.y as f64 + boxed.h as f64 / 2.0 - side / 2.0;
    let scale = side / size as f64;
    let (width, height) = (frame.width as i64, frame.height as i64);
    let span = js_round(scale).max(1.0) as i64;
    let n = (span * span) as f64;
    let plane = (size * size) as usize;
    let mut out = vec![0.0; 3 * plane];
    for y in 0..size as usize {
        let sy = top + (y as f64 + 0.5) * scale - span as f64 / 2.0;
        for x in 0..size as usize {
            let sx = left + (x as f64 + 0.5) * scale - span as f64 / 2.0;
            let mut sum = [0u32; 3];
            for dy in 0..span {
                let row = (js_round(sy + dy as f64) as i64).clamp(0, height - 1) * width;
                for dx in 0..span {
                    let i = (row + (js_round(sx + dx as f64) as i64).clamp(0, width - 1)) as usize * 3;
                    sum[0] += frame.data[i] as u32;
                    sum[1] += frame.data[i + 1] as u32;
                    sum[2] += frame.data[i + 2] as u32;
                }
            }
            let o = y * size as usize + x;
            for c in 0..3 {
                out[c * plane + o] = (sum[c] as f64 / n).round_ties_even() as f32 / 255.0;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank(width: u32, height: u32) -> Frame {
        Frame { data: vec![0; (width * height * 3) as usize], width, height }
    }

    fn geometry(width: u32, height: u32, size: u32) -> (i32, i32, f32) {
        let (_, lb) = letterbox(&blank(width, height), size, 0.5);
        (lb.oy, lb.ox, lb.scale)
    }
    fn ox(width: u32, height: u32, size: u32) -> i32 { geometry(width, height, size).1 }
    fn oy(width: u32, height: u32, size: u32) -> i32 { geometry(width, height, size).0 }

    #[test]
    fn letterbox_landscape() {
        assert_eq!(geometry(1920, 1080, 320), (70, 0, 1.0 / 6.0));
    }

    #[test]
    fn letterbox_portrait() {
        assert_eq!(ox(480, 854, 320), 70);
        assert_eq!(oy(480, 854, 320), 0);
    }

    #[test]
    fn letterbox_fills_the_bands_with_the_pad() {
        let (tensor, _) = letterbox(&blank(1920, 1080), 320, 0.5);
        assert_eq!(tensor.len(), 3 * 320 * 320);
        assert_eq!(tensor[0], 0.5); // top band
        assert_eq!(tensor[160 * 320 + 160], 0.0); // the black frame
    }

    #[test]
    fn reader_crop_is_square_around_the_box() {
        let c = reader_crop(&BoxF { x: 100., y: 100., w: 40., h: 80. }, 1.5);
        assert_eq!((c.x, c.y, c.w, c.h), (60., 80., 120., 120.));
    }

    #[test]
    fn unletterbox_undoes_letterbox() {
        let (_, lb) = letterbox(&blank(1920, 1080), 320, 0.5);
        let b = unletterbox([0.0, 0.21875, 1.0, 0.78125], 320, &lb);
        for (got, want) in [(b.x, 0.0), (b.y, 0.0), (b.w, 1920.0), (b.h, 1080.0)] {
            assert!((got - want).abs() <= 0.5, "{got} vs {want}");
        }
    }

    #[test]
    fn square_crop_averages_the_colours_inside() {
        let mut frame = blank(40, 40);
        for px in frame.data.chunks_mut(3) { px.copy_from_slice(&[255, 0, 51]); }
        let t = square_crop(&frame, &BoxF { x: 0., y: 0., w: 40., h: 40. }, 20);
        assert_eq!(t.len(), 3 * 20 * 20);
        assert_eq!((t[0], t[400], t[800]), (1.0, 0.0, 0.2));
    }
}
