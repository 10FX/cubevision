//! A check of this PC without a window: the models on a device, timed, then the camera read with them. What it
//! prints is meant to be sent back when something does not work. No image is kept.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use cube_capture::{Camera, CameraRequest};
use cube_infer::device::Device;
use cube_infer::follow::{CubeFollower, Detector, Look};
use cube_infer::geometry::square_crop;
use cube_infer::models::Models;
use cube_infer::output::{Reader, TURNING, readings_from_output};
use cube_infer::{BoxF, Frame};

/// The value under which `share` percent of the values lie (0 without any)
pub fn percentile(values: &mut [f64], share: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * share / 100.0).round() as usize]
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

/// A grey picture with a few coloured squares: something for the models to chew on
fn picture(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let square = ((x / 61 + y / 47) % 6) as u8;
            data.extend([90 + square * 25, 110, 200 - square * 30, 255]);
        }
    }
    data
}

/// The models loaded on a device and run for `seconds` on a still picture: how long each takes
pub fn models(dir: &Path, device: Option<Device>, seconds: f64) -> Result<Models> {
    let loading = Instant::now();
    let mut models = Models::load(dir, device)?;
    println!("device: {}", models.device.name());
    if let Some(why) = &models.fallback {
        println!("note: {why}");
    }
    println!("loaded and first run in {:.1} s", loading.elapsed().as_secs_f64());
    let data = picture(1280, 720);
    let frame = Frame { data: &data, width: 1280, height: 720 };
    let boxed = BoxF { x: 400.0, y: 120.0, w: 480.0, h: 480.0 };
    let (mut detector, mut reader) = (Vec::new(), Vec::new());
    let until = Instant::now() + Duration::from_secs_f64(seconds);
    while Instant::now() < until {
        let t = Instant::now();
        models.detector.look(&[Look { frame: &frame, region: None }])?;
        detector.push(ms(t));
        let t = Instant::now();
        models.reader.read(&[&square_crop(&frame, &boxed, models.reader.size())])?;
        reader.push(ms(t));
    }
    let mut total: Vec<f64> = detector.iter().zip(&reader).map(|(d, r)| d + r).collect();
    println!("frames: {} in {seconds:.0} s", total.len());
    println!("detector ms: p50 {:.1}, p95 {:.1}", percentile(&mut detector.clone(), 50.0), percentile(&mut detector, 95.0));
    println!("reader ms:   p50 {:.1}, p95 {:.1}", percentile(&mut reader.clone(), 50.0), percentile(&mut reader, 95.0));
    let p95 = percentile(&mut total.clone(), 95.0);
    println!("both ms:     p50 {:.1}, p95 {p95:.1} ({:.0} frames a second)", percentile(&mut total, 50.0), 1000.0 / p95.max(1e-3));
    Ok(models)
}

/// An RGBA picture written as a PNG
pub fn save_png(file: &Path, rgba: &[u8], width: usize, height: usize) -> Result<()> {
    let out = std::io::BufWriter::new(std::fs::File::create(file)?);
    let mut encoder = png::Encoder::new(out, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&rgba[..width * height * 4])?;
    Ok(())
}

/// The camera read with the models for `seconds`, as the live mode reads it. `crop`: where to save the crop the
/// reader saw the most of a cube in, for the user to look at (nothing is saved otherwise).
pub fn camera(models: Models, request: &CameraRequest, seconds: f64, crop: Option<&Path>) -> Result<()> {
    let camera = Camera::open(request)?;
    let f = camera.format();
    println!("camera: {} {}×{} at {} fps, pixels as {}", f.name, f.width, f.height, f.fps, f.pixels);
    let mut colour = [0u64; 3];
    let mut sampled = 0u64;
    // What the reader says before any threshold: how present the likeliest face is, how sure its stickers
    let (mut presence, mut turned, mut present_faces, mut kept_faces) = (0.0f64, 0.0f64, 0usize, 0usize);
    let mut best: Option<(f32, Vec<u8>)> = None;
    let Models { detector, mut reader, .. } = models;
    let mut follower = CubeFollower::new(detector, 3);
    let (mut frames, mut found, mut read, mut turning, mut faces) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut times = Vec::new();
    let start = Instant::now();
    while start.elapsed().as_secs_f64() < seconds {
        let Some(image) = camera.wait(Duration::from_millis(200))? else { continue };
        let t = Instant::now();
        let frame = Frame { data: &image.data, width: image.width, height: image.height };
        for px in image.data.chunks_exact(4).step_by(97) {
            (0..3).for_each(|c| colour[c] += px[c] as u64);
            sampled += 1;
        }
        if let Some(boxed) = follower.crop(&frame)? {
            let square = square_crop(&frame, &boxed, reader.size());
            let out = reader.read(&[&square])?.remove(0);
            let readings = readings_from_output(&out, reader.ordered(), false);
            let most = out.present.iter().fold(0.0f32, |a, &b| a.max(b));
            presence += most as f64;
            turned += out.turning as f64;
            present_faces += out.present.iter().filter(|&&p| p >= 0.5).count();
            kept_faces += readings.len();
            if crop.is_some() && best.as_ref().is_none_or(|(seen, _)| most > *seen) {
                best = Some((most, square));
            }
            follower.read(!readings.is_empty() || out.turning > TURNING);
            found += 1;
            read += !readings.is_empty() as usize;
            turning += (out.turning > TURNING) as usize;
            faces += readings.len();
        }
        times.push(ms(t));
        frames += 1;
    }
    let share = |n: usize| 100.0 * n as f64 / frames.max(1) as f64;
    println!("frames read: {frames} in {seconds:.0} s ({:.1} a second), {} skipped", frames as f64 / seconds, camera.dropped());
    println!("ms a frame: p50 {:.1}, p95 {:.1}", percentile(&mut times.clone(), 50.0), percentile(&mut times, 95.0));
    println!(
        "cube found: {found} ({:.0} %); faces read on {read} ({:.0} %), {:.2} faces each; mid-turn: {turning} ({:.0} %)",
        share(found),
        share(read),
        faces as f64 / read.max(1) as f64,
        share(turning)
    );
    let mean = |c: usize| colour[c] as f64 / sampled.max(1) as f64;
    println!("mean colour of the frames (red green blue): {:.0} {:.0} {:.0}", mean(0), mean(1), mean(2));
    let per = found.max(1) as f64;
    println!(
        "reader, where the cube is found: likeliest face present {:.2} on average, turning {:.2}; faces present {present_faces}, kept as sure enough {kept_faces}",
        presence / per,
        turned / per
    );
    if let (Some(file), Some((seen, square))) = (crop, &best) {
        save_png(file, square, reader.size(), reader.size())?;
        println!("crop saved to {} (likeliest face present {seen:.2}): what the reader was given, to look at", file.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_of_a_few_values() {
        let mut values = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!((percentile(&mut values, 50.0), percentile(&mut values, 95.0), percentile(&mut values, 0.0)), (3.0, 5.0, 1.0));
        assert_eq!(percentile(&mut [], 50.0), 0.0);
        assert_eq!(picture(8, 4).len(), 8 * 4 * 4);
    }
}
