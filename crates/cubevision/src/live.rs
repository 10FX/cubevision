//! The live mode: the camera read frame after frame in a thread of its own (ChronoCube's `liveWorker.ts`,
//! without a scramble to follow), the moves written with a delay, the 3D cube playing them.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use cube_capture::{Camera, CameraRequest, Image, VideoOptions, VideoReader};
use cube_core::Cube;
use cube_decode::delayed::DelayedTracker;
use cube_decode::state::{MOVES, solved_state};
use cube_infer::device::Device;
use cube_infer::follow::CubeFollower;
use cube_infer::geometry::square_crop;
use cube_infer::models::Models;
use cube_infer::output::{Reader, TURNING, readings_from_output};
use cube_infer::{BoxF, Frame};
use cube_view::CubeView;
use eframe::egui;

use crate::window::{Snapshot, token};

/// Where the frames come from
#[derive(Clone, Debug)]
pub enum Source {
    Camera(CameraRequest),
    /// A video played at its own pace, as a camera would show it: to try the live mode without one
    Video(PathBuf),
}

pub struct LiveOptions {
    pub source: Source,
    pub models: PathBuf,
    pub device: Option<Device>,
    pub snapshot: Snapshot,
}

/// Widest picture of the camera handed to the window
const PREVIEW: usize = 640;

/// What the reading thread shows the window
#[derive(Default)]
struct Shared {
    /// What is going on before frames come: "loading the models…"
    status: String,
    error: Option<String>,
    /// The latest frame, shrunk (RGBA), taken by the window when it draws
    preview: Option<(Vec<u8>, usize, usize)>,
    /// The reader's square in that frame, as shares of its width and height
    boxed: Option<[f32; 4]>,
    /// Faces read in that frame
    faces: usize,
    turning: bool,
    confirmed: Vec<usize>,
    pending: Vec<usize>,
    /// Frames read a second, and the time one takes (ms), smoothed
    fps: f32,
    read_ms: f32,
    /// Since when the cube is in sight without a face read
    unread_since: Option<Instant>,
    source: String,
    device: String,
    note: Option<String>,
    dropped: u64,
}

struct Frames {
    camera: Option<Camera>,
    video: Option<(VideoReader, Instant)>,
}

impl Frames {
    fn open(source: &Source) -> Result<(Frames, String)> {
        match source {
            Source::Camera(request) => {
                let camera = Camera::open(request)?;
                let f = camera.format();
                let name = format!("{} {}×{} at {} fps", f.name, f.width, f.height, f.fps);
                Ok((Frames { camera: Some(camera), video: None }, name))
            }
            Source::Video(path) => {
                let video = VideoReader::open(path, &VideoOptions::default())?;
                let name =
                    format!("{} {}×{}", path.file_name().unwrap_or_default().to_string_lossy(), video.info().width, video.info().height);
                Ok((Frames { camera: None, video: Some((video, Instant::now())) }, name))
            }
        }
    }

    /// The next frame to read: the camera's latest, or the video's next one when its time has come.
    /// None: none yet (or the video is over).
    fn next(&mut self) -> Result<Option<Image>> {
        if let Some(camera) = &self.camera {
            return camera.wait(Duration::from_millis(200));
        }
        let Some((video, started)) = &mut self.video else { return Ok(None) };
        match video.next_image()? {
            Some(image) => {
                let due = *started + Duration::from_secs_f64(image.t.max(0.0) / 1000.0);
                std::thread::sleep(due.saturating_duration_since(Instant::now()));
                Ok(Some(image))
            }
            None => {
                std::thread::sleep(Duration::from_millis(200));
                Ok(None)
            }
        }
    }

    fn dropped(&self) -> u64 {
        self.camera.as_ref().map_or(0, Camera::dropped)
    }
}

/// The frame shrunk for the window: one pixel in so many, flipped left to right when `mirror` (a camera facing
/// its user is watched as a mirror)
fn preview(image: &Image, mirror: bool) -> (Vec<u8>, usize, usize) {
    let step = image.width.div_ceil(PREVIEW).max(1);
    let (width, height) = (image.width / step, image.height / step);
    let mut out = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let from = if mirror { width - 1 - x } else { x };
            let i = (y * step * image.width + from * step) * 4;
            out.extend_from_slice(&image.data[i..i + 4]);
        }
    }
    (out, width, height)
}

fn reading(options: &LiveOptions, shared: &Mutex<Shared>, reset: &AtomicBool, stop: &AtomicBool, ctx: &egui::Context) -> Result<()> {
    let say = |status: &str| {
        shared.lock().unwrap().status = status.to_owned();
        ctx.request_repaint();
    };
    say("Loading the models…");
    let models = Models::load(&options.models, options.device)?;
    {
        let mut shared = shared.lock().unwrap();
        shared.device = models.device.name().to_owned();
        shared.note = models.fallback.clone();
    }
    let Models { detector, mut reader, .. } = models;
    say("Opening the camera…");
    let (mut frames, name) = Frames::open(&options.source)?;
    shared.lock().unwrap().source = name;
    say("");
    let mirror = matches!(options.source, Source::Camera(_));
    // As the site's live mode: the detector looks at one frame in 3 while the reader reads the cube
    let mut follower = CubeFollower::new(detector, 3);
    let mut tracker = DelayedTracker::new(solved_state());
    let mut last = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        let Some(image) = frames.next()? else { continue };
        if reset.swap(false, Ordering::Relaxed) {
            tracker.reset(solved_state());
            follower.boxes.reset();
        }
        let started = Instant::now();
        let frame = Frame { data: &image.data, width: image.width, height: image.height };
        let boxed = follower.crop(&frame)?;
        let (mut faces, mut turning) = (Vec::new(), false);
        if let Some(boxed) = &boxed {
            let out = reader.read(&[&square_crop(&frame, boxed, reader.size())])?.remove(0);
            faces = readings_from_output(&out, reader.ordered(), false);
            turning = out.turning > TURNING;
            follower.read(!faces.is_empty() || turning);
        }
        tracker.update(&faces);
        let read_ms = started.elapsed().as_secs_f32() * 1000.0;
        let between = last.elapsed().as_secs_f32().max(1e-3);
        last = Instant::now();

        let mut shared = shared.lock().unwrap();
        shared.preview = Some(preview(&image, mirror));
        shared.boxed = boxed.map(|b: BoxF| {
            let (w, h) = (image.width as f64, image.height as f64);
            let x = if mirror { 1.0 - (b.x + b.w) / w } else { b.x / w };
            [x as f32, (b.y / h) as f32, (b.w / w) as f32, (b.h / h) as f32]
        });
        shared.faces = faces.len();
        shared.turning = turning;
        shared.unread_since = match (boxed.is_some() && faces.is_empty() && !turning, shared.unread_since) {
            (true, since) => since.or(Some(Instant::now())),
            (false, _) => None,
        };
        shared.confirmed = tracker.confirmed().to_vec();
        shared.pending = tracker.pending().to_vec();
        shared.fps += (1.0 / between - shared.fps) * 0.1;
        shared.read_ms += (read_ms - shared.read_ms) * 0.1;
        shared.dropped = frames.dropped();
        drop(shared);
        ctx.request_repaint();
    }
    Ok(())
}

struct LiveApp {
    shared: Arc<Mutex<Shared>>,
    reset: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    cube: CubeView,
    /// Moves written already handed to the 3D cube
    played: usize,
    picture: Option<egui::TextureHandle>,
    snapshot: Snapshot,
}

impl LiveApp {
    fn new(options: LiveOptions, ctx: &egui::Context) -> Self {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (reset, stop) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let snapshot = options.snapshot.clone();
        {
            let (shared, reset, stop, ctx) = (shared.clone(), reset.clone(), stop.clone(), ctx.clone());
            std::thread::spawn(move || {
                if let Err(e) = reading(&options, &shared, &reset, &stop, &ctx) {
                    shared.lock().unwrap().error = Some(format!("{e:#}"));
                    ctx.request_repaint();
                }
            });
        }
        LiveApp { shared, reset, stop, cube: CubeView::new(Cube::solved()), played: 0, picture: None, snapshot }
    }

    fn start_again(&mut self) {
        self.reset.store(true, Ordering::Relaxed);
        self.cube.set(Cube::solved());
        self.played = 0;
        let mut shared = self.shared.lock().unwrap();
        shared.confirmed.clear();
        shared.pending.clear();
    }
}

impl eframe::App for LiveApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.snapshot.tick(ctx);
        if ctx.input(|i| i.key_pressed(egui::Key::R) || i.key_pressed(egui::Key::Space)) {
            self.start_again();
        }
        let (status, error, boxed, faces, turning, confirmed, pending, fps, read_ms, unread, source, device, note, dropped) = {
            let mut s = self.shared.lock().unwrap();
            if let Some((data, width, height)) = s.preview.take() {
                let image = egui::ColorImage::from_rgba_unmultiplied([width, height], &data);
                match &mut self.picture {
                    Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                    None => self.picture = Some(ctx.load_texture("camera", image, egui::TextureOptions::LINEAR)),
                }
            }
            (
                s.status.clone(),
                s.error.clone(),
                s.boxed,
                s.faces,
                s.turning,
                s.confirmed.clone(),
                s.pending.clone(),
                s.fps,
                s.read_ms,
                s.unread_since.is_some_and(|since| since.elapsed() > Duration::from_secs(2)),
                s.source.clone(),
                s.device.clone(),
                s.note.clone(),
                s.dropped,
            )
        };
        // The moves written since the last frame go to the 3D cube; fewer than played: a new start
        if confirmed.len() < self.played {
            self.cube.set(Cube::solved());
            self.played = 0;
        }
        for &m in &confirmed[self.played..] {
            self.cube.play(MOVES[m]);
        }
        self.played = confirmed.len();

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if !device.is_empty() {
                    ui.label(format!("Models on {device}"));
                    ui.separator();
                    ui.label(format!("{fps:.0} frames a second, {read_ms:.0} ms each"));
                    ui.separator();
                    ui.label(&source);
                    if dropped > 0 {
                        ui.separator();
                        ui.label(format!("{dropped} frames skipped"));
                    }
                }
                if let Some(note) = &note {
                    ui.separator();
                    ui.label(egui::RichText::new(note).weak());
                }
            });
        });
        egui::SidePanel::right("cube").resizable(false).exact_width((ctx.screen_rect().width() * 0.42).clamp(320.0, 560.0)).show(
            ctx,
            |ui| {
                ui.add_space(8.0);
                let side = ui.available_width().min(ui.available_height() * 0.55);
                ui.vertical_centered(|ui| self.cube.show(ui, side));
                ui.horizontal(|ui| {
                    ui.heading(format!("{} moves", confirmed.len()));
                    if ui.button("Start again (R)").on_hover_text("The cube is solved, white on top, green in front").clicked() {
                        self.start_again();
                    }
                });
                egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 10.0;
                        for &m in &confirmed {
                            token(ui, MOVES[m], ui.visuals().strong_text_color());
                        }
                        // Seen, not settled yet: they may still change
                        for &m in &pending {
                            token(ui, MOVES[m], ui.visuals().weak_text_color());
                        }
                    });
                });
            },
        );
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(error) = &error {
                ui.add_space(20.0);
                ui.heading("The live mode stopped");
                ui.label(egui::RichText::new(error).color(egui::Color32::from_rgb(255, 120, 100)));
                return;
            }
            let Some(picture) = &self.picture else {
                ui.centered_and_justified(|ui| ui.heading(if status.is_empty() { "Waiting for the camera…" } else { &status }));
                return;
            };
            let size = picture.size_vec2();
            let scale = (ui.available_width() / size.x).min((ui.available_height() - 64.0) / size.y).max(0.1);
            let (rect, _) = ui.allocate_exact_size(size * scale, egui::Sense::hover());
            ui.painter().image(picture.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            if let Some([x, y, w, h]) = boxed {
                let found = egui::Rect::from_min_size(rect.min + egui::vec2(x * rect.width(), y * rect.height()), egui::vec2(w * rect.width(), h * rect.height()));
                let colour = if faces > 0 { egui::Color32::from_rgb(80, 220, 120) } else { egui::Color32::from_rgb(240, 200, 60) };
                ui.painter_at(rect).rect_stroke(found, 4.0, egui::Stroke::new(2.5, colour), egui::StrokeKind::Inside);
            }
            ui.add_space(6.0);
            let what = match (boxed.is_some(), faces, turning) {
                (false, _, _) => "No cube in sight".to_owned(),
                (true, 0, true) => "A layer is turning".to_owned(),
                (true, 0, false) if unread => "The cube is seen but not read: more light, or hold it still a moment".to_owned(),
                (true, 0, false) => "Cube in sight".to_owned(),
                (true, n, _) => format!("{n} face{} read", if n == 1 { "" } else { "s" }),
            };
            ui.label(egui::RichText::new(what).size(16.0));
            ui.label(egui::RichText::new("Start from a solved cube, white on top, green in front. If the cube on screen no longer matches yours, solve it and press R.").weak());
        });
    }

    fn on_exit(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Opens the live mode's window; returns when it is closed
pub fn run(options: LiveOptions) -> Result<()> {
    crate::window::run("cubevision: live", [1180.0, 680.0], |cc| Box::new(LiveApp::new(options, &cc.egui_ctx)))
}
