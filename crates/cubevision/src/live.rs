//! The live mode: the camera read frame after frame in a thread of its own (ChronoCube's `liveWorker.ts`,
//! without a scramble to follow), the moves written with a delay, the 3D cube playing them.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use cube_capture::{Camera, CameraRequest, Image, VideoOptions, VideoReader};
use cube_core::Cube;
use cube_decode::delayed::DelayedTracker;
use cube_decode::reading::State;
use cube_decode::state::{MOVES, apply_move, solved_state, state_of};
use cube_infer::device::Device;
use cube_infer::follow::CubeFollower;
use cube_infer::geometry::square_crop;
use cube_infer::models::Models;
use cube_infer::output::{Reader, TURNING, readings_from_output};
use cube_infer::{BoxF, Frame};
use cube_view::CubeView;
use eframe::egui;

use crate::theme::{self, Chip};
use crate::window::{Next, Screen, Snapshot};

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
    /// The camera shown as a mirror; the screen's button changes it
    pub mirror: Arc<AtomicBool>,
    /// The scramble the cube starts from, held white on top and green in front (None: solved)
    pub start: Option<String>,
    pub snapshot: Snapshot,
}

/// Below this share of the stickers read fitting the cube followed, the two may no longer match. Measured on a
/// synthetic solve: about 0.83 while the cube is followed, 0.6 to 0.7 once it is lost (the tracker bends its
/// moves to what it reads, so a lost cube still fits in part)
pub const LOST_BELOW: f64 = 0.7;
/// Seconds it must stay below before it is said
pub const LOST_AFTER: f64 = 2.0;

/// How well what the camera reads fits the cube followed, smoothed over the frames read: a cube that was
/// turned without being seen, or that did not start as told, keeps fitting badly
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Agreement {
    level: Option<f64>,
    low_since: Option<f64>,
}

impl Agreement {
    /// One frame: the share of its stickers that fit (None: none read), at time `t` (seconds)
    pub fn update(&mut self, fit: Option<f64>, t: f64) {
        let Some(fit) = fit else { return };
        let level = self.level.map_or(fit, |level| level + (fit - level) * 0.05);
        self.level = Some(level);
        self.low_since = if level < LOST_BELOW { self.low_since.or(Some(t)) } else { None };
    }

    pub fn level(&self) -> Option<f64> {
        self.level
    }

    /// The cube followed is no longer the one in front of the camera
    pub fn lost(&self, t: f64) -> bool {
        self.low_since.is_some_and(|since| t - since >= LOST_AFTER)
    }

    pub fn reset(&mut self) {
        *self = Agreement::default();
    }
}

/// The moves written once the last one is taken back, and the cube's state then; None: none to take back
pub fn take_back(start: &State, written: &[usize]) -> Option<(Vec<usize>, State)> {
    let kept = written.split_last()?.1.to_vec();
    let state = kept.iter().fold(*start, |state, &m| apply_move(&state, m));
    Some((kept, state))
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
    /// Share of the stickers read that fit the cube followed, smoothed; whether the two no longer match
    fit: Option<f64>,
    lost: bool,
    source: String,
    device: String,
    note: Option<String>,
    dropped: u64,
}

pub(crate) struct Frames {
    camera: Option<Camera>,
    video: Option<(VideoReader, Instant)>,
}

impl Frames {
    pub(crate) fn open(source: &Source) -> Result<(Frames, String)> {
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
    pub(crate) fn next(&mut self) -> Result<Option<Image>> {
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

    pub(crate) fn dropped(&self) -> u64 {
        self.camera.as_ref().map_or(0, Camera::dropped)
    }
}

/// The frame shrunk for the window: one pixel in so many, flipped left to right when `mirror` (a camera facing
/// its user is watched as a mirror)
pub(crate) fn preview(image: &Image, mirror: bool) -> (Vec<u8>, usize, usize) {
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

/// What the window asks of the reading thread
#[derive(Default)]
struct Asked {
    reset: AtomicBool,
    undo: AtomicBool,
    stop: AtomicBool,
}

fn reading(options: &LiveOptions, shared: &Mutex<Shared>, asked: &Asked, ctx: &egui::Context) -> Result<()> {
    let say = |status: &str| {
        shared.lock().unwrap().status = status.to_owned();
        ctx.request_repaint();
    };
    let start = match &options.start {
        Some(scramble) => {
            crate::colours::scramble(scramble).context("the scramble the cube starts from")?;
            state_of(&Cube::solved().apply(scramble))
        }
        None => solved_state(),
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
    // As the site's live mode: the detector looks at one frame in 3 while the reader reads the cube
    let mut follower = CubeFollower::new(detector, 3);
    let mut tracker = DelayedTracker::new(start);
    // The moves written before the tracker's own: those kept when one was taken back
    let mut kept: Vec<usize> = Vec::new();
    let mut agreement = Agreement::default();
    let (began, mut last) = (Instant::now(), Instant::now());
    while !asked.stop.load(Ordering::Relaxed) {
        let Some(image) = frames.next()? else { continue };
        if asked.reset.swap(false, Ordering::Relaxed) {
            tracker.reset(start);
            kept.clear();
            agreement.reset();
            follower.boxes.reset();
        }
        if asked.undo.swap(false, Ordering::Relaxed) {
            let written: Vec<usize> = kept.iter().chain(tracker.confirmed()).copied().collect();
            if let Some((fewer, state)) = take_back(&start, &written) {
                tracker.reset(state);
                kept = fewer;
                agreement.reset();
            }
        }
        let mirror = options.mirror.load(Ordering::Relaxed);
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
        let now = began.elapsed().as_secs_f64();
        // A layer mid-turn is read between two states: it says nothing of how well the cube is followed
        if !turning {
            agreement.update(crate::label::fit(tracker.state(), &faces), now);
        }
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
        shared.fit = agreement.level();
        shared.lost = agreement.lost(now);
        shared.confirmed = kept.iter().chain(tracker.confirmed()).copied().collect();
        shared.pending = tracker.pending().to_vec();
        shared.fps += (1.0 / between - shared.fps) * 0.1;
        shared.read_ms += (read_ms - shared.read_ms) * 0.1;
        shared.dropped = frames.dropped();
        drop(shared);
        ctx.request_repaint();
    }
    Ok(())
}

/// A reading thread asked to stop is waited for a moment, so that the camera is free for the next screen
pub(crate) fn wait_for(worker: Option<std::thread::JoinHandle<()>>) {
    let Some(worker) = worker else { return };
    let since = Instant::now();
    while !worker.is_finished() && since.elapsed() < Duration::from_millis(1500) {
        std::thread::sleep(Duration::from_millis(10));
    }
    if worker.is_finished() {
        let _ = worker.join();
    }
}

/// The camera's picture drawn as large as fits above `below` points of text, its corners rounded; where it is
pub(crate) fn picture(ui: &mut egui::Ui, texture: &egui::TextureHandle, below: f32) -> egui::Rect {
    let size = texture.size_vec2();
    let scale = (ui.available_width() / size.x).min((ui.available_height() - below) / size.y).max(0.1);
    let image = egui::Image::new(egui::load::SizedTexture::new(texture.id(), size * scale)).corner_radius(10.0);
    ui.add(image).rect
}

/// The square the reader reads, drawn on the picture
pub(crate) fn frame_box(ui: &egui::Ui, rect: egui::Rect, [x, y, w, h]: [f32; 4], colour: egui::Color32) {
    let found = egui::Rect::from_min_size(
        rect.min + egui::vec2(x * rect.width(), y * rect.height()),
        egui::vec2(w * rect.width(), h * rect.height()),
    );
    ui.painter_at(rect).rect_stroke(found, 8.0, egui::Stroke::new(2.5, colour), egui::StrokeKind::Inside);
}

/// What is going on, as a dot of its colour and a sentence
pub(crate) fn state_line(ui: &mut egui::Ui, colour: egui::Color32, text: &str) {
    ui.horizontal(|ui| {
        let (dot, _) = ui.allocate_exact_size(egui::Vec2::splat(10.0), egui::Sense::hover());
        ui.painter().circle_filled(dot.center(), 5.0, colour);
        ui.label(egui::RichText::new(text).size(16.0));
    });
}

pub struct LiveScreen {
    shared: Arc<Mutex<Shared>>,
    asked: Arc<Asked>,
    worker: Option<std::thread::JoinHandle<()>>,
    mirror: Arc<AtomicBool>,
    /// Whether the frames come from a camera (a video is not shown as a mirror)
    camera: bool,
    start: Cube,
    cube: CubeView,
    /// Moves written already handed to the 3D cube
    played: usize,
    picture: Option<egui::TextureHandle>,
    snapshot: Snapshot,
}

impl LiveScreen {
    pub fn new(options: LiveOptions, ctx: &egui::Context) -> Self {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let asked = Arc::new(Asked::default());
        let snapshot = options.snapshot.clone();
        let camera = matches!(options.source, Source::Camera(_));
        let mirror = options.mirror.clone();
        if !camera {
            mirror.store(false, Ordering::Relaxed);
        }
        // A scramble that cannot be read is told by the reading thread
        let start =
            options.start.as_deref().filter(|s| crate::colours::scramble(s).is_ok()).map_or(Cube::solved(), |s| Cube::solved().apply(s));
        let worker = {
            let (shared, asked, ctx) = (shared.clone(), asked.clone(), ctx.clone());
            std::thread::spawn(move || {
                if let Err(e) = reading(&options, &shared, &asked, &ctx) {
                    shared.lock().unwrap().error = Some(format!("{e:#}"));
                    ctx.request_repaint();
                }
            })
        };
        LiveScreen {
            shared,
            asked,
            worker: Some(worker),
            mirror,
            camera,
            cube: CubeView::new(start.clone()),
            start,
            played: 0,
            picture: None,
            snapshot,
        }
    }

    fn start_again(&mut self) {
        self.asked.reset.store(true, Ordering::Relaxed);
        self.cube.set(self.start.clone());
        self.played = 0;
        let mut shared = self.shared.lock().unwrap();
        shared.confirmed.clear();
        shared.pending.clear();
        shared.lost = false;
    }
}

impl Screen for LiveScreen {
    fn show(&mut self, ctx: &egui::Context) -> Next {
        self.snapshot.tick(ctx);
        let (again, back, escape) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::R) || i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::Backspace),
                i.key_pressed(egui::Key::Escape),
            )
        });
        if again {
            self.start_again();
        }
        if back {
            self.asked.undo.store(true, Ordering::Relaxed);
        }
        let (status, error, boxed, faces, turning, confirmed, pending, fps, read_ms, unread, fit, lost, source, device, note, dropped) = {
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
                s.fit,
                s.lost,
                s.source.clone(),
                s.device.clone(),
                s.note.clone(),
                s.dropped,
            )
        };
        // The moves written since the last frame go to the 3D cube; fewer than played: a new start, or one taken back
        if confirmed.len() < self.played {
            let made: Vec<&str> = confirmed.iter().map(|&m| MOVES[m]).collect();
            self.cube.set(self.start.apply(&made.join(" ")));
        } else {
            for &m in &confirmed[self.played..] {
                self.cube.play(MOVES[m]);
            }
        }
        self.played = confirmed.len();

        let mut mirror = self.mirror.load(Ordering::Relaxed);
        let camera = self.camera;
        let home = theme::top_bar(ctx, "Live", theme::LIVE, |ui| {
            if camera && ui.toggle_value(&mut mirror, "Mirror").on_hover_text("Show the camera as a mirror does").changed() {
                self.mirror.store(mirror, Ordering::Relaxed);
            }
        });
        let mut facts = Vec::new();
        if !device.is_empty() {
            facts.push(format!("Models on {device}"));
            facts.push(format!("{fps:.0} frames a second, {read_ms:.0} ms each"));
            facts.push(source.clone());
            if dropped > 0 {
                facts.push(format!("{dropped} frames skipped"));
            }
        }
        facts.extend(note.clone());
        theme::foot(ctx, &facts);

        let side = egui::Frame::new().fill(theme::PLASTIC).inner_margin(egui::Margin { left: 6, right: 18, top: 4, bottom: 8 });
        egui::SidePanel::right("cube")
            .frame(side)
            .show_separator_line(false)
            .resizable(false)
            .exact_width((ctx.screen_rect().width() * 0.42).clamp(340.0, 580.0))
            .show(ctx, |ui| {
                let size = ui.available_width().min(ui.available_height() * 0.5);
                ui.vertical_centered(|ui| self.cube.show(ui, size));
                ui.horizontal(|ui| {
                    ui.label(theme::strong(confirmed.len().to_string(), 30.0));
                    ui.label(theme::muted(if confirmed.len() == 1 { "move" } else { "moves" }));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Start again").on_hover_text("Your cube is solved, white on top, green in front (R)").clicked() {
                            self.start_again();
                        }
                        let take = ui.add_enabled(!confirmed.is_empty(), egui::Button::new("Take back"));
                        if take.on_hover_text("Take the last move written back, when you did not make it (Backspace)").clicked() {
                            self.asked.undo.store(true, Ordering::Relaxed);
                        }
                    });
                });
                if lost {
                    theme::banner(ui, theme::WARNING, "The cube on screen may no longer match yours", |ui| {
                        ui.add(
                            egui::Label::new(
                                "If it does not: solve your cube, hold it white on top and green in front, then press Start again.",
                            )
                            .wrap(),
                        );
                    });
                }
                ui.add_space(2.0);
                egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
                        for (i, &m) in confirmed.iter().enumerate() {
                            theme::chip(ui, MOVES[m], if i + 1 == confirmed.len() { Chip::Now } else { Chip::Done });
                        }
                        // Seen, not settled yet: they may still change
                        for &m in &pending {
                            theme::chip(ui, MOVES[m], Chip::Unsettled);
                        }
                    });
                    if confirmed.is_empty() && pending.is_empty() {
                        ui.label(theme::muted(
                            "Start from a solved cube, white on top and green in front. Each move you make is written here.",
                        ));
                    }
                });
            });
        let centre = egui::Frame::new().fill(theme::PLASTIC).inner_margin(egui::Margin { left: 18, right: 12, top: 4, bottom: 8 });
        egui::CentralPanel::default().frame(centre).show(ctx, |ui| {
            if let Some(error) = &error {
                theme::failure(ui, "The live mode stopped", error);
                return;
            }
            let Some(texture) = &self.picture else {
                ui.centered_and_justified(|ui| {
                    ui.label(theme::title(if status.is_empty() { "Waiting for the camera…" } else { &status }))
                });
                return;
            };
            let rect = picture(ui, texture, 40.0);
            if let Some(boxed) = boxed {
                frame_box(ui, rect, boxed, if faces > 0 { theme::GOOD } else { theme::WARNING });
            }
            ui.add_space(4.0);
            let (colour, what) = match (boxed.is_some(), faces, turning) {
                (false, _, _) => (theme::FAINT, "No cube in sight".to_owned()),
                (true, 0, true) => (theme::BLUE, "A layer is turning".to_owned()),
                (true, 0, false) if unread => {
                    (theme::WARNING, "The cube is seen but not read: more light, or hold it still a moment".to_owned())
                }
                (true, 0, false) => (theme::WARNING, "Cube in sight".to_owned()),
                (true, n, _) => (theme::GOOD, format!("{n} face{} read", if n == 1 { "" } else { "s" })),
            };
            ui.horizontal(|ui| {
                state_line(ui, colour, &what);
                if let Some(fit) = fit {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(theme::faint(format!("{:.0} % of the stickers read fit the cube on screen", fit * 100.0)));
                    });
                }
            });
        });
        if home || escape { Next::Home } else { Next::Stay }
    }

    fn leave(&mut self) {
        self.asked.stop.store(true, Ordering::Relaxed);
        wait_for(self.worker.take());
    }
}
