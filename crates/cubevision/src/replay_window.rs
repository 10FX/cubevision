//! The window of a solve found in a video: the 3D cube playing it, a bar to move through it, what was measured

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use cube_capture::{Image, Latest, VideoOptions, VideoReader};
use cube_decode::pipeline::VisionResult;
use cube_view::CubeView;
use eframe::egui;

use crate::replay::Replay;
use crate::window::{Snapshot, token};

/// What the window shows besides the result itself
pub struct Shown {
    /// The video, shown next to the cube, and the ffmpeg that reads it
    pub video: PathBuf,
    pub ffmpeg: String,
    pub title: String,
    pub scramble: String,
    pub device: String,
    pub mirrored: bool,
    pub warning: Option<String>,
    pub result: VisionResult,
    pub snapshot: Snapshot,
}

/// Widest picture of the video shown
const PICTURE: u32 = 640;

/// The video's frames for the window, in a thread of its own: it keeps to the replay's clock (the time wanted,
/// ms), reading on while the clock moves forward and starting again from there when it jumps
fn pictures(video: PathBuf, ffmpeg: String, wanted: Arc<Mutex<f64>>, out: Latest<Image>, stop: Arc<AtomicBool>, ctx: egui::Context) {
    let mut reader: Option<VideoReader> = None;
    // The time of the last frame read, and whether the video ended there
    let (mut at, mut ended) = (f64::NEG_INFINITY, false);
    while !stop.load(Ordering::Relaxed) {
        let target = *wanted.lock().unwrap();
        // Behind the last frame, or far ahead of it: read again from there (a little before, to show a frame at once)
        if reader.is_none() || target < at - 60.0 || target > at + 1500.0 {
            let options = VideoOptions { ffmpeg: ffmpeg.clone(), max_width: Some(PICTURE), start: (target - 40.0).max(0.0) / 1000.0 };
            match VideoReader::open(&video, &options) {
                Ok(opened) => reader = Some(opened),
                Err(e) => return out.fail(format!("{e:#}")),
            }
            (at, ended) = (target - 41.0, false);
        }
        if ended || at >= target {
            std::thread::sleep(Duration::from_millis(4));
            continue;
        }
        match reader.as_mut().unwrap().next_image() {
            Ok(Some(image)) => {
                at = image.t;
                // The frames on the way to the time wanted are not shown
                if image.t >= target - 45.0 {
                    out.put(image);
                    ctx.request_repaint();
                }
            }
            // The end of the video, or a start past it: the last picture stays
            Ok(None) | Err(_) => ended = true,
        }
    }
}

struct ReplayApp {
    shown: Shown,
    /// The time of the video the picture should show, its frames as they come, the one on screen
    wanted: Arc<Mutex<f64>>,
    frames: Latest<Image>,
    picture: Option<egui::TextureHandle>,
    stop: Arc<AtomicBool>,
    replay: Replay,
    cube: CubeView,
    /// The video's time now, ms, and the moves the cube has been handed
    clock: f64,
    made: usize,
    playing: bool,
    speed: f64,
    last: Option<f64>,
}

impl ReplayApp {
    fn new(shown: Shown, ctx: &egui::Context) -> Self {
        let replay = Replay::new(&shown.scramble, &shown.result);
        let cube = CubeView::new(replay.cube_at(0));
        let clock = replay.span().0;
        let (wanted, frames, stop) = (Arc::new(Mutex::new(clock)), Latest::new(), Arc::new(AtomicBool::new(false)));
        {
            let (video, ffmpeg, wanted, frames, stop, ctx) =
                (shown.video.clone(), shown.ffmpeg.clone(), wanted.clone(), frames.clone(), stop.clone(), ctx.clone());
            std::thread::spawn(move || pictures(video, ffmpeg, wanted, frames, stop, ctx));
        }
        ReplayApp { shown, wanted, frames, picture: None, stop, replay, cube, clock, made: 0, playing: true, speed: 1.0, last: None }
    }

    /// The cube set to a time at once
    fn seek(&mut self, clock: f64) {
        let (from, to) = self.replay.span();
        self.clock = clock.clamp(from, to);
        self.made = self.replay.moves_at(self.clock);
        self.cube.set(self.replay.cube_at(self.made));
    }

    /// One move forward (played) or back (at once)
    fn step(&mut self, forward: bool) {
        self.playing = false;
        if forward && self.made < self.replay.len() {
            self.cube.play(&self.replay.turns[self.made]);
            self.made += 1;
            self.clock = self.replay.time_of(self.made);
        } else if !forward && self.made > 0 {
            self.seek(self.replay.time_of(self.made - 1));
            if self.made > 0 {
                // Just after the move before, not on the one undone
                self.clock = self.replay.time_of(self.made);
            }
        }
    }

    fn tick(&mut self, now: f64) {
        let elapsed = self.last.map_or(0.0, |last| now - last);
        self.last = Some(now);
        if !self.playing {
            return;
        }
        self.clock += elapsed * 1000.0 * self.speed;
        let due = self.replay.moves_at(self.clock);
        for turn in &self.replay.turns[self.made.min(due)..due] {
            self.cube.play(turn);
        }
        self.made = self.made.max(due);
        if self.clock >= self.replay.span().1 {
            self.playing = false;
        }
    }
}

impl eframe::App for ReplayApp {
    fn on_exit(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.shown.snapshot.tick(ctx);
        self.tick(ctx.input(|i| i.time));
        self.cube.held = self.replay.orientation(self.clock);
        *self.wanted.lock().unwrap() = self.clock;
        if let Some(image) = self.frames.take() {
            let picture = egui::ColorImage::from_rgba_unmultiplied([image.width, image.height], &image.data);
            match &mut self.picture {
                Some(texture) => texture.set(picture, egui::TextureOptions::LINEAR),
                None => self.picture = Some(ctx.load_texture("video", picture, egui::TextureOptions::LINEAR)),
            }
        }
        if self.playing {
            ctx.request_repaint();
        }
        let (space, left, right) =
            ctx.input(|i| (i.key_pressed(egui::Key::Space), i.key_pressed(egui::Key::ArrowLeft), i.key_pressed(egui::Key::ArrowRight)));
        if left || right {
            self.step(right);
        }
        let solve_start = self.shown.result.reconstruction.start;
        let (from, to) = self.replay.span();

        egui::TopBottomPanel::bottom("bar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let restart = !self.playing && self.clock >= to;
                if ui.button(if self.playing { "Pause" } else { "Play" }).clicked() || space {
                    if restart {
                        self.seek(from);
                    }
                    self.playing = !self.playing;
                }
                if ui.button("◀ move").clicked() {
                    self.step(false);
                }
                if ui.button("move ▶").clicked() {
                    self.step(true);
                }
                for speed in [0.25, 0.5, 1.0] {
                    if ui.selectable_label(self.speed == speed, format!("×{speed}")).clicked() {
                        self.speed = speed;
                    }
                }
                let mut clock = self.clock;
                ui.spacing_mut().slider_width = (ui.available_width() - 90.0).max(120.0);
                if ui.add(egui::Slider::new(&mut clock, from..=to).show_value(false)).changed() {
                    self.seek(clock);
                }
                ui.label(format!("{:.2} s", (self.clock - solve_start).max(0.0) / 1000.0));
            });
            ui.add_space(4.0);
        });
        egui::SidePanel::right("facts").resizable(false).exact_width((ctx.screen_rect().width() * 0.44).clamp(340.0, 620.0)).show(
            ctx,
            |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(6.0);
                    let r = &self.shown.result;
                    let rec = &r.reconstruction;
                    ui.heading(if r.solved { "Solved" } else { "Not followed to the end" });
                    if let Some(warning) = &self.shown.warning {
                        ui.label(egui::RichText::new(warning).color(egui::Color32::from_rgb(240, 200, 60)));
                    }
                    ui.label(format!("{} moves in {:.2} s: {:.2} turns a second", rec.moves, rec.time / 1000.0, rec.tps));
                    ui.add_space(6.0);
                    egui::Grid::new("steps").striped(true).num_columns(4).show(ui, |ui| {
                        for step in &rec.splits {
                            ui.label(format!("{:?}", step.phase).to_lowercase());
                            ui.label(format!("{:.2} s", step.duration / 1000.0));
                            ui.label(format!("{} moves", step.moves));
                            ui.add(egui::Label::new(egui::RichText::new(&step.alg).monospace()).wrap());
                            ui.end_row();
                        }
                    });
                    ui.add_space(8.0);
                    ui.label(format!("Stickers read that fit the moves: {:.1} %", r.agreement * 100.0));
                    ui.label(format!("Frames read: {} of {}", r.read_frames, r.frames));
                    if let Some(orientation) = &r.orientation {
                        ui.label(format!("Rotations of the whole cube: {}", orientation.rotations));
                    }
                    let (wide, slices) = self.replay.written.iter().flatten().fold((0, 0), |(wide, slices), t| {
                        (
                            wide + (t.contains('w') || t.starts_with(|c: char| c.is_ascii_lowercase() && !"xyz".contains(c))) as usize,
                            slices + t.starts_with(['M', 'E', 'S']) as usize,
                        )
                    });
                    ui.label(format!("Wide moves: {wide}; slices: {slices}"));
                    if !r.gaps.is_empty() {
                        let at: Vec<String> = r.gaps.iter().map(|g| format!("{:.1} s", g / 1000.0)).collect();
                        ui.label(format!("Turns not followed around: {}", at.join(", ")));
                    }
                    ui.label(format!(
                        "Models on {}{}",
                        self.shown.device,
                        if self.shown.mirrored { "; the video is mirrored" } else { "" }
                    ));
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("Scramble").strong());
                    ui.add(egui::Label::new(egui::RichText::new(&self.shown.scramble).monospace()).wrap());
                });
            },
        );
        egui::CentralPanel::default().show(ctx, |ui| {
            // The video and the cube side by side, the same height
            let height = (ui.available_height() * 0.62).min(ui.available_width() * 0.5);
            ui.horizontal(|ui| {
                if let Some(picture) = &self.picture {
                    let size = picture.size_vec2();
                    let scale = (height / size.y).min((ui.available_width() - height) / size.x).max(0.05);
                    let (rect, _) = ui.allocate_exact_size(size * scale, egui::Sense::hover());
                    ui.painter().image(
                        picture.id(),
                        rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                } else if let Some(error) = self.frames.error() {
                    ui.add_sized(
                        [ui.available_width() - height, height],
                        egui::Label::new(egui::RichText::new(format!("The video cannot be shown: {error}")).weak()).wrap(),
                    );
                }
                self.cube.show(ui, height.min(ui.available_width()));
            });
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 9.0;
                    for (i, moves) in self.replay.written.iter().enumerate() {
                        let colour = match i + 1 {
                            n if n == self.made => egui::Color32::from_rgb(80, 220, 120),
                            n if n < self.made => ui.visuals().strong_text_color(),
                            _ => ui.visuals().weak_text_color(),
                        };
                        moves.iter().for_each(|t| token(ui, t, colour));
                    }
                });
            });
        });
    }
}

/// Opens the replay's window; returns when it is closed
pub fn run(shown: Shown) -> Result<()> {
    let title = format!("cubevision: {}", shown.title);
    crate::window::run(&title, [1280.0, 720.0], |cc| Box::new(ReplayApp::new(shown, &cc.egui_ctx)))
}
