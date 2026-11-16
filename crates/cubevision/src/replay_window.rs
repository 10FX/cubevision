//! The window of a solve found in a video: the 3D cube playing it, a bar to move through it, what was measured

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cube_capture::{Image, Latest, VideoOptions, VideoReader};
use cube_decode::pipeline::VisionResult;
use cube_view::CubeView;
use eframe::egui;

use cube_decode::reconstruct::Phase;

use crate::replay::{Change, Playback, Replay, Stage, stages, written};
use crate::theme::{self, Chip};
use crate::window::{Next, Screen, Snapshot};

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

/// The colour a stage is drawn in: the cross white as most cubers make it, the pairs green, the last layer's
/// two steps yellow then orange
fn stage_colour(phase: Phase) -> egui::Color32 {
    match phase {
        Phase::Cross => theme::WHITE,
        Phase::F2l1 | Phase::F2l2 | Phase::F2l3 | Phase::F2l4 => theme::GREEN,
        Phase::Oll => theme::YELLOW,
        Phase::Pll => theme::ORANGE,
    }
}

/// The solve as a bar: its stages one after the other in their colours, a mark at each turn, a line at the
/// time shown. The time asked for by a click or a drag.
fn timeline(ui: &mut egui::Ui, stages: &[Stage], turns: &[f64], (from, to): (f64, f64), clock: f64) -> Option<f64> {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 46.0), egui::Sense::click_and_drag());
    let bar = egui::Rect::from_min_max(rect.min + egui::vec2(0.0, 4.0), egui::pos2(rect.right(), rect.top() + 26.0));
    let x_of = |t: f64| bar.left() + ((t - from) / (to - from).max(1.0)).clamp(0.0, 1.0) as f32 * bar.width();
    let painter = ui.painter_at(rect);
    painter.rect_filled(bar, 6.0, theme::RAISED);
    for stage in stages {
        let (left, right) = (x_of(stage.from) + 1.0, x_of(stage.to) - 1.0);
        if right <= left {
            continue;
        }
        let colour = stage_colour(stage.phase);
        let whole = egui::Rect::from_min_max(egui::pos2(left, bar.top()), egui::pos2(right, bar.bottom()));
        painter.rect_filled(whole, 4.0, colour.gamma_multiply(0.32));
        let reached = x_of(clock).min(right);
        if reached > left {
            painter.rect_filled(egui::Rect::from_min_max(whole.min, egui::pos2(reached, bar.bottom())), 4.0, colour);
        }
        let now = clock >= stage.from && clock < stage.to;
        let name = painter.layout_no_wrap(
            stage.name().to_owned(),
            egui::FontId::new(12.5, theme::medium()),
            if now { theme::TEXT } else { theme::FAINT },
        );
        if name.size().x + 4.0 <= right - left {
            painter.galley(egui::pos2(left + 2.0, bar.bottom() + 3.0), name, theme::FAINT);
        }
    }
    for &t in turns {
        let x = x_of(t);
        painter.line_segment(
            [egui::pos2(x, bar.bottom() - 6.0), egui::pos2(x, bar.bottom())],
            egui::Stroke::new(1.0, theme::PLASTIC.gamma_multiply(0.7)),
        );
    }
    let x = x_of(clock);
    painter.line_segment([egui::pos2(x, bar.top() - 4.0), egui::pos2(x, bar.bottom() + 4.0)], egui::Stroke::new(2.0, egui::Color32::WHITE));
    let time_at = |x: f32| from + f64::from(((x - bar.left()) / bar.width()).clamp(0.0, 1.0)) * (to - from);
    let asked = (response.clicked() || response.dragged()).then(|| response.interact_pointer_pos()).flatten().map(|at| time_at(at.x));
    if let Some(at) = response.hover_pos() {
        let t = time_at(at.x);
        if let Some(stage) = stages.iter().find(|stage| t >= stage.from && t < stage.to) {
            response.on_hover_text_at_pointer(format!(
                "{}: {:.2} s, {} moves",
                stage.name(),
                (stage.to - stage.from) / 1000.0,
                stage.moves
            ));
        }
    }
    asked
}

pub struct ReplayScreen {
    shown: Shown,
    /// The time of the video the picture should show, its frames as they come, the one on screen
    wanted: Arc<Mutex<f64>>,
    frames: Latest<Image>,
    picture: Option<egui::TextureHandle>,
    stop: Arc<AtomicBool>,
    replay: Replay,
    stages: Vec<Stage>,
    cube: CubeView,
    /// Where the replay stands
    play: Playback,
    last: Option<f64>,
    /// When the solve was last copied, in the window's clock: said for a moment
    copied: Option<f64>,
}

impl ReplayScreen {
    pub fn new(shown: Shown, ctx: &egui::Context) -> Self {
        let replay = Replay::new(&shown.scramble, &shown.result);
        let cube = CubeView::new(replay.cube_at(0));
        let clock = replay.span().0;
        let (wanted, frames, stop) = (Arc::new(Mutex::new(clock)), Latest::new(), Arc::new(AtomicBool::new(false)));
        {
            let (video, ffmpeg, wanted, frames, stop, ctx) =
                (shown.video.clone(), shown.ffmpeg.clone(), wanted.clone(), frames.clone(), stop.clone(), ctx.clone());
            std::thread::spawn(move || pictures(video, ffmpeg, wanted, frames, stop, ctx));
        }
        let stages = stages(&shown.result.reconstruction);
        ReplayScreen {
            shown,
            wanted,
            frames,
            picture: None,
            stop,
            play: Playback::new(&replay),
            replay,
            stages,
            cube,
            last: None,
            copied: None,
        }
    }

    /// The cube does what the playback's move asks
    fn follow(&mut self, change: Change) {
        match change {
            Change::Nothing => {}
            Change::Play(turns) => self.replay.turns[turns].iter().for_each(|t| self.cube.play(t)),
            Change::Set(made) => self.cube.set(self.replay.cube_at(made)),
        }
    }

    /// The cube set to a time at once
    fn seek(&mut self, clock: f64) {
        let change = self.play.seek(&self.replay, clock);
        self.follow(change);
    }

    /// One move forward (played) or back (at once)
    fn step(&mut self, forward: bool) {
        let change = self.play.step(&self.replay, forward);
        self.follow(change);
    }

    fn tick(&mut self, now: f64) {
        let elapsed = self.last.map_or(0.0, |last| now - last);
        self.last = Some(now);
        let change = self.play.tick(&self.replay, elapsed);
        self.follow(change);
    }
}

impl Screen for ReplayScreen {
    fn leave(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    fn show(&mut self, ctx: &egui::Context) -> Next {
        self.shown.snapshot.tick(ctx);
        let now = ctx.input(|i| i.time);
        self.tick(now);
        self.cube.held = self.replay.orientation(self.play.clock);
        *self.wanted.lock().unwrap() = self.play.clock;
        if let Some(image) = self.frames.take() {
            let picture = egui::ColorImage::from_rgba_unmultiplied([image.width, image.height], &image.data);
            match &mut self.picture {
                Some(texture) => texture.set(picture, egui::TextureOptions::LINEAR),
                None => self.picture = Some(ctx.load_texture("video", picture, egui::TextureOptions::LINEAR)),
            }
        }
        if self.play.playing {
            ctx.request_repaint();
        }
        let (space, left, right, escape) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Escape),
            )
        });
        if left || right {
            self.step(right);
        }
        let solve_start = self.shown.result.reconstruction.start;
        let (from, to) = self.replay.span();

        let title = self.shown.title.clone();
        let home = theme::top_bar(ctx, "Video", theme::VIDEO, |ui| {
            ui.label(theme::faint(title));
        });
        theme::foot(
            ctx,
            &[
                format!("Models on {}", self.shown.device),
                format!("Frames read: {} of {}", self.shown.result.read_frames, self.shown.result.frames),
                format!("Stickers read that fit the moves: {:.1} %", self.shown.result.agreement * 100.0),
                if self.shown.mirrored { "The video is mirrored".to_owned() } else { String::new() },
            ],
        );

        let bar = egui::Frame::new().fill(theme::PLASTIC).inner_margin(egui::Margin { left: 18, right: 18, top: 2, bottom: 6 });
        egui::TopBottomPanel::bottom("bar").frame(bar).show_separator_line(false).show(ctx, |ui| {
            if let Some(clock) = timeline(ui, &self.stages, &self.replay.times, (from, to), self.play.clock) {
                self.seek(clock);
            }
            ui.horizontal(|ui| {
                let restart = !self.play.playing && self.play.clock >= to;
                let label = if self.play.playing {
                    "Pause"
                } else if restart {
                    "Play again"
                } else {
                    "Play"
                };
                if theme::primary(ui, true, label, theme::VIDEO).on_hover_text("Space").clicked() || space {
                    if restart {
                        self.seek(from);
                    }
                    self.play.playing = !self.play.playing;
                }
                if ui.button("Back a move").on_hover_text("Left arrow").clicked() {
                    self.step(false);
                }
                if ui.button("On a move").on_hover_text("Right arrow").clicked() {
                    self.step(true);
                }
                ui.add_space(8.0);
                ui.label(theme::faint("Speed"));
                for speed in [0.25, 0.5, 1.0, 2.0] {
                    if ui.selectable_label(self.play.speed == speed, format!("×{speed}")).clicked() {
                        self.play.speed = speed;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let all = self.shown.result.reconstruction.time / 1000.0;
                    let at = ((self.play.clock - solve_start).max(0.0) / 1000.0).min(all);
                    ui.label(theme::muted(format!("of {all:.2} s")));
                    ui.label(theme::strong(format!("{at:.2}"), 18.0));
                });
            });
        });

        let mut stage_asked = None;
        let side = egui::Frame::new().fill(theme::PLASTIC).inner_margin(egui::Margin { left: 6, right: 18, top: 4, bottom: 4 });
        egui::SidePanel::right("facts")
            .frame(side)
            .show_separator_line(false)
            .resizable(false)
            .exact_width((ctx.screen_rect().width() * 0.36).clamp(340.0, 520.0))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let r = &self.shown.result;
                    let rec = &r.reconstruction;
                    ui.horizontal(|ui| {
                        ui.label(theme::strong(format!("{:.2}", rec.time / 1000.0), 34.0));
                        ui.label(theme::muted("seconds"));
                    });
                    ui.label(theme::muted(format!("{} moves, {:.2} turns a second", rec.moves, rec.tps)));
                    if let Some(warning) = &self.shown.warning {
                        theme::banner(ui, theme::WARNING, "Not followed to the end", |ui| {
                            ui.add(egui::Label::new(warning.as_str()).wrap());
                        });
                    }
                    ui.add_space(6.0);
                    let mut go = None;
                    theme::card(ui, None, |ui| {
                        ui.set_width(ui.available_width());
                        egui::Grid::new("stages").num_columns(4).spacing([14.0, 8.0]).show(ui, |ui| {
                            for (stage, split) in self.stages.iter().zip(&rec.splits) {
                                let now = self.play.clock >= stage.from && self.play.clock < stage.to;
                                ui.horizontal(|ui| {
                                    let (sticker, _) = ui.allocate_exact_size(egui::Vec2::splat(10.0), egui::Sense::hover());
                                    ui.painter().rect_filled(sticker, 3.0, stage_colour(stage.phase));
                                    let name = egui::RichText::new(stage.name()).color(if now { theme::TEXT } else { theme::MUTED });
                                    if ui
                                        .add(egui::Label::new(name).sense(egui::Sense::click()))
                                        .on_hover_text("Go to this stage")
                                        .clicked()
                                    {
                                        go = Some(stage.from);
                                    }
                                });
                                ui.label(format!("{:.2} s", split.duration / 1000.0));
                                ui.label(theme::muted(format!("{} moves", split.moves)));
                                ui.add(egui::Label::new(theme::faint(split.alg.as_str())).wrap());
                                ui.end_row();
                            }
                        });
                    });
                    stage_asked = go;
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.button("Copy the solve").on_hover_text("The scramble and each stage's moves, as text").clicked() {
                            ctx.copy_text(written(&self.shown.scramble, rec));
                            self.copied = Some(now);
                        }
                        if self.copied.is_some_and(|at| now - at < 2.0) {
                            ui.label(egui::RichText::new("Copied").color(theme::GOOD));
                            ctx.request_repaint();
                        }
                    });
                    ui.add_space(6.0);
                    ui.label(theme::faint("Scramble"));
                    ui.add(egui::Label::new(theme::muted(self.shown.scramble.as_str())).wrap());
                    ui.add_space(6.0);
                    if let Some(orientation) = &r.orientation {
                        ui.label(theme::faint(format!("Rotations of the whole cube: {}", orientation.rotations)));
                    }
                    let (wide, slices) = self.replay.written.iter().flatten().fold((0, 0), |(wide, slices), t| {
                        (
                            wide + (t.contains('w') || t.starts_with(|c: char| c.is_ascii_lowercase() && !"xyz".contains(c))) as usize,
                            slices + t.starts_with(['M', 'E', 'S']) as usize,
                        )
                    });
                    ui.label(theme::faint(format!("Wide moves: {wide}; slices: {slices}")));
                    if !r.gaps.is_empty() {
                        let at: Vec<String> = r.gaps.iter().map(|g| format!("{:.1} s", g / 1000.0)).collect();
                        ui.label(theme::faint(format!("Turns not followed around: {}", at.join(", "))));
                    }
                });
            });

        if let Some(clock) = stage_asked {
            self.seek(clock);
        }
        let centre = egui::Frame::new().fill(theme::PLASTIC).inner_margin(egui::Margin { left: 18, right: 12, top: 4, bottom: 4 });
        egui::CentralPanel::default().frame(centre).show(ctx, |ui| {
            // The video and the cube side by side, the same height
            let height = (ui.available_height() * 0.6).min(ui.available_width() * 0.5);
            ui.horizontal(|ui| {
                if let Some(picture) = &self.picture {
                    let size = picture.size_vec2();
                    let scale = (height / size.y).min((ui.available_width() - height) / size.x).max(0.05);
                    ui.add(egui::Image::new(egui::load::SizedTexture::new(picture.id(), size * scale)).corner_radius(10.0));
                } else if let Some(error) = self.frames.error() {
                    ui.add_sized(
                        [ui.available_width() - height, height],
                        egui::Label::new(theme::muted(format!("The video cannot be shown: {error}"))).wrap(),
                    );
                }
                self.cube.show(ui, height.min(ui.available_width()));
            });
            ui.add_space(4.0);
            let mut go = None;
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
                    for (i, moves) in self.replay.written.iter().enumerate() {
                        let kind = match i + 1 {
                            n if n == self.play.made => Chip::Now,
                            n if n < self.play.made => Chip::Done,
                            _ => Chip::Ahead,
                        };
                        for token in moves {
                            if theme::chip(ui, token, kind).on_hover_text("Go to this move").clicked() {
                                go = Some(i + 1);
                            }
                        }
                    }
                });
            });
            if let Some(made) = go {
                self.play.playing = false;
                self.seek(self.replay.time_of(made));
            }
        });
        if home || escape { Next::Home } else { Next::Stay }
    }
}
