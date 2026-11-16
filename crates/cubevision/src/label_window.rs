//! The labelling mode's window and its recording: the camera on the left, the cube as it should be on the right,
//! the move to make in large. Each frame's crop of the cube is saved with what the session knows of it
//! (`label.rs`); the whole frames are only shown, never written.
//!
//! A session's folder: `images/000000.jpg`… (the reader's square around the cube, `SIZE` pixels) and
//! `log.jsonl`, one record a line: `session` (how it was recorded), `frame` (an image and its label), `press`,
//! `discard` (segments whose frames a move taken back made wrong), `sequence` (over: `ok` tells whether the
//! cube came back solved), `end`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use cube_core::Cube;
use cube_decode::state::MOVES;
use cube_infer::device::Device;
use cube_infer::follow::CubeFollower;
use cube_infer::geometry::square_crop;
use cube_infer::models::Models;
use cube_infer::output::{Reader, readings_from_output};
use cube_infer::{BoxF, Frame};
use cube_view::CubeView;
use eframe::egui;
use serde_json::{Value, json};

use crate::label::{Label, Pause, Phase, Session, fit, ms, state_text};
use crate::live::{Frames, Source, frame_box, picture, preview, state_line, wait_for};
use crate::theme;
use crate::window::Snapshot;
use crate::window::{Next, Screen};

/// Side of the crops saved: the renders' size
pub const SIZE: usize = 256;
/// Still frames are saved this far apart at least (seconds): frames closer are nearly the same picture
const STILL_GAP: f64 = 0.066;
/// Frames of a move are saved for this long after it is asked for (seconds): a user who went away fills nothing
const MOVE_CAP: f64 = 5.0;
const QUALITY: u8 = 92;

pub struct LabelOptions {
    pub source: Source,
    pub models: PathBuf,
    pub device: Option<Device>,
    /// The session's folder (made)
    pub out: PathBuf,
    pub seed: u64,
    /// Moves that scramble in a sequence
    pub length: usize,
    /// Where and how the session is filmed (the room, the light, the cube), kept in the log
    pub note: String,
    /// Space pressed by the program this long after each move is asked for (to try the mode unattended)
    pub auto: Option<f64>,
    pub snapshot: Snapshot,
}

/// The session's log, written by the reading thread and by the window
pub struct Log {
    file: std::fs::File,
}

impl Log {
    pub fn create(out: &Path) -> Result<Log> {
        std::fs::create_dir_all(out.join("images")).with_context(|| format!("cannot make the folder {}", out.display()))?;
        let path = out.join("log.jsonl");
        let file =
            std::fs::OpenOptions::new().create_new(true).write(true).open(&path).with_context(|| {
                format!("cannot start {}: a session is in this folder already? Give another with --out", path.display())
            })?;
        Ok(Log { file })
    }

    /// One record, on a line of its own, on the disk at once
    pub fn write(&mut self, record: &Value) -> Result<()> {
        writeln!(self.file, "{record}").context("cannot write the session's log")
    }
}

/// A crop saved as JPEG
pub fn save_jpeg(file: &Path, rgba: &[u8], size: usize) -> Result<()> {
    let encoder = jpeg_encoder::Encoder::new_file(file, QUALITY).with_context(|| format!("cannot write {}", file.display()))?;
    encoder
        .encode(rgba, size as u16, size as u16, jpeg_encoder::ColorType::Rgba)
        .with_context(|| format!("cannot write {}", file.display()))
}

/// A frame's record. `faces`, `present`, `turning`, `orientation`: what the reader made of the crop, to tell
/// later how the cube was turned and where a move began; `fit`: the share of its stickers that are as the state
/// has them (`fit_after`: as the state after the move asked for has them).
pub fn frame_record(image: &str, t: f64, label: &Label, boxed: &BoxF, frame: (usize, usize), read: &Value) -> Value {
    json!({
        "type": "frame",
        "image": image,
        "t": ms(t),
        "sequence": label.sequence,
        "step": label.step,
        "segment": label.segment,
        "phase": label.phase,
        "state": state_text(&label.state),
        "move": label.mv.map(|m| MOVES[m]),
        "box": [boxed.x.round(), boxed.y.round(), boxed.w.round(), boxed.h.round()],
        "frame": [frame.0, frame.1],
        "read": read,
    })
}

/// What the reading thread shows the window
#[derive(Default)]
struct Shared {
    status: String,
    error: Option<String>,
    preview: Option<(Vec<u8>, usize, usize)>,
    boxed: Option<[f32; 4]>,
    /// How well the reader's stickers fit the state, smoothed (None: nothing read lately)
    fit: Option<f32>,
    still: usize,
    moving: usize,
    fps: f32,
    source: String,
    device: String,
    note: Option<String>,
}

fn unix_seconds() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn recording(
    options: &LabelOptions,
    shared: &Mutex<Shared>,
    session: &Mutex<Session>,
    log: &Mutex<Log>,
    start: Instant,
    stop: &AtomicBool,
    ctx: &egui::Context,
) -> Result<()> {
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
    let device = models.device.name().to_owned();
    let Models { detector, mut reader, .. } = models;
    say("Opening the camera…");
    let (mut frames, name) = Frames::open(&options.source)?;
    shared.lock().unwrap().source = name.clone();
    say("");
    log.lock().unwrap().write(&json!({
        "type": "session",
        "version": env!("CARGO_PKG_VERSION"),
        "started": unix_seconds(),
        "seed": options.seed,
        "length": options.length,
        "still": crate::label::STILL,
        "size": SIZE,
        "camera": name,
        "device": device,
        "models": options.models.file_name().map(|n| n.to_string_lossy().into_owned()),
        "note": options.note,
    }))?;
    let mirror = matches!(options.source, Source::Camera(_));
    // Every frame looked at by the detector: the crop saved must be where the cube is, not where it was
    let mut follower = CubeFollower::new(detector, 1);
    let (mut saved, mut last_saved, mut last) = (0usize, f64::NEG_INFINITY, Instant::now());
    while !stop.load(Ordering::Relaxed) {
        let Some(image) = frames.next()? else { continue };
        let t = start.elapsed().as_secs_f64();
        let frame = Frame { data: &image.data, width: image.width, height: image.height };
        let boxed = follower.crop(&frame)?;
        let (label, events) = {
            let mut session = session.lock().unwrap();
            session.tick(t);
            (session.label(t), session.take_events())
        };
        for event in &events {
            log.lock().unwrap().write(event)?;
        }
        let mut agreement = None;
        let (mut still, mut moving) = (0, 0);
        if let Some(boxed) = &boxed {
            let out = reader.read(&[&square_crop(&frame, boxed, reader.size())])?.remove(0);
            // The reader's "turning" is not heeded: the session knows when a move is on its way
            let faces = readings_from_output(&out, reader.ordered(), true);
            follower.read(true);
            if let Some(label) = &label {
                agreement = fit(&label.state, &faces);
                let due = match label.phase {
                    "still" => t - last_saved >= STILL_GAP,
                    _ => label.age <= MOVE_CAP,
                };
                if due {
                    let image_name = format!("images/{saved:06}.jpg");
                    save_jpeg(&options.out.join(&image_name), &square_crop(&frame, boxed, SIZE), SIZE)?;
                    let after = label.mv.and_then(|m| fit(&cube_decode::state::apply_move(&label.state, m), &faces));
                    let read = json!({
                        "faces": faces.len(),
                        "present": out.present,
                        "turning": out.turning,
                        "orientation": out.orientation64(),
                        "fit": agreement,
                        "fit_after": after,
                    });
                    log.lock().unwrap().write(&frame_record(&image_name, t, label, boxed, (image.width, image.height), &read))?;
                    saved += 1;
                    last_saved = t;
                    if label.phase == "still" { still = 1 } else { moving = 1 }
                }
            }
        }
        let between = last.elapsed().as_secs_f32().max(1e-3);
        last = Instant::now();
        let mut shared = shared.lock().unwrap();
        shared.preview = Some(preview(&image, mirror));
        shared.boxed = boxed.map(|b: BoxF| {
            let (w, h) = (image.width as f64, image.height as f64);
            let x = if mirror { 1.0 - (b.x + b.w) / w } else { b.x / w };
            [x as f32, (b.y / h) as f32, (b.w / w) as f32, (b.h / h) as f32]
        });
        shared.fit = match (agreement, shared.fit) {
            (Some(a), Some(f)) => Some(f + (a as f32 - f) * 0.15),
            (Some(a), None) => Some(a as f32),
            // Paused, or nothing read: the old value says nothing of now
            (None, f) => f.filter(|_| label.is_some() && boxed.is_some()),
        };
        shared.still += still;
        shared.moving += moving;
        shared.fps += (1.0 / between - shared.fps) * 0.1;
        drop(shared);
        ctx.request_repaint();
    }
    Ok(())
}

/// The colours of the faces U R F D L B, as they are called
const FACE_NAMES: [&str; 6] = ["white", "red", "green", "yellow", "orange", "blue"];

/// A move in words: which face, which way (as seen facing that face)
pub fn in_words(mv: usize) -> String {
    let way = ["a quarter turn clockwise", "a half turn", "a quarter turn counter-clockwise"][mv % 3];
    format!("{} face, {way}", FACE_NAMES[mv / 3])
}

pub struct LabelScreen {
    shared: Arc<Mutex<Shared>>,
    session: Arc<Mutex<Session>>,
    log: Arc<Mutex<Log>>,
    start: Instant,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    /// The session's end is written
    ended: bool,
    cube: CubeView,
    /// The sequence and the step the 3D cube shows
    shown: (usize, usize),
    picture: Option<egui::TextureHandle>,
    out: PathBuf,
    validated: usize,
    auto: Option<f64>,
    snapshot: Snapshot,
}

impl LabelScreen {
    pub fn new(options: LabelOptions, log: Log, ctx: &egui::Context) -> Self {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let session = Arc::new(Mutex::new(Session::new(options.seed, options.length)));
        let log = Arc::new(Mutex::new(log));
        let stop = Arc::new(AtomicBool::new(false));
        let start = Instant::now();
        let (out, snapshot, auto) = (options.out.clone(), options.snapshot.clone(), options.auto);
        let worker = {
            let (shared, session, log, stop, ctx) = (shared.clone(), session.clone(), log.clone(), stop.clone(), ctx.clone());
            std::thread::spawn(move || {
                if let Err(e) = recording(&options, &shared, &session, &log, start, &stop, &ctx) {
                    shared.lock().unwrap().error = Some(format!("{e:#}"));
                    ctx.request_repaint();
                }
            })
        };
        LabelScreen {
            shared,
            session,
            log,
            start,
            stop,
            worker: Some(worker),
            ended: false,
            cube: CubeView::new(Cube::solved()),
            shown: (0, 0),
            picture: None,
            out,
            validated: 0,
            auto,
            snapshot,
        }
    }

    /// The 3D cube follows the session: a move made is played, anything else is set at once
    fn follow(&mut self, session: &Session) {
        let now = (session.sequence(), session.step());
        if now == self.shown {
            return;
        }
        if now.0 == self.shown.0 && now.1 == self.shown.1 + 1 {
            self.cube.play(MOVES[session.plan()[now.1 - 1]]);
        } else {
            let made: Vec<&str> = session.plan()[..now.1].iter().map(|&m| MOVES[m]).collect();
            self.cube.set(Cube::solved().apply(&made.join(" ")));
        }
        self.shown = now;
    }
}

impl Screen for LabelScreen {
    fn show(&mut self, ctx: &egui::Context) -> Next {
        self.snapshot.tick(ctx);
        let t = self.start.elapsed().as_secs_f64();
        let (space, undo, lost, quit) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::Backspace),
                i.key_pressed(egui::Key::X),
                i.key_pressed(egui::Key::Escape),
            )
        });
        let (phase, prompt, upcoming, step, total, sequence, progress) = {
            let handle = self.session.clone();
            let mut session = handle.lock().unwrap();
            let space = space || self.auto.is_some_and(|after| session.phase() != Phase::Still && session.age(t) >= after);
            if space {
                if session.phase() == Phase::Paused(Pause::Done) {
                    self.validated += 1;
                }
                session.space(t);
            }
            if undo {
                session.undo(t);
            }
            if lost {
                session.lost(t);
            }
            session.tick(t);
            let session = &*session;
            self.follow(session);
            let next = session.step() + 1;
            let upcoming: Vec<usize> = session.plan().get(next..(next + 3).min(session.plan().len())).unwrap_or(&[]).to_vec();
            (
                session.phase(),
                session.prompt(),
                upcoming,
                session.step(),
                session.plan().len(),
                session.sequence(),
                session.still_progress(t),
            )
        };
        let (status, error, boxed, agreement, still, moving, fps, source, device, note) = {
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
                s.fit,
                s.still,
                s.moving,
                s.fps,
                s.source.clone(),
                s.device.clone(),
                s.note.clone(),
            )
        };
        let recording = !matches!(phase, Phase::Paused(_));

        let home = theme::top_bar(ctx, "Label", theme::LABEL, |ui| {
            ui.label(theme::faint(format!("Saved in {}", self.out.display())));
        });
        let mut facts = vec![
            format!("{} images saved ({still} still, {moving} moving)", still + moving),
            format!("{} sequence{} validated", self.validated, if self.validated == 1 { "" } else { "s" }),
        ];
        if !device.is_empty() {
            facts.push(format!("Models on {device}, {fps:.0} frames a second, {source}"));
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
                let size = ui.available_width().min(ui.available_height() * 0.46);
                ui.vertical_centered(|ui| self.cube.show(ui, size));
                ui.label(theme::faint("Your cube should look like this one."));
                ui.add_space(6.0);
                theme::card(ui, Some(theme::LABEL), |ui| {
                    ui.set_width(ui.available_width());
                    match phase {
                        Phase::Paused(Pause::Start) => {
                            ui.label(theme::title("Show the solved cube to the camera"));
                            ui.label(theme::muted("Then press Space. Hold it any way you like: the colours tell which face is which."));
                        }
                        Phase::Paused(Pause::Undone) => {
                            ui.label(theme::title("Move taken back"));
                            ui.label(theme::muted("Put your cube back as the one above, then press Space."));
                        }
                        Phase::Paused(Pause::Lost) => {
                            ui.label(theme::title("Sequence set aside"));
                            ui.label(theme::muted("Solve your cube, then press Space: a new sequence starts."));
                        }
                        Phase::Paused(Pause::Done) => {
                            ui.label(theme::title("Sequence finished: is your cube solved?"));
                            ui.label(theme::muted("Space: yes, the next sequence starts. X: no, this one is set aside."));
                        }
                        Phase::Still => {
                            ui.label(theme::title("Show the cube, turn nothing"));
                            ui.label(theme::muted("Tilt it, bring it closer, further, show another corner."));
                            ui.add(
                                egui::ProgressBar::new(progress as f32).fill(theme::LABEL).desired_width(ui.available_width().min(320.0)),
                            );
                            if let Some(mv) = prompt {
                                ui.horizontal(|ui| {
                                    ui.label(theme::faint("Next"));
                                    theme::chip(ui, MOVES[mv], theme::Chip::Ahead);
                                });
                            }
                        }
                        Phase::Move => {
                            if let Some(mv) = prompt {
                                ui.horizontal(|ui| {
                                    let large = egui::FontId::new(60.0, theme::semibold());
                                    ui.label(egui::RichText::new(MOVES[mv]).font(large).color(theme::face_colour(MOVES[mv])));
                                    ui.vertical(|ui| {
                                        ui.add_space(12.0);
                                        ui.label(egui::RichText::new(in_words(mv)).size(18.0));
                                        ui.label(theme::muted("as seen facing that face. Then press Space."));
                                    });
                                });
                            }
                        }
                    }
                });
                if recording {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.label(theme::muted(format!("Sequence {}, move {} of {total}", sequence + 1, (step + 1).min(total))));
                        if !upcoming.is_empty() && phase == Phase::Move {
                            ui.add_space(8.0);
                            ui.label(theme::faint("then"));
                            for &m in &upcoming {
                                theme::chip(ui, MOVES[m], theme::Chip::Ahead);
                            }
                        }
                    });
                }
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(14.0, 4.0);
                        theme::key(ui, "Space", "done");
                        theme::key(ui, "Backspace", "take the last move back");
                        theme::key(ui, "X", "my cube no longer matches");
                        theme::key(ui, "Esc", "end");
                    });
                });
            });
        let centre = egui::Frame::new().fill(theme::PLASTIC).inner_margin(egui::Margin { left: 18, right: 12, top: 4, bottom: 8 });
        egui::CentralPanel::default().frame(centre).show(ctx, |ui| {
            if let Some(error) = &error {
                theme::failure(ui, "The session stopped", error);
                return;
            }
            let Some(texture) = &self.picture else {
                ui.centered_and_justified(|ui| {
                    ui.label(theme::title(if status.is_empty() { "Waiting for the camera…" } else { &status }))
                });
                return;
            };
            let rect = picture(ui, texture, 62.0);
            if let Some(boxed) = boxed {
                frame_box(ui, rect, boxed, if recording { theme::GOOD } else { theme::WARNING });
            }
            ui.add_space(4.0);
            let (what, colour) = match (boxed.is_some(), recording, agreement) {
                (false, _, _) => ("No cube in sight: nothing is saved".to_owned(), theme::WARNING),
                (true, false, _) => ("Paused: nothing is saved".to_owned(), theme::WARNING),
                (true, true, None) => ("Saving. The reader reads no face here".to_owned(), theme::MUTED),
                (true, true, Some(a)) if a >= 0.75 => {
                    (format!("Saving. The reader agrees with the cube on screen ({:.0} %)", a * 100.0), theme::GOOD)
                }
                (true, true, Some(a)) if a >= 0.5 => {
                    (format!("Saving. The reader half agrees with the cube on screen ({:.0} %)", a * 100.0), theme::WARNING)
                }
                (true, true, Some(a)) => (
                    format!("Saving. The reader disagrees with the cube on screen ({:.0} %): check your cube against it", a * 100.0),
                    theme::ERROR,
                ),
            };
            state_line(ui, colour, &what);
            ui.label(theme::faint("Only the square around the cube is saved, never the whole picture."));
        });
        // The still time's bar moves without a new frame
        if phase == Phase::Still {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        if home || quit { Next::Home } else { Next::Stay }
    }

    fn leave(&mut self) {
        if std::mem::replace(&mut self.ended, true) {
            return;
        }
        self.stop.store(true, Ordering::Relaxed);
        wait_for(self.worker.take());
        let t = self.start.elapsed().as_secs_f64();
        let events = self.session.lock().unwrap().take_events();
        let (still, moving) = {
            let s = self.shared.lock().unwrap();
            (s.still, s.moving)
        };
        let mut log = self.log.lock().unwrap();
        for event in events.iter().chain(std::iter::once(&json!({"type": "end", "t": ms(t), "still": still, "moving": moving}))) {
            if let Err(e) = log.write(event) {
                eprintln!("error: {e:#}");
            }
        }
        println!("{} images saved in {}", still + moving, self.out.display());
    }
}

/// The labelling screen on a new session's folder
pub fn screen(options: LabelOptions, ctx: &egui::Context) -> Result<LabelScreen> {
    let log = Log::create(&options.out)?;
    Ok(LabelScreen::new(options, log, ctx))
}
