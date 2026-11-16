//! The window: a first screen to choose what to do (the camera live, a video, labelling) and where the models
//! run, then that mode's screen. What a mode needs and lacks (ffmpeg, the models) is said there, with the way
//! to get it.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use cube_capture::CameraRequest;
use cube_core::Cube;
use cube_decode::state::MOVES;
use cube_infer::device::Device;
use cube_view::CubeView;
use eframe::egui;

use crate::label_window::{self, LabelOptions};
use crate::live::{LiveOptions, LiveScreen, Source};
use crate::replay_window::{ReplayScreen, Shown};
use crate::settings::{self, Engine, Settings};
use crate::solve::{Progress, Request, Solve, solve};
use crate::window::{Next, Screen, Snapshot};
use crate::{fetch, needs, theme};

/// The screen the window opens on, when the command line asked for a mode
pub enum First {
    Live(LiveOptions),
    Label(LabelOptions),
    Replay(Box<Shown>),
    /// A video read in the window, then its replay
    Read(Request, Option<Device>),
}

pub struct Start {
    /// The models' folder
    pub models: PathBuf,
    pub first: Option<First>,
    /// Of the window whatever it shows, and of the screen a video read ends on
    pub snapshot: Snapshot,
    pub screen: Snapshot,
}

/// Something run aside, and what it gave
enum Job<T> {
    Idle,
    Running(Receiver<T>),
}

impl<T: Send + 'static> Job<T> {
    fn start(ctx: &egui::Context, work: impl FnOnce() -> T + Send + 'static) -> Self {
        let (done, wait) = std::sync::mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = done.send(work());
            ctx.request_repaint();
        });
        Job::Running(wait)
    }

    fn running(&self) -> bool {
        matches!(self, Job::Running(_))
    }

    /// What it gave, once
    fn take(&mut self) -> Option<T> {
        let Job::Running(wait) = self else { return None };
        match wait.try_recv() {
            Ok(out) => {
                *self = Job::Idle;
                Some(out)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                *self = Job::Idle;
                None
            }
        }
    }
}

struct Home {
    /// The program's folder, where the released zip keeps its runtimes
    beside: PathBuf,
    models: PathBuf,
    settings: Settings,
    file: Option<PathBuf>,
    engines: Vec<Engine>,
    engine: Engine,
    cameras: Vec<(u32, String)>,
    mirror: Arc<AtomicBool>,
    /// The cube that turns on the first screen, and when it next does
    cube: CubeView,
    turn_at: f64,
    seed: u64,
    video: Option<PathBuf>,
    /// The path typed, where no file dialog is built
    typed: String,
    ffmpeg: bool,
    can_install: Option<bool>,
    installing: Job<Result<(), String>>,
    install_failed: Option<String>,
    have_models: bool,
    /// Bytes received and to come
    downloaded: Arc<Mutex<(u64, Option<u64>)>>,
    downloading: Job<Result<(), String>>,
    download_failed: Option<String>,
    /// What could not be started, said on top
    problem: Option<String>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Home {
    fn new(models: PathBuf, intel_graphics: bool) -> Self {
        let beside = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)).unwrap_or_default();
        let file = Settings::file();
        let settings = file.as_deref().map_or_else(Settings::default, Settings::load);
        let engines = settings::engines(&beside, cfg!(windows));
        let engine =
            settings.engine.filter(|engine| engines.contains(engine)).unwrap_or_else(|| settings::first_engine(&engines, intel_graphics));
        let mut cube = CubeView::new(Cube::solved());
        cube.quarter_seconds = 0.45;
        Home {
            have_models: fetch::present(&models),
            beside,
            models,
            mirror: Arc::new(AtomicBool::new(settings.mirror)),
            typed: String::new(),
            settings,
            file,
            engines,
            engine,
            cameras: cube_capture::cameras(),
            cube,
            turn_at: 1.0,
            seed: now() | 1,
            video: None,
            ffmpeg: needs::find_ffmpeg(),
            can_install: None,
            installing: Job::Idle,
            install_failed: None,
            downloaded: Arc::default(),
            downloading: Job::Idle,
            download_failed: None,
            problem: None,
        }
    }

    fn save(&mut self) {
        self.settings.engine = Some(self.engine);
        self.settings.mirror = self.mirror.load(Ordering::Relaxed);
        if let Some(file) = &self.file
            && let Err(e) = self.settings.save(file)
        {
            eprintln!("note: the settings were not saved: {e:#}");
        }
    }

    /// Back from a mode: what may have changed meanwhile is looked at again
    fn back(&mut self) {
        self.cameras = cube_capture::cameras();
        self.have_models = fetch::present(&self.models);
        self.save();
    }

    /// The models' runtime set for the engine chosen. False: this process already loaded another one, which it
    /// keeps until it ends; the program is started again and this window closes.
    fn use_engine(&mut self, ctx: &egui::Context) -> bool {
        let wanted = settings::point_at(self.engine, &self.beside);
        if let (Some(loaded), Some(wanted)) = (cube_infer::models::loaded_library(), wanted)
            && loaded != wanted
        {
            self.save();
            match std::env::current_exe().and_then(|exe| std::process::Command::new(exe).spawn()) {
                Ok(_) => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
                Err(e) => self.problem = Some(format!("cubevision could not start again to change where the models run: {e}")),
            }
            return false;
        }
        true
    }

    fn camera(&self) -> CameraRequest {
        CameraRequest { index: self.settings.camera, ..CameraRequest::default() }
    }

    fn next_random(&mut self, below: u64) -> u64 {
        // SplitMix64
        self.seed = self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) % below
    }

    fn show(&mut self, ctx: &egui::Context) -> Option<Box<dyn Screen>> {
        if let Some(result) = self.installing.take() {
            self.ffmpeg = needs::find_ffmpeg();
            self.install_failed = match result {
                Err(e) => Some(e),
                Ok(()) if !self.ffmpeg => Some("ffmpeg was installed but cannot be run yet: close cubevision and open it again".into()),
                Ok(()) => None,
            };
        }
        if let Some(result) = self.downloading.take() {
            self.have_models = fetch::present(&self.models);
            self.download_failed = result.err();
        }
        // A video dropped on the window
        let (dropped, hovering) =
            ctx.input(|i| (i.raw.dropped_files.first().and_then(|file| file.path.clone()), !i.raw.hovered_files.is_empty()));
        if let Some(path) = dropped {
            self.typed = path.display().to_string();
            self.video = Some(path);
        }
        // The cube of the first screen turns on its own, slowly
        let time = ctx.input(|i| i.time);
        if time >= self.turn_at && !self.cube.busy() {
            let m = self.next_random(18) as usize;
            self.cube.play(MOVES[m]);
            self.turn_at = time + 1.6;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(200));

        let mut next: Option<Box<dyn Screen>> = None;
        let frame = egui::Frame::new().fill(theme::PLASTIC).inner_margin(egui::Margin::symmetric(30, 22));
        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            let left = (ui.available_width() * 0.38).clamp(300.0, 500.0);
            let height = ui.available_height();
            ui.horizontal_top(|ui| {
                ui.allocate_ui_with_layout(egui::vec2(left, height), egui::Layout::top_down(egui::Align::LEFT), |ui| {
                    let size = left.min(height * 0.58);
                    self.cube.show(ui, size);
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("cubevision").font(egui::FontId::new(44.0, theme::semibold())).color(theme::TEXT));
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new("Reads a Rubik's Cube from a camera or a video: every move, as it is made or afterwards.")
                                .size(16.0)
                                .color(theme::MUTED),
                        )
                        .wrap(),
                    );
                    ui.add_space(6.0);
                    ui.label(theme::faint(format!("Version {}", env!("CARGO_PKG_VERSION"))));
                });
                ui.add_space(26.0);
                ui.vertical(|ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 12.0;
                        next = self.modes(ui, ctx, hovering);
                    });
                });
            });
        });
        next
    }

    /// The right of the first screen: what stands in the way, the three modes, the settings
    fn modes(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, hovering: bool) -> Option<Box<dyn Screen>> {
        let mut next: Option<Box<dyn Screen>> = None;
        if let Some(problem) = self.problem.clone() {
            theme::banner(ui, theme::ERROR, "That could not be started", |ui| {
                ui.add(egui::Label::new(problem).wrap());
            });
        }
        if !self.have_models {
            self.models_missing(ui, ctx);
        }
        let ready = self.have_models;
        let camera = !self.cameras.is_empty();

        theme::card(ui, Some(theme::LIVE), |ui| {
            ui.set_width(ui.available_width());
            ui.label(theme::title("Live"));
            ui.add(
                egui::Label::new(theme::muted(
                    "Turn your cube in front of the camera: each move is written as you make it, on a cube that follows yours.",
                ))
                .wrap(),
            );
            if !camera {
                ui.label(egui::RichText::new("No camera was found on this PC.").color(theme::WARNING));
            }
            if theme::primary(ui, ready && camera, "Start live", theme::LIVE).clicked() && self.use_engine(ctx) {
                self.problem = None;
                self.save();
                next = Some(Box::new(LiveScreen::new(
                    LiveOptions {
                        source: Source::Camera(self.camera()),
                        models: self.models.clone(),
                        device: self.engine.device(),
                        mirror: self.mirror.clone(),
                        start: None,
                        snapshot: Snapshot::default(),
                    },
                    ctx,
                )));
            }
        });

        let lit = if hovering { theme::WHITE } else { theme::VIDEO };
        theme::card(ui, Some(lit), |ui| {
            ui.set_width(ui.available_width());
            ui.label(theme::title("Video"));
            ui.add(egui::Label::new(theme::muted("A filmed solve and its scramble give back every move, timed, stage by stage.")).wrap());
            self.choose_video(ui);
            ui.add(theme::field(&mut self.settings.scramble, "The scramble, as on the timer: R U2 F' D …"));
            let scramble = self.settings.scramble.trim().to_owned();
            let valid = match crate::colours::scramble(&scramble) {
                Ok(()) => !scramble.is_empty(),
                Err(e) => {
                    if !scramble.is_empty() {
                        ui.label(egui::RichText::new(format!("{e:#}")).color(theme::ERROR));
                    }
                    false
                }
            };
            if !self.ffmpeg {
                self.ffmpeg_missing(ui, ctx);
            }
            let video = self.video.clone().filter(|video| video.is_file());
            if self.video.is_some() && video.is_none() {
                ui.label(egui::RichText::new("That file was not found.").color(theme::ERROR));
            }
            let can = ready && self.ffmpeg && valid && video.is_some();
            if theme::primary(ui, can, "Find the moves", theme::VIDEO).clicked()
                && let Some(video) = video
                && self.use_engine(ctx)
            {
                self.problem = None;
                self.settings.folder = video.parent().map(Path::to_path_buf);
                self.save();
                next = Some(Box::new(ReadingScreen::new(
                    Request::new(video, scramble),
                    self.models.clone(),
                    self.engine.device(),
                    Snapshot::default(),
                    ctx,
                )));
            }
        });

        theme::card(ui, Some(theme::LABEL), |ui| {
            ui.set_width(ui.available_width());
            ui.label(theme::title("Label"));
            ui.add(egui::Label::new(theme::muted("Make labelled images of your own cube, to train the reader: you follow the moves asked for, the square around the cube is saved with what it shows.")).wrap());
            ui.add(theme::field(&mut self.settings.note, "Where is it filmed? The room, the light, the cube"));
            if theme::primary(ui, ready && camera, "Start labelling", theme::LABEL).clicked() && self.use_engine(ctx) {
                self.problem = None;
                self.save();
                let stamp = now();
                let note = self.settings.note.trim();
                let options = LabelOptions {
                    source: Source::Camera(self.camera()),
                    models: self.models.clone(),
                    device: self.engine.device(),
                    out: PathBuf::from("cubevision-labels").join(format!("session-{stamp}")),
                    seed: stamp,
                    length: 20,
                    note: if note.is_empty() { "not told".to_owned() } else { note.to_owned() },
                    auto: None,
                    snapshot: Snapshot::default(),
                };
                match label_window::screen(options, ctx) {
                    Ok(screen) => next = Some(Box::new(screen)),
                    Err(e) => self.problem = Some(format!("{e:#}")),
                }
            }
        });

        theme::card(ui, None, |ui| {
            ui.set_width(ui.available_width());
            egui::Grid::new("settings").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
                ui.label(theme::muted("Camera"));
                let shown = self.cameras.iter().find(|(index, _)| *index == self.settings.camera).or(self.cameras.first());
                let name = shown.map_or("None found".to_owned(), |(_, name)| name.clone());
                if let Some((index, _)) = shown {
                    self.settings.camera = *index;
                }
                egui::ComboBox::from_id_salt("camera").selected_text(name).width(280.0).show_ui(ui, |ui| {
                    for (index, name) in &self.cameras {
                        ui.selectable_value(&mut self.settings.camera, *index, name);
                    }
                });
                ui.end_row();
                ui.label(theme::muted("Models run on"));
                let before = self.engine;
                egui::ComboBox::from_id_salt("engine").selected_text(self.engine.label()).width(280.0).show_ui(ui, |ui| {
                    for &engine in &self.engines {
                        ui.selectable_value(&mut self.engine, engine, engine.label());
                    }
                });
                if self.engine != before {
                    self.save();
                    self.use_engine(ctx);
                }
                ui.end_row();
            });
            ui.label(theme::faint("Intel graphics are fastest with OpenVINO, other graphics cards with DirectML. Changing it after a mode has run restarts cubevision."));
        });
        next
    }

    fn choose_video(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            #[cfg(windows)]
            if ui.button("Choose a video").clicked() {
                let mut dialog = rfd::FileDialog::new().add_filter("Videos", &["mp4", "webm", "mov", "mkv", "avi", "m4v"]);
                if let Some(folder) = self.settings.folder.as_ref().filter(|folder| folder.is_dir()) {
                    dialog = dialog.set_directory(folder);
                }
                if let Some(path) = dialog.pick_file() {
                    self.typed = path.display().to_string();
                    self.video = Some(path);
                }
            }
            #[cfg(not(windows))]
            {
                let edit = theme::field(&mut self.typed, "The video's path").desired_width(ui.available_width() * 0.6);
                if ui.add(edit).changed() {
                    let typed = self.typed.trim();
                    self.video = (!typed.is_empty()).then(|| PathBuf::from(typed));
                }
            }
            match &self.video {
                Some(video) if cfg!(windows) => {
                    let name = video.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    ui.label(egui::RichText::new(name).color(theme::TEXT)).on_hover_text(video.display().to_string());
                }
                _ => {
                    ui.label(theme::faint("or drop one on this window"));
                }
            }
        });
    }

    fn ffmpeg_missing(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let can = *self.can_install.get_or_insert_with(needs::can_install_ffmpeg);
        theme::banner(ui, theme::WARNING, "Reading a video needs ffmpeg, which this PC does not have", |ui| {
            if self.installing.running() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Installing ffmpeg: a minute or two, about 100 MB to download.");
                });
                return;
            }
            if let Some(failed) = &self.install_failed {
                ui.add(egui::Label::new(egui::RichText::new(failed).color(theme::ERROR)).wrap());
            }
            ui.horizontal_wrapped(|ui| {
                if can && ui.button("Install ffmpeg").on_hover_text("With winget, Windows' own installer").clicked() {
                    self.install_failed = None;
                    self.installing = Job::start(ctx, || needs::install_ffmpeg().map_err(|e| format!("{e:#}")));
                }
                if ui.button("Check again").clicked() {
                    self.ffmpeg = needs::find_ffmpeg();
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(theme::faint(if can {
                    "Or type this in a terminal, then check again:"
                } else {
                    "Type this in a terminal, then check again:"
                }));
                ui.label(egui::RichText::new(needs::ffmpeg_command()).monospace().color(theme::TEXT));
            });
        });
    }

    fn models_missing(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let folder = self.models.display().to_string();
        theme::banner(ui, theme::WARNING, "The models that read the cube are not on this PC yet", |ui| {
            if self.downloading.running() {
                let (got, all) = *self.downloaded.lock().unwrap();
                let megabytes = got as f32 / (1 << 20) as f32;
                match all {
                    Some(all) if all > 0 => {
                        let text = format!("{megabytes:.0} of {:.0} MB", all as f32 / (1 << 20) as f32);
                        ui.add(egui::ProgressBar::new(got as f32 / all as f32).fill(theme::WARNING).text(text).desired_width(360.0));
                    }
                    _ => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(format!("Downloading: {megabytes:.0} MB"));
                        });
                    }
                }
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
                return;
            }
            if let Some(failed) = &self.download_failed {
                ui.add(egui::Label::new(egui::RichText::new(failed).color(theme::ERROR)).wrap());
            }
            ui.horizontal_wrapped(|ui| {
                if ui.button("Download the models").on_hover_text("About 70 MB, from cubevision's releases on GitHub").clicked() {
                    self.download_failed = None;
                    *self.downloaded.lock().unwrap() = (0, None);
                    let (dir, told) = (self.models.clone(), self.downloaded.clone());
                    self.downloading = Job::start(ctx, move || {
                        fetch::download(&dir, fetch::URL, &mut |got, all| *told.lock().unwrap() = (got, all))
                            .map(|_| ())
                            .map_err(|e| format!("{e:#}"))
                    });
                }
                ui.label(theme::faint(format!("They go in {folder}")));
            });
        });
    }
}

/// A video being read, then its replay
struct ReadingScreen {
    title: String,
    request: Arc<Request>,
    progress: Arc<Mutex<Progress>>,
    cancel: Arc<AtomicBool>,
    job: Job<Result<Option<Solve>, String>>,
    failed: Option<(String, String)>,
    replay: Option<ReplayScreen>,
    started: Option<f64>,
    /// Of the replay, or of this screen when told to be taken before the replay comes
    snapshot: Snapshot,
}

impl ReadingScreen {
    fn new(request: Request, models: PathBuf, device: Option<Device>, snapshot: Snapshot, ctx: &egui::Context) -> Self {
        let request = Arc::new(request);
        let progress = Arc::new(Mutex::new(Progress::Loading));
        let cancel = Arc::new(AtomicBool::new(false));
        let job = {
            let (request, progress, cancel, repaint) = (request.clone(), progress.clone(), cancel.clone(), ctx.clone());
            Job::start(ctx, move || {
                let mut tell = |now: Progress| {
                    *progress.lock().unwrap() = now;
                    repaint.request_repaint();
                    !cancel.load(Ordering::Relaxed)
                };
                solve(&request, &models, device, &mut tell).map_err(|e| format!("{e:#}"))
            })
        };
        let title = request.video.file_name().unwrap_or_default().to_string_lossy().into_owned();
        ReadingScreen { title, request, progress, cancel, job, failed: None, replay: None, started: None, snapshot }
    }
}

/// What a solve found in a video shows in its replay
pub fn shown(request: &Request, solve: Solve, snapshot: Snapshot) -> Shown {
    Shown {
        video: request.video.clone(),
        ffmpeg: request.ffmpeg.clone(),
        title: request.video.file_name().unwrap_or_default().to_string_lossy().into_owned(),
        scramble: request.scramble.clone(),
        device: solve.device,
        mirrored: solve.mirrored,
        warning: solve.warning.map(str::to_owned),
        result: solve.result,
        snapshot,
    }
}

impl Screen for ReadingScreen {
    fn show(&mut self, ctx: &egui::Context) -> Next {
        if let Some(replay) = &mut self.replay {
            return replay.show(ctx);
        }
        match self.job.take() {
            Some(Ok(Some(solve))) if solve.mismatch => {
                self.failed = Some((
                    "This scramble does not seem to be this video's".to_owned(),
                    "The moves found explain too little of what was read in the video. Check the scramble: it must be the one the cube was scrambled with, held white on top and green in front.".to_owned(),
                ));
            }
            Some(Ok(Some(solve))) => {
                let replay = self.replay.insert(ReplayScreen::new(shown(&self.request, solve, std::mem::take(&mut self.snapshot)), ctx));
                return replay.show(ctx);
            }
            Some(Ok(None)) => return Next::Home,
            Some(Err(e)) => self.failed = Some(("The video could not be read".to_owned(), e)),
            None => {}
        }
        let time = ctx.input(|i| i.time);
        let since = time - *self.started.get_or_insert(time);
        let escape = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        let title = self.title.clone();
        let home = theme::top_bar(ctx, "Video", theme::VIDEO, |ui| {
            ui.label(theme::faint(title));
        });
        let frame = egui::Frame::new().fill(theme::PLASTIC).inner_margin(egui::Margin::symmetric(30, 20));
        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            if let Some((headline, error)) = &self.failed {
                theme::failure(ui, headline, error);
                return;
            }
            ui.add_space(ui.available_height() * 0.3);
            ui.vertical_centered(|ui| {
                ui.set_max_width(520.0);
                let progress = self.progress.lock().unwrap().clone();
                match progress {
                    Progress::Loading => {
                        ui.label(theme::title("Loading the models"));
                        ui.add_space(6.0);
                        ui.spinner();
                    }
                    Progress::Reading { read, all } => {
                        ui.label(theme::title("Reading the video"));
                        ui.add_space(6.0);
                        match all.filter(|&all| all > 0) {
                            Some(all) => {
                                let bar = egui::ProgressBar::new((read as f32 / all as f32).min(1.0))
                                    .fill(theme::VIDEO)
                                    .text(format!("{read} of {all} frames"));
                                ui.add(bar.desired_width(520.0));
                            }
                            None => {
                                ui.label(theme::muted(format!("{read} frames")));
                            }
                        }
                    }
                    Progress::Decoding { read, seconds } => {
                        ui.label(theme::title("Finding the moves"));
                        ui.add_space(6.0);
                        ui.spinner();
                        ui.label(theme::muted(format!(
                            "{read} frames read in {seconds:.0} s. This step takes from a few seconds to a minute."
                        )));
                    }
                }
                ui.add_space(10.0);
                ui.label(theme::faint(format!("{since:.0} s")));
            });
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        });
        if home || escape { Next::Home } else { Next::Stay }
    }

    fn leave(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(replay) = &mut self.replay {
            replay.leave();
        }
    }
}

struct Shell {
    home: Home,
    screen: Option<Box<dyn Screen>>,
    snapshot: Snapshot,
}

impl eframe::App for Shell {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        match &mut self.screen {
            Some(screen) => {
                if screen.show(ctx) == Next::Home {
                    screen.leave();
                    self.screen = None;
                    self.home.back();
                }
            }
            None => self.screen = self.home.show(ctx),
        }
        self.snapshot.tick(ctx);
    }

    fn on_exit(&mut self) {
        if let Some(screen) = &mut self.screen {
            screen.leave();
        }
        self.home.save();
    }
}

/// Opens the window; returns when it is closed
pub fn run(start: Start) -> Result<()> {
    crate::window::run("cubevision", [1240.0, 780.0], move |cc| {
        // Intel's graphics, as the window's own renderer names them: OpenVINO is their fastest engine
        let intel = cc.wgpu_render_state.as_ref().is_some_and(|state| state.adapter.get_info().vendor == 0x8086);
        let mut home = Home::new(start.models, intel);
        let ctx = &cc.egui_ctx;
        let screen: Option<Box<dyn Screen>> = match start.first {
            None => None,
            Some(First::Live(options)) => Some(Box::new(LiveScreen::new(options, ctx))),
            Some(First::Replay(shown)) => Some(Box::new(ReplayScreen::new(*shown, ctx))),
            Some(First::Read(request, device)) => {
                Some(Box::new(ReadingScreen::new(request, home.models.clone(), device, start.screen.clone(), ctx)))
            }
            Some(First::Label(options)) => match label_window::screen(options, ctx) {
                Ok(screen) => Some(Box::new(screen)),
                Err(e) => {
                    home.problem = Some(format!("{e:#}"));
                    None
                }
            },
        };
        Box::new(Shell { home, screen, snapshot: start.snapshot })
    })
}
