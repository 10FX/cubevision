//! The window: what the live mode and the replay share

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use eframe::egui;

/// A picture of the window saved once it has run for some seconds, then the window closed: to check what it
/// shows without anyone in front of it
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub file: Option<PathBuf>,
    pub after: f64,
    asked: bool,
}

impl Snapshot {
    pub fn new(file: Option<PathBuf>, after: f64) -> Self {
        Snapshot { file, after, asked: false }
    }

    /// To call at each frame
    pub fn tick(&mut self, ctx: &egui::Context) {
        let Some(file) = &self.file else { return };
        let taken = ctx.input(|i| {
            i.events.iter().find_map(|event| match event {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = taken {
            if let Err(e) = save(file, &image) {
                eprintln!("error: {e:#}");
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if !self.asked && ctx.input(|i| i.time) >= self.after {
            self.asked = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        ctx.request_repaint();
    }
}

fn save(file: &std::path::Path, image: &egui::ColorImage) -> Result<()> {
    let out = std::io::BufWriter::new(std::fs::File::create(file)?);
    let mut encoder = png::Encoder::new(out, image.width() as u32, image.height() as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let data: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
    encoder.write_header()?.write_image_data(&data)?;
    Ok(())
}

/// Opens the window and runs the app until it is closed
pub fn run(title: &str, size: [f32; 2], app: impl FnOnce(&eframe::CreationContext) -> Box<dyn eframe::App> + 'static) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(size).with_min_inner_size([760.0, 480.0]).with_title(title),
        ..Default::default()
    };
    eframe::run_native(
        title,
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(app(cc))
        }),
    )
    .map_err(|e| anyhow!("the window could not be opened: {e}"))
}

/// A move written large, in a flow of them
pub fn token(ui: &mut egui::Ui, text: &str, colour: egui::Color32) {
    ui.label(egui::RichText::new(text).monospace().size(20.0).color(colour));
}
