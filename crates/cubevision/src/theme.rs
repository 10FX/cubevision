//! How the window looks: the cube's six colours on its black plastic, set in Rubik (the typeface drawn for the
//! cube). A move is a chip marked with the colour of the face it turns.

use std::sync::Arc;

use eframe::egui::{
    self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Margin, Response, RichText, Sense, Stroke, StrokeKind,
    TextStyle, Ui, Vec2, vec2,
};

/// The window, a panel on it, a control on a panel, a line between two things
pub const PLASTIC: Color32 = Color32::from_rgb(0x12, 0x13, 0x16);
pub const PANEL: Color32 = Color32::from_rgb(0x1B, 0x1D, 0x21);
pub const RAISED: Color32 = Color32::from_rgb(0x27, 0x2A, 0x30);
pub const LINE: Color32 = Color32::from_rgb(0x36, 0x3A, 0x42);
/// Text: what is read, what goes with it, what may be skipped
pub const TEXT: Color32 = Color32::from_rgb(0xEE, 0xEF, 0xF1);
pub const MUTED: Color32 = Color32::from_rgb(0xA6, 0xAC, 0xB6);
pub const FAINT: Color32 = Color32::from_rgb(0x70, 0x76, 0x81);
/// The stickers, as bright as a screen shows them next to text
pub const WHITE: Color32 = Color32::from_rgb(0xF4, 0xF4, 0xF1);
pub const RED: Color32 = Color32::from_rgb(0xE5, 0x42, 0x47);
pub const GREEN: Color32 = Color32::from_rgb(0x25, 0xB8, 0x6C);
pub const YELLOW: Color32 = Color32::from_rgb(0xFF, 0xD4, 0x26);
pub const ORANGE: Color32 = Color32::from_rgb(0xFF, 0x8C, 0x26);
pub const BLUE: Color32 = Color32::from_rgb(0x3D, 0x86, 0xEE);

/// What each colour says outside the cube: a mode each, and how things went
pub const LIVE: Color32 = GREEN;
pub const VIDEO: Color32 = BLUE;
pub const LABEL: Color32 = ORANGE;
pub const GOOD: Color32 = GREEN;
pub const WARNING: Color32 = YELLOW;
pub const ERROR: Color32 = RED;

pub fn medium() -> FontFamily {
    FontFamily::Name("rubik-medium".into())
}

pub fn semibold() -> FontFamily {
    FontFamily::Name("rubik-semibold".into())
}

/// The fonts and the look, set once for the window
pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let files: [(&str, &'static [u8]); 3] = [
        ("rubik", include_bytes!("../assets/fonts/Rubik-400.ttf")),
        ("rubik-medium", include_bytes!("../assets/fonts/Rubik-500.ttf")),
        ("rubik-semibold", include_bytes!("../assets/fonts/Rubik-600.ttf")),
    ];
    for (name, bytes) in files {
        fonts.font_data.insert(name.to_owned(), Arc::new(FontData::from_static(bytes)));
    }
    // Rubik holds Latin only: egui's own fonts stay behind it for the other signs
    let behind = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    fonts.families.entry(FontFamily::Proportional).or_default().insert(0, "rubik".to_owned());
    for name in ["rubik-medium", "rubik-semibold"] {
        let mut family = vec![name.to_owned()];
        family.extend(behind.iter().cloned());
        fonts.families.insert(FontFamily::Name(name.into()), family);
    }
    ctx.set_fonts(fonts);

    ctx.style_mut(|style| {
        style.text_styles = [
            (TextStyle::Heading, FontId::new(22.0, semibold())),
            (TextStyle::Body, FontId::new(15.0, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(15.0, medium())),
            (TextStyle::Small, FontId::new(12.5, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(14.0, FontFamily::Monospace)),
        ]
        .into();
        style.spacing.item_spacing = vec2(10.0, 8.0);
        style.spacing.button_padding = vec2(14.0, 7.0);
        style.spacing.interact_size.y = 32.0;
        style.spacing.combo_width = 220.0;
        let v = &mut style.visuals;
        *v = egui::Visuals::dark();
        v.panel_fill = PLASTIC;
        v.window_fill = PANEL;
        v.extreme_bg_color = Color32::from_rgb(0x10, 0x11, 0x13);
        v.faint_bg_color = PANEL;
        v.window_stroke = Stroke::new(1.0, LINE);
        v.window_corner_radius = CornerRadius::same(10);
        v.selection.bg_fill = BLUE.gamma_multiply(0.55);
        v.selection.stroke = Stroke::new(1.0, TEXT);
        v.hyperlink_color = BLUE;
        v.warn_fg_color = WARNING;
        v.error_fg_color = ERROR;
        let radius = CornerRadius::same(7);
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, LINE);
        v.widgets.noninteractive.corner_radius = radius;
        for (widget, fill, edge, text) in [
            (&mut v.widgets.inactive, RAISED, Color32::TRANSPARENT, TEXT),
            (&mut v.widgets.hovered, Color32::from_rgb(0x32, 0x36, 0x3E), FAINT, Color32::WHITE),
            (&mut v.widgets.active, Color32::from_rgb(0x3D, 0x42, 0x4B), MUTED, Color32::WHITE),
            (&mut v.widgets.open, Color32::from_rgb(0x32, 0x36, 0x3E), FAINT, Color32::WHITE),
        ] {
            widget.weak_bg_fill = fill;
            widget.bg_fill = fill;
            widget.bg_stroke = Stroke::new(1.0, edge);
            widget.fg_stroke = Stroke::new(1.0, text);
            widget.corner_radius = radius;
            widget.expansion = 0.0;
        }
    });
}

/// The colour of the face a move turns (the face's centre, the cube held white on top and green in front);
/// a rotation of the whole cube or a slice turns none
pub fn face_colour(token: &str) -> Color32 {
    match token.chars().next().map(|letter| letter.to_ascii_uppercase()) {
        Some('U') => WHITE,
        Some('R') => RED,
        Some('F') => GREEN,
        Some('D') => YELLOW,
        Some('L') => ORANGE,
        Some('B') => BLUE,
        _ => FAINT,
    }
}

/// Where a move stands in what is shown
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chip {
    /// Made
    Done,
    /// The one just made: filled with its face's colour
    Now,
    /// Not made yet
    Ahead,
    /// Seen, but it may still change
    Unsettled,
}

/// A move as a chip: its face's colour as a small sticker, then the move
pub fn chip(ui: &mut Ui, token: &str, kind: Chip) -> Response {
    let face = face_colour(token);
    let (fill, edge, text, pip) = match kind {
        Chip::Done => (RAISED, Color32::TRANSPARENT, TEXT, face),
        Chip::Now => (face, Color32::TRANSPARENT, PLASTIC, PLASTIC.gamma_multiply(0.55)),
        Chip::Ahead => (Color32::TRANSPARENT, LINE, MUTED, face.gamma_multiply(0.6)),
        Chip::Unsettled => (Color32::TRANSPARENT, LINE, FAINT, face.gamma_multiply(0.35)),
    };
    let galley = ui.painter().layout_no_wrap(token.to_owned(), FontId::new(16.0, medium()), text);
    let (rect, response) = ui.allocate_exact_size(vec2(galley.size().x + 28.0, 28.0), Sense::click());
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        let edge = if response.hovered() { MUTED } else { edge };
        painter.rect(rect, CornerRadius::same(7), fill, Stroke::new(1.0, edge), StrokeKind::Inside);
        let sticker = egui::Rect::from_center_size(egui::pos2(rect.left() + 11.0, rect.center().y), Vec2::splat(7.0));
        painter.rect_filled(sticker, CornerRadius::same(2), pip);
        painter.galley(egui::pos2(rect.left() + 20.0, rect.center().y - galley.size().y / 2.0), galley, text);
    }
    response
}

/// The button of what a screen is for, in its colour
pub fn primary(ui: &mut Ui, enabled: bool, text: &str, colour: Color32) -> Response {
    let label = RichText::new(text).font(FontId::new(15.0, semibold())).color(PLASTIC);
    let fill = if enabled { colour } else { colour.gamma_multiply(0.35) };
    ui.add_enabled(enabled, egui::Button::new(label).fill(fill).min_size(vec2(0.0, 36.0)))
}

/// A title in a panel
pub fn title(text: &str) -> RichText {
    RichText::new(text).font(FontId::new(19.0, semibold())).color(TEXT)
}

pub fn muted(text: impl Into<String>) -> RichText {
    RichText::new(text).color(MUTED)
}

pub fn faint(text: impl Into<String>) -> RichText {
    RichText::new(text).color(FAINT).size(12.5)
}

/// A number or a word that matters, larger
pub fn strong(text: impl Into<String>, size: f32) -> RichText {
    RichText::new(text).font(FontId::new(size, semibold())).color(TEXT)
}

/// A line of text to fill in, with what to write there shown while it is empty
pub fn field<'a>(text: &'a mut String, hint: &str) -> egui::TextEdit<'a> {
    egui::TextEdit::singleline(text)
        .hint_text(RichText::new(hint).color(FAINT))
        .margin(Margin::symmetric(10, 7))
        .desired_width(f32::INFINITY)
}

/// A panel: what belongs together, with the colour of what it is about down its left edge
pub fn card<R>(ui: &mut Ui, accent: Option<Color32>, add: impl FnOnce(&mut Ui) -> R) -> R {
    let frame = egui::Frame::new().fill(PANEL).corner_radius(CornerRadius::same(10)).inner_margin(Margin::symmetric(18, 14));
    let shown = frame.show(ui, add);
    if let Some(accent) = accent {
        let rect = shown.response.rect;
        let bar = egui::Rect::from_min_max(rect.left_top(), egui::pos2(rect.left() + 4.0, rect.bottom()));
        ui.painter().rect_filled(bar, CornerRadius { nw: 10, sw: 10, ne: 0, se: 0 }, accent);
    }
    shown.inner
}

/// Something to know now: what happened, then what to do about it
pub fn banner<R>(ui: &mut Ui, colour: Color32, headline: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    let fill = Color32::from_rgb(
        ((PANEL.r() as u16 * 5 + colour.r() as u16) / 6) as u8,
        ((PANEL.g() as u16 * 5 + colour.g() as u16) / 6) as u8,
        ((PANEL.b() as u16 * 5 + colour.b() as u16) / 6) as u8,
    );
    let frame = egui::Frame::new().fill(fill).corner_radius(CornerRadius::same(8)).inner_margin(Margin::symmetric(14, 10));
    let shown = frame.show(ui, |ui| {
        ui.label(RichText::new(headline).font(FontId::new(15.0, semibold())).color(colour));
        add(ui)
    });
    let rect = shown.response.rect;
    let bar = egui::Rect::from_min_max(rect.left_top(), egui::pos2(rect.left() + 3.0, rect.bottom()));
    ui.painter().rect_filled(bar, CornerRadius { nw: 8, sw: 8, ne: 0, se: 0 }, colour);
    shown.inner
}

/// A key of the keyboard and what it does
pub fn key(ui: &mut Ui, name: &str, does: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let galley = ui.painter().layout_no_wrap(name.to_owned(), FontId::new(12.5, medium()), MUTED);
        let (rect, _) = ui.allocate_exact_size(vec2(galley.size().x + 14.0, 22.0), Sense::hover());
        ui.painter().rect(rect, CornerRadius::same(5), RAISED, Stroke::new(1.0, LINE), StrokeKind::Inside);
        ui.painter().galley(egui::pos2(rect.left() + 7.0, rect.center().y - galley.size().y / 2.0), galley, MUTED);
        ui.label(faint(does));
    });
}

/// The bar on top of a mode's screen: the way home, the mode and its colour, then what the screen adds on the
/// right. True: home was asked for.
pub fn top_bar(ctx: &egui::Context, name: &str, colour: Color32, add: impl FnOnce(&mut Ui)) -> bool {
    let mut home = false;
    let frame = egui::Frame::new().fill(PLASTIC).inner_margin(Margin::symmetric(14, 9));
    egui::TopBottomPanel::top("top").frame(frame).show_separator_line(false).show(ctx, |ui| {
        ui.horizontal(|ui| {
            home = ui.button("Home").on_hover_text("Back to the first screen (Esc)").clicked();
            ui.add_space(6.0);
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::hover());
            ui.painter().rect_filled(rect, CornerRadius::same(3), colour);
            ui.label(RichText::new(name).font(FontId::new(17.0, semibold())).color(TEXT));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), add);
        });
    });
    home
}

/// The line of facts at the foot of a screen
pub fn foot(ctx: &egui::Context, facts: &[String]) {
    let frame = egui::Frame::new().fill(PLASTIC).inner_margin(Margin::symmetric(14, 6));
    egui::TopBottomPanel::bottom("foot").frame(frame).show_separator_line(false).show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 18.0;
            for fact in facts.iter().filter(|fact| !fact.is_empty()) {
                ui.label(faint(fact.as_str()));
            }
        });
    });
}

/// What went wrong, in the middle of the screen, and what the user can do
pub fn failure(ui: &mut Ui, headline: &str, error: &str) {
    ui.add_space(24.0);
    let width = ui.available_width().min(640.0);
    ui.horizontal(|ui| {
        ui.add_space((ui.available_width() - width) / 2.0);
        ui.vertical(|ui| {
            ui.set_width(width);
            banner(ui, ERROR, headline, |ui| {
                ui.add(egui::Label::new(RichText::new(error).color(TEXT)).wrap());
            });
        });
    });
}
