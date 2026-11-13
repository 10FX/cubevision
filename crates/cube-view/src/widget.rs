//! The cube in an egui window: it plays the moves handed to it one after the other, and turns under the mouse

use std::collections::VecDeque;

use cube_core::Cube;
use egui::{Color32, Pos2, Response, Sense, Shape, Stroke, Ui, Vec2};

use crate::geometry::{IDENTITY, Mat3, mul, quads, view};

/// A face turn being played
struct Playing {
    token: String,
    face: usize,
    /// Quarter turns, clockwise: 1, 2 or -1
    quarters: f32,
    /// When it started, in egui's clock (seconds)
    started: f64,
}

pub struct CubeView {
    cube: Cube,
    waiting: VecDeque<String>,
    playing: Option<Playing>,
    /// How the cube is turned for the eye, degrees; the mouse changes them
    pub yaw: f32,
    pub pitch: f32,
    /// Seconds a quarter turn takes on screen; moves waiting make it shorter
    pub quarter_seconds: f32,
    /// How the whole cube is turned in the hands, before the eye's own view: a replay sets it as the solve goes
    pub held: Mat3,
}

/// A face turn's face and quarter turns, clockwise ("U", "R2", "F'"); None for anything else
pub fn face_turn(token: &str) -> Option<(usize, f32)> {
    let mut chars = token.chars();
    let face = "URFDLB".find(chars.next()?)?;
    match chars.as_str() {
        "" => Some((face, 1.0)),
        "2" | "2'" => Some((face, 2.0)),
        "'" => Some((face, -1.0)),
        _ => None,
    }
}

impl CubeView {
    pub fn new(cube: Cube) -> Self {
        CubeView { cube, waiting: VecDeque::new(), playing: None, yaw: -30.0, pitch: 25.0, quarter_seconds: 0.18, held: IDENTITY }
    }

    /// The cube as it is once every move handed over is played
    pub fn cube(&self) -> Cube {
        let mut cube = self.cube.clone();
        for token in self.playing.iter().map(|p| &p.token).chain(&self.waiting) {
            cube = cube.apply(token);
        }
        cube
    }

    /// This cube, at once: the moves waiting are dropped
    pub fn set(&mut self, cube: Cube) {
        self.cube = cube;
        self.waiting.clear();
        self.playing = None;
    }

    /// One more move to play: a face turn is shown turning, anything else (a rotation, a wide move, a slice)
    /// changes the cube at once when its turn comes
    pub fn play(&mut self, token: &str) {
        self.waiting.push_back(token.to_owned());
    }

    /// Whether a move is still being played or waiting
    pub fn busy(&self) -> bool {
        self.playing.is_some() || !self.waiting.is_empty()
    }

    /// The moves move on to time `now` (seconds); the layer turning and how far (radians), if any
    pub fn advance(&mut self, now: f64) -> Option<(usize, f32)> {
        loop {
            if let Some(playing) = &self.playing {
                // The more moves wait, the faster each goes: the cube catches up
                let seconds = self.quarter_seconds * playing.quarters.abs().powf(0.7) / (1.0 + 0.6 * self.waiting.len() as f32);
                let progress = ((now - playing.started) as f32 / seconds.max(1e-3)).clamp(0.0, 1.0);
                if progress < 1.0 {
                    let eased = progress * progress * (3.0 - 2.0 * progress);
                    return Some((playing.face, eased * playing.quarters * std::f32::consts::FRAC_PI_2));
                }
                self.cube = self.cube.apply(&playing.token);
                self.playing = None;
            }
            let token = self.waiting.pop_front()?;
            match face_turn(&token) {
                Some((face, quarters)) => self.playing = Some(Playing { token, face, quarters, started: now }),
                None => self.cube = self.cube.apply(&token),
            }
        }
    }

    /// The cube drawn in a square of `side` points; dragging turns it
    pub fn show(&mut self, ui: &mut Ui, side: f32) -> Response {
        let (rect, response) = ui.allocate_exact_size(Vec2::splat(side), Sense::drag());
        if response.dragged() {
            let moved = response.drag_delta();
            self.yaw = (self.yaw + moved.x * 0.5) % 360.0;
            self.pitch = (self.pitch + moved.y * 0.5).clamp(-89.0, 89.0);
        }
        let turning = self.advance(ui.input(|i| i.time));
        if self.busy() {
            ui.ctx().request_repaint();
        }
        let painter = ui.painter_at(rect);
        let (centre, half) = (rect.center(), side / 2.0);
        for quad in quads(&self.cube.colors(), turning, &mul(&view(self.yaw, self.pitch), &self.held)) {
            let points: Vec<Pos2> = quad.points.iter().map(|p| Pos2::new(centre.x + p[0] * half, centre.y + p[1] * half)).collect();
            let [r, g, b] = quad.colour;
            painter.add(Shape::convex_polygon(points, Color32::from_rgb(r, g, b), Stroke::NONE));
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_turns_are_told_from_the_rest() {
        assert_eq!(face_turn("U"), Some((0, 1.0)));
        assert_eq!(face_turn("B'"), Some((5, -1.0)));
        assert_eq!(face_turn("R2"), Some((1, 2.0)));
        for other in ["", "x", "y2", "Rw", "r", "M'", "U3", "2R"] {
            assert_eq!(face_turn(other), None, "{other}");
        }
    }

    #[test]
    fn moves_are_played_in_order_and_end_on_the_cube_they_make() {
        let mut view = CubeView::new(Cube::solved());
        for token in ["R", "y", "U2", "M", "F'"] {
            view.play(token);
        }
        let want = Cube::solved().apply("R y U2 M F'");
        assert_eq!(view.cube().colors(), want.colors());
        // Mid-way through the first: the R layer is turning, nothing is applied yet
        assert_eq!(view.advance(0.0), Some((1, 0.0)));
        let (face, angle) = view.advance(0.01).unwrap();
        assert!(face == 1 && angle > 0.0 && angle < std::f32::consts::FRAC_PI_2);
        assert!(view.busy());
        // Long after: all played, the cube at rest
        let mut now = 0.0;
        while view.busy() {
            now += 0.05;
            view.advance(now);
            assert!(now < 10.0, "the moves never end");
        }
        assert_eq!(view.advance(now + 1.0), None);
        assert_eq!(view.cube().colors(), want.colors());
        // A cube set at once drops what was waiting
        view.play("L");
        view.set(Cube::solved());
        assert!(!view.busy() && view.cube().colors() == Cube::solved().colors());
    }
}
