//! What a reader saw (`search.ts`, `pipeline.ts` types)

/// Colour of each of the 54 stickers (U R F D L B, nine each)
pub type State = [u8; 54];

/// One face seen: named by its centre's colour
#[derive(Clone, Debug, PartialEq)]
pub struct FaceReading {
    /// Colour of the centre = the face (U R F D L B)
    pub center: u8,
    /// Probability of each colour for the nine stickers, row by row (None: hidden)
    pub stickers: [Option<[f32; 6]>; 9],
    /// Probability that the stickers come turned by each of the 4 rotations (`ROT`), when the reader tells
    pub rotation: Option<[f32; 4]>,
}

impl FaceReading {
    /// A cell or a rotation with a value no probability can have (a NaN, an infinity, below zero) counts as not seen
    pub fn sanitized(self) -> FaceReading {
        let clean = |xs: &[f32]| xs.iter().all(|x| x.is_finite() && *x >= 0.0);
        FaceReading {
            center: self.center,
            stickers: self.stickers.map(|cell| cell.filter(|p| clean(p))),
            rotation: self.rotation.filter(|r| clean(r)),
        }
    }
}

/// Frames that read alike, merged
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    /// First and last frame (ms from the start of the video) showing this
    pub from: f64,
    pub to: f64,
    pub faces: Vec<FaceReading>,
    /// Whether anything moved since the reading before began (Some(false): no turn between them; None: not known)
    pub moved: Option<bool>,
}

/// How much the cube's crop changed since the frame before
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Motion {
    /// Motion is not measured at all (TS: undefined)
    NotMeasured,
    /// Not known for this frame: the cube was just found again (TS: null)
    Unknown,
    Change(f64),
}

/// One frame read by the model
#[derive(Clone, Debug, PartialEq)]
pub struct FrameReadings {
    /// ms from the start of the video
    pub t: f64,
    pub faces: Vec<FaceReading>,
    /// The cube's orientation in the camera's frame (row-major 3×3), from models that give it
    pub orientation: Option<[f64; 9]>,
    /// Probability of a layer mid-turn
    pub turning: Option<f64>,
    pub motion: Motion,
}

/// A turn and when it ended (ms)
#[derive(Clone, Debug, PartialEq)]
pub struct TimedMove {
    pub token: String,
    pub t: f64,
}
