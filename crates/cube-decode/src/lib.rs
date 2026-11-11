//! From readings of a cube (the colours a model saw, frame after frame) to the moves turned: ChronoCube's
//! `src/lib/vision` and `src/lib/smartcube`, the parts that need neither a model nor a camera.

pub mod decode;
pub mod help;
pub mod present;
pub mod reading;
pub mod reconstruct;
pub mod state;
pub mod truth;

pub use reading::{FaceReading, FrameReadings, Motion, Reading, State, TimedMove};
