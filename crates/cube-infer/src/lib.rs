//! From a frame to what a reader saw of the cube in it: ChronoCube's `src/lib/vision/detector.ts`, `model.ts` and
//! the reader's crop of `roi.ts`, on ONNX Runtime instead of the browser.

pub mod follow;
pub mod geometry;

pub use cube_decode::motion::Frame;
pub use geometry::{BoxF, FrameError, Letterbox};
