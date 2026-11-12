//! Frames to read a cube in: out of a video file (ffmpeg) or a camera. This crate stands in for the browser of
//! ChronoCube, which decodes videos and opens the camera there.

pub mod camera;
pub mod latest;
pub mod video;

pub use camera::{Camera, CameraFormat, CameraRequest};
pub use latest::Latest;
pub use video::{VideoInfo, VideoOptions, VideoReader};

/// A frame: RGBA, four bytes a pixel, row by row
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// When it was shown: ms from the start of the video, or since the camera opened
    pub t: f64,
}
