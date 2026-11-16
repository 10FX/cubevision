//! Frames to read a cube in: out of a video file (ffmpeg) or a camera. This crate stands in for the browser of
//! ChronoCube, which decodes videos and opens the camera there.

pub mod camera;
pub mod latest;
pub mod video;

pub use camera::{Camera, CameraFormat, CameraRequest, cameras};
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

/// A program run for its output, without a console window of its own: on Windows a program that has no console
/// (the window started by a double click) would open one for each
pub fn quiet(command: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW
        command.creation_flags(0x0800_0000);
    }
    command
}
