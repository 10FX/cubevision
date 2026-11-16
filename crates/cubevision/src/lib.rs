//! The program's parts: a video or a camera read frame after frame as ChronoCube's
//! workers do (`worker.ts`, `liveWorker.ts`), and the result written out.

pub mod app;
pub mod check;
pub mod colours;
pub mod fetch;
pub mod json;
pub mod label;
pub mod label_window;
pub mod live;
pub mod needs;
pub mod read;
pub mod replay;
pub mod replay_window;
pub mod settings;
pub mod solve;
pub mod theme;
pub mod tools;
pub mod window;
