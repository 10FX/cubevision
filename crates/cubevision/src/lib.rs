//! The program's parts that are not its window: a video or a camera read frame after frame as ChronoCube's
//! workers do (`worker.ts`, `liveWorker.ts`), and the result written out.

pub mod check;
pub mod colours;
pub mod fetch;
pub mod json;
pub mod live;
pub mod read;
pub mod replay;
pub mod replay_window;
pub mod tools;
pub mod window;
