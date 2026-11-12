//! The camera: only what needs none is tested by default
//!   cargo test -p cube-capture --release --test camera -- --include-ignored   (with a camera plugged in)

use std::time::Duration;

use cube_capture::{Camera, CameraRequest, Latest};

#[test]
fn only_the_latest_frame_is_kept() {
    let latest = Latest::new();
    assert_eq!((latest.take(), latest.dropped()), (None, 0));
    for frame in [1, 2, 3] {
        latest.put(frame);
    }
    assert_eq!((latest.take(), latest.take(), latest.dropped()), (Some(3), None, 2));
    latest.put(4);
    assert_eq!((latest.take(), latest.dropped()), (Some(4), 2));
}

#[test]
fn a_frame_put_by_another_thread_ends_the_wait() {
    let latest = Latest::new();
    assert_eq!(latest.wait(Duration::from_millis(20)), None);
    let writer = latest.clone();
    let thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        writer.put(7);
    });
    assert_eq!(latest.wait(Duration::from_secs(5)), Some(7));
    thread.join().unwrap();
}

#[test]
fn a_failure_reaches_the_reader() {
    let latest: Latest<u8> = Latest::new();
    assert_eq!(latest.error(), None);
    latest.put(1);
    latest.fail("the camera was unplugged".into());
    // The frame already there is still handed over; the wait does not hang on a camera that stopped
    assert_eq!(latest.error().as_deref(), Some("the camera was unplugged"));
    assert_eq!(latest.take(), Some(1));
    assert_eq!(latest.wait(Duration::from_secs(5)), None);
}

#[test]
fn missing_camera_is_explained() {
    let error = format!("{:#}", Camera::open(&CameraRequest { index: 99, ..CameraRequest::default() }).err().expect("no camera 99"));
    assert!(error.contains("camera 99"), "{error}");
    assert_eq!(CameraRequest::default(), CameraRequest { index: 0, width: 1280, height: 720, fps: 30 });
}

#[test]
#[ignore = "needs a camera"]
fn the_camera_delivers_frames() {
    let camera = Camera::open(&CameraRequest::default()).unwrap();
    let format = camera.format().clone();
    println!("{format:?}");
    let (mut frames, mut last) = (0, -1.0);
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(2) {
        if let Some(image) = camera.wait(Duration::from_millis(500)).unwrap() {
            assert_eq!(image.data.len(), image.width * image.height * 4);
            assert_eq!((image.width, image.height), (format.width, format.height));
            assert!(image.t > last, "times go forward: {} after {last}", image.t);
            last = image.t;
            frames += 1;
        }
    }
    println!("{frames} frames in 2 s, {} dropped", camera.dropped());
    assert!(frames >= 10, "{frames} frames in 2 s");
}
