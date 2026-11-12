mod common;

use common::*;
use cube_infer::Frame;
use cube_infer::geometry::{BoxF, check, letterbox, reader_crop, square_crop, tensor, unletterbox};

#[test]
fn geometry_matches_typescript() {
    for f in fixture("geometry.json")["frames"].as_array().unwrap() {
        let (width, height) = (f["width"].as_u64().unwrap() as usize, f["height"].as_u64().unwrap() as usize);
        let data = bytes(f["data"].as_str().unwrap());
        let frame = Frame { data: &data, width, height };
        for (k, case) in f["letterboxes"].as_array().unwrap().iter().enumerate() {
            let what = format!("{width}×{height} letterbox {k}");
            let region = boxf(&case["region"]);
            let (square, lb) = letterbox(&frame, 32, 0.5, region.as_ref());
            assert!(close(lb.scale, num(&case["scale"])), "{what}: scale {} want {}", lb.scale, case["scale"]);
            assert_eq!((lb.ox, lb.oy), (case["ox"].as_i64().unwrap(), case["oy"].as_i64().unwrap()), "{what}: offsets");
            assert_eq!((lb.rx, lb.ry), (num(&case["rx"]), num(&case["ry"])), "{what}: corner");
            assert_eq!(square.len(), 3 * 32 * 32, "{what}: size");
            assert_eq!(&square[..8], f32s(&case["start"]).as_slice(), "{what}: start");
            assert_eq!(&square[1536..1544], f32s(&case["middle"]).as_slice(), "{what}: middle");
            assert_eq!(float_hash(&square), case["hash"].as_u64().unwrap(), "{what}: tensor");
            for b in case["boxes"].as_array().unwrap() {
                let boxed: [f32; 4] = f32s(&b["box"]).try_into().unwrap();
                let back = unletterbox(boxed, 32, &lb);
                assert!(same_box(&back, &boxf(&b["back"]).unwrap()), "{what}: {back:?} want {}", b["back"]);
            }
        }
        for (k, case) in f["crops"].as_array().unwrap().iter().enumerate() {
            let what = format!("{width}×{height} crop {k}");
            let boxed = boxf(&case["box"]).unwrap();
            for (margin, want) in [1.5, 1.0].into_iter().zip(case["reader"].as_array().unwrap()) {
                assert!(same_box(&reader_crop(&boxed, margin), &boxf(want).unwrap()), "{what}: reader crop ×{margin}");
            }
            assert_eq!(hex(&square_crop(&frame, &boxed, 8)), case["square8"].as_str().unwrap(), "{what}: 8×8");
            let square = square_crop(&frame, &boxed, 24);
            assert_eq!(square.len(), 24 * 24 * 4, "{what}: size");
            assert_eq!(byte_hash(&square), case["square24"].as_u64().unwrap(), "{what}: 24×24");
        }
    }
}

#[test]
fn a_short_or_empty_frame_is_refused() {
    let data = [0u8; 4 * 6 - 1];
    assert!(check(&Frame { data: &data, width: 3, height: 2 }).unwrap_err().to_string().contains("3×2"));
    assert!(check(&Frame { data: &data, width: 0, height: 2 }).is_err());
    assert!(check(&Frame { data: &data, width: 3, height: 0 }).is_err());
    assert!(check(&Frame { data: &data, width: 2, height: 2 }).is_ok());
    // Sizes whose product overflows are refused too, not wrapped
    assert!(check(&Frame { data: &data, width: usize::MAX, height: 8 }).is_err());
}

#[test]
#[should_panic(expected = "shorter than")]
fn reading_a_short_frame_says_why() {
    let data = [0u8; 7];
    square_crop(&Frame { data: &data, width: 2, height: 2 }, &BoxF { x: 0.0, y: 0.0, w: 2.0, h: 2.0 }, 4);
}

#[test]
fn boxes_no_frame_can_hold_do_not_panic() {
    let data: Vec<u8> = (0..40 * 30 * 4).map(|i| (i % 251) as u8).collect();
    let frame = Frame { data: &data, width: 40, height: 30 };
    let odd = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -5.0, 1e300, 1e-300];
    for &x in &odd {
        for &w in &odd {
            let boxed = BoxF { x, y: 3.0, w, h: 7.0 };
            let (square, _) = letterbox(&frame, 32, 0.5, Some(&boxed));
            assert_eq!(square.len(), 3 * 32 * 32);
            assert!(square.iter().all(|v| (0.0..=1.0).contains(v)), "letterbox of {boxed:?}");
            let crop = square_crop(&frame, &boxed, 16);
            assert_eq!(crop.len(), 16 * 16 * 4, "crop of {boxed:?}");
            let both = BoxF { x, y: x, w, h: w };
            letterbox(&frame, 32, 0.5, Some(&both));
            square_crop(&frame, &both, 16);
        }
    }
    // A region off the frame keeps the pad; a box off the frame repeats its edge
    let (square, _) = letterbox(&frame, 32, 0.5, Some(&BoxF { x: 500.0, y: 500.0, w: 10.0, h: 10.0 }));
    assert!(square.iter().all(|&v| v == 0.5));
    let crop = square_crop(&frame, &BoxF { x: 500.0, y: 500.0, w: 10.0, h: 10.0 }, 4);
    assert!(crop.chunks(4).all(|px| px[..3] == data[data.len() - 4..data.len() - 1] && px[3] == 255));
}

#[test]
fn a_crop_becomes_planes_of_shares() {
    let rgba = [255, 0, 51, 255, 0, 255, 102, 255];
    let mut out = [9.0f32; 6];
    tensor(&rgba, &mut out);
    assert_eq!(out, [1.0, 0.0, 0.0, 1.0, 0.2, 0.4]);
}
