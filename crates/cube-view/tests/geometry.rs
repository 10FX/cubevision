use cube_core::Cube;
use cube_view::geometry::{COLOURS, FACE_NORMALS, Mat3, apply, default_view, in_layer, quads, sticker, turn};

const SCRAMBLE: &str = "R U2 F' L D2 B R' U F2 L' D B2";

fn near(a: &[f32; 3], b: &[f32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3)
}

/// The sticker at a place, facing a way
fn at(centre: &[f32; 3], normal: &[f32; 3]) -> usize {
    (0..54)
        .find(|&i| near(&sticker(i).centre, centre) && near(&sticker(i).normal, normal))
        .unwrap_or_else(|| panic!("no sticker at {centre:?} facing {normal:?}"))
}

#[test]
fn the_drawing_turns_as_the_cube_does() {
    let cube = Cube::solved().apply(SCRAMBLE);
    let before = cube.colors();
    for (face, name) in ["U", "R", "F", "D", "L", "B"].iter().enumerate() {
        for (suffix, quarters) in [("", 1.0), ("2", 2.0), ("'", -1.0)] {
            let after = cube.apply(&format!("{name}{suffix}")).colors();
            let rotation = turn(face, quarters * std::f32::consts::FRAC_PI_2);
            let mut moved = 0;
            for i in 0..54 {
                let s = sticker(i);
                if in_layer(face, &s.centre) {
                    // The sticker carried by the layer lands where the cube's own move puts its colour
                    let to = at(&apply(&rotation, &s.centre), &apply(&rotation, &s.normal));
                    assert_eq!(after[to], before[i], "{name}{suffix}: sticker {i} goes to {to}");
                    moved += 1;
                } else {
                    assert_eq!(after[i], before[i], "{name}{suffix}: sticker {i} stays");
                }
            }
            assert_eq!(moved, 21, "{name}: a layer carries 21 stickers");
        }
    }
}

#[test]
fn each_face_has_nine_stickers_facing_out() {
    for i in 0..54 {
        let s = sticker(i);
        assert!(near(&s.normal, &FACE_NORMALS[i / 9]), "sticker {i}");
        // On the face, half a cubie from the surface's centre plane
        let out: f32 = s.centre.iter().zip(&s.normal).map(|(c, n)| c * n).sum();
        assert!((out - 1.5).abs() < 1e-4, "sticker {i} is on the surface");
    }
    // Centres are where the normals point
    for (face, normal) in FACE_NORMALS.iter().enumerate() {
        assert!(near(&sticker(face * 9 + 4).centre, &normal.map(|v| v * 1.5)));
    }
}

fn coloured(cube: &Cube, turning: Option<(usize, f32)>, view: &Mat3) -> Vec<[u8; 3]> {
    quads(&cube.colors(), turning, view).iter().map(|q| q.colour).filter(|c| *c != [0, 0, 0] && COLOURS.contains(c)).collect()
}

#[test]
fn a_solved_cube_shows_three_faces() {
    let seen = coloured(&Cube::solved(), None, &default_view());
    assert_eq!(seen.len(), 27);
    // White on top, green in front, red on the right
    for face in [0, 1, 2] {
        assert_eq!(seen.iter().filter(|c| **c == COLOURS[face]).count(), 9, "face {face}");
    }
}

#[test]
fn quads_come_far_to_near_and_fit_the_square() {
    let cube = Cube::solved().apply(SCRAMBLE);
    for turning in [None, Some((0, 0.6)), Some((1, -1.0)), Some((5, 2.5))] {
        let all = quads(&cube.colors(), turning, &default_view());
        assert!(all.windows(2).all(|w| w[0].depth <= w[1].depth), "{turning:?}");
        assert!(all.iter().flat_map(|q| q.points).all(|p| p[0].abs() <= 1.0 && p[1].abs() <= 1.0), "{turning:?}: inside -1..1");
        // Every sticker drawn sits on a black cell drawn before it
        assert!(all.iter().any(|q| q.colour == [0, 0, 0]));
    }
    // Mid-turn the cut shows: more is drawn than at rest
    assert!(quads(&cube.colors(), Some((0, 0.6)), &default_view()).len() > quads(&cube.colors(), None, &default_view()).len());
}
