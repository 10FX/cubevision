//! The cube's stickers in space, a layer mid-turn, the whole seen from a side and flattened for the screen.
//!
//! The cube is 3 wide, centred on the origin: x to the right (R), y up (U), z towards the viewer (F). Stickers
//! are numbered as cube-core does: faces U R F D L B, nine each, row by row.

pub type Mat3 = [[f32; 3]; 3];

/// Outward direction of the faces U R F D L B
pub const FACE_NORMALS: [[f32; 3]; 6] =
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];

/// The colours of the faces U R F D L B, the WCA's: white, red, green, yellow, orange, blue
pub const COLOURS: [[u8; 3]; 6] = [[245, 245, 245], [200, 36, 44], [0, 158, 84], [255, 214, 0], [255, 122, 0], [0, 84, 190]];

/// A sticker's place: its centre and the way it faces
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sticker {
    pub centre: [f32; 3],
    pub normal: [f32; 3],
}

/// Where sticker `index` (0..54) sits on a cube at rest
pub fn sticker(index: usize) -> Sticker {
    let (face, row, col) = (index / 9, (index % 9 / 3) as f32, (index % 3) as f32);
    let centre = match face {
        0 => [col - 1.0, 1.5, row - 1.0],  // seen from above, the front at the bottom
        1 => [1.5, 1.0 - row, 1.0 - col],  // seen from the right, the front on the left
        2 => [col - 1.0, 1.0 - row, 1.5],  // seen from the front
        3 => [col - 1.0, -1.5, 1.0 - row], // seen from below, the front at the top
        4 => [-1.5, 1.0 - row, col - 1.0], // seen from the left, the front on the right
        _ => [1.0 - col, 1.0 - row, -1.5], // seen from behind
    };
    Sticker { centre, normal: FACE_NORMALS[face] }
}

fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn apply(m: &Mat3, v: &[f32; 3]) -> [f32; 3] {
    [dot(&m[0], v), dot(&m[1], v), dot(&m[2], v)]
}

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A rotation about a unit axis, counter-clockwise seen from its tip
pub fn rotation(axis: &[f32; 3], angle: f32) -> Mat3 {
    let (s, c) = angle.sin_cos();
    let [x, y, z] = *axis;
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

/// Whether a point of the cube's surface is carried by a turn of this face
pub fn in_layer(face: usize, point: &[f32; 3]) -> bool {
    dot(&FACE_NORMALS[face], point) > 0.5
}

/// A face's layer turned clockwise (seen from outside) by `angle` radians
pub fn turn(face: usize, angle: f32) -> Mat3 {
    rotation(&FACE_NORMALS[face], -angle)
}

/// The cube seen from the front, a little from above and from the right: white on top, green in front, red on the right
pub fn default_view() -> Mat3 {
    view(-30.0, 25.0)
}

/// The cube turned `yaw` degrees about the vertical, then tilted `pitch` degrees towards the viewer
pub fn view(yaw: f32, pitch: f32) -> Mat3 {
    mul(&rotation(&[1.0, 0.0, 0.0], pitch.to_radians()), &rotation(&[0.0, 1.0, 0.0], yaw.to_radians()))
}

/// A flat piece to draw: its corners on a square screen (-1..1, y down), its colour, how near it is
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub points: [[f32; 2]; 4],
    pub colour: [u8; 3],
    pub depth: f32,
}

/// How far the eye is from the cube's centre, and the share of the screen's half-side a unit takes
const EYE: f32 = 11.0;
const UNIT: f32 = 1.0 / 3.7;
/// Half-side of a sticker, in cubies: the rest of its cell is the black plastic
const STICKER: f32 = 0.44;

/// Two directions along a face, at right angles
fn along(normal: &[f32; 3]) -> ([f32; 3], [f32; 3]) {
    if normal[0] != 0.0 {
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    } else if normal[1] != 0.0 {
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    } else {
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    }
}

/// A square in space, seen: None when it faces away
fn seen(centre: &[f32; 3], normal: &[f32; 3], half: f32, colour: [u8; 3], placed: &Mat3) -> Option<Quad> {
    let (a, b) = along(normal);
    let (c, n) = (apply(placed, centre), apply(placed, normal));
    if dot(&n, &[-c[0], -c[1], EYE - c[2]]) <= 0.0 {
        return None;
    }
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(u, v): (f32, f32)| {
        let p = apply(placed, &std::array::from_fn(|k| centre[k] + half * (u * a[k] + v * b[k])));
        let scale = EYE / (EYE - p[2]) * UNIT;
        [p[0] * scale, -p[1] * scale]
    });
    Some(Quad { points: corners, colour, depth: c[2] })
}

/// The cube as flat pieces, the farthest first: each sticker on its black cell, and the black cut of a layer
/// mid-turn. `colours`: the 54 stickers' colours (0..6); `turning`: a face and how far its layer has turned
/// (radians, clockwise); `view`: how the whole cube is turned.
pub fn quads(colours: &[u8; 54], turning: Option<(usize, f32)>, view: &Mat3) -> Vec<Quad> {
    let turned = turning.map(|(face, angle)| (face, mul(view, &turn(face, angle))));
    let mut out = Vec::with_capacity(120);
    for (i, &colour) in colours.iter().enumerate() {
        let s = sticker(i);
        let placed = match &turned {
            Some((face, placed)) if in_layer(*face, &s.centre) => placed,
            _ => view,
        };
        if let Some(cell) = seen(&s.centre, &s.normal, 0.5, [0, 0, 0], placed) {
            out.push(cell);
            out.extend(seen(&s.centre, &s.normal, STICKER, COLOURS[colour.min(5) as usize], placed));
        }
    }
    if let Some((face, placed)) = &turned {
        // The cut between the layer and the rest: black on both sides
        let normal = FACE_NORMALS[*face];
        let centre = normal.map(|v| v * 0.5);
        out.extend(seen(&centre, &normal, 1.5, [0, 0, 0], view));
        out.extend(seen(&centre, &normal.map(|v| -v), 1.5, [0, 0, 0], placed));
    }
    // Stable: a sticker stays after its cell
    out.sort_by(|a, b| a.depth.total_cmp(&b.depth));
    out
}
