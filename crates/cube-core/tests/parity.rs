//! The same results as ChronoCube's TypeScript, on the fixtures its script writes (scripts/vision/fixtures.test.ts)

use cube_core::*;
use serde_json::Value;

fn fixtures() -> Value {
    serde_json::from_str(include_str!("fixtures/core.json")).unwrap()
}

fn family(name: &str) -> Family {
    use Family::*;
    match name { "U" => U, "D" => D, "F" => F, "B" => B, "R" => R, "L" => L, "M" => M, "E" => E, "S" => S, "x" => X, "y" => Y, "z" => Z, _ => panic!("{name}") }
}

fn style(name: &str) -> Style {
    match name { "face" => Style::Face, "wide" => Style::Wide, "lower" => Style::Lower, "slice" => Style::Slice, "range" => Style::Range, _ => panic!("{name}") }
}

fn move_of(v: &Value) -> Option<Move> {
    if v.is_null() {
        return None;
    }
    let n = |k: &str| v[k].as_u64().unwrap();
    Some(Move { family: family(v["family"].as_str().unwrap()), from: n("from") as u32, to: n("to") as u32, turns: n("turns") as u8, style: style(v["style"].as_str().unwrap()) })
}

#[test]
fn notation_matches_typescript() {
    let f = fixtures();
    for case in f["moves"].as_array().unwrap() {
        let token = case["token"].as_str().unwrap();
        let parsed = parse_move(token);
        assert_eq!(parsed, move_of(&case["move"]), "{token}");
        if let Some(m) = parsed {
            assert_eq!(format_move(&m), case["formatted"].as_str().unwrap(), "{token}");
        }
    }
    for case in f["inverts"].as_array().unwrap() {
        assert_eq!(invert_alg(case["alg"].as_str().unwrap()), case["inverse"].as_str().unwrap(), "{}", case["alg"]);
    }
}

#[test]
fn parentheses_only_group() {
    assert_eq!(tokenize("(R U) R'"), ["R", "U", "R'"]);
    assert_eq!(parse_alg("R foo U").invalid, ["foo"]);
}

fn numbers(v: &Value) -> Vec<u64> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
}

#[test]
fn cube_matches_typescript() {
    for case in fixtures()["cubes"].as_array().unwrap() {
        let alg = case["alg"].as_str().unwrap();
        let cube = Cube::solved().apply(alg);
        let stickers: Vec<u64> = cube.stickers().iter().map(|&s| s as u64).collect();
        assert_eq!(stickers, numbers(&case["stickers"]), "{alg}");
        let c = numbers(&case["cubie"][3]);
        let at = &case["cubie"];
        let (face, row, col) = (at[0].as_u64().unwrap() as usize, at[1].as_u64().unwrap() as usize, at[2].as_u64().unwrap() as usize);
        assert_eq!(cube.cubie_colors(face, row, col).iter().map(|&x| x as u64).collect::<Vec<_>>(), c, "{alg}");
    }
}

#[test]
fn layers_beyond_the_cube_turn_nothing() {
    assert_eq!(Cube::solved().apply("4R"), Cube::solved());
    assert_eq!(Cube::solved().apply("4294967295R 99999999999Rw"), Cube::solved().apply("x"), "every layer: the whole cube");
}

#[test]
fn unknown_tokens_are_skipped() {
    assert_eq!(Cube::solved().apply("R foo U"), Cube::solved().apply("R U"));
}
