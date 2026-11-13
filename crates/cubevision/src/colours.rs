//! Colours as a cuber names them, and how a cube is held

use anyhow::{Result, bail};
use cube_core::Cube;
use cube_core::goals::ORIENTATIONS;

/// The colours of the faces U R F D L B of a cube held white on top, green in front
pub const COLOURS: [&str; 6] = ["white", "red", "green", "yellow", "orange", "blue"];

/// A colour by its name or its first letter (white, w, W…): its face's index
pub fn colour(name: &str) -> Result<u8> {
    let wanted = name.trim().to_lowercase();
    match COLOURS.iter().position(|c| !wanted.is_empty() && (*c == wanted || c[..1] == wanted)) {
        Some(index) => Ok(index as u8),
        None => bail!("unknown colour {name:?}: {}", COLOURS.join(", ")),
    }
}

/// How a cube is held with these colours on top and in front: the rotation from white on top, green in front
pub fn hold(top: u8, front: u8) -> Result<&'static str> {
    for rotation in ORIENTATIONS {
        let held = Cube::solved().apply(rotation);
        if (held.color(0, 1, 1), held.color(2, 1, 1)) == (top, front) {
            return Ok(rotation);
        }
    }
    bail!("no cube has {} on top and {} in front: they are the same face or opposite ones", COLOURS[top as usize], COLOURS[front as usize])
}

/// A scramble a cube can be turned by: at least one move, every token a move
pub fn scramble(text: &str) -> Result<()> {
    let mut tokens = text.split_whitespace().peekable();
    if tokens.peek().is_none() {
        bail!("the scramble is empty: the moves are found from the scrambled cube");
    }
    match tokens.find(|token| cube_core::parse_move(token).is_none()) {
        Some(token) => bail!("the scramble holds {token:?}, which is not a move (R, U', F2, Rw, x…)"),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scramble_is_made_of_moves() {
        assert!(scramble("R U2 F' Rw x M2 D’").is_ok());
        assert!(scramble("  ").unwrap_err().to_string().contains("empty"));
        for bad in ["R Q7 banana", "R U, F", "hello"] {
            let error = scramble(bad).unwrap_err().to_string();
            assert!(error.contains("not a move"), "{bad}: {error}");
        }
        assert!(scramble("R Q7 banana").unwrap_err().to_string().contains("Q7"));
    }

    #[test]
    fn colours_are_named_or_lettered() {
        assert_eq!((colour("white").unwrap(), colour("G").unwrap(), colour(" Blue ").unwrap(), colour("o").unwrap()), (0, 2, 5, 4));
        assert!(colour("pink").unwrap_err().to_string().contains("orange"));
        assert!(colour("").is_err());
    }

    #[test]
    fn a_hold_puts_the_colours_asked_on_top_and_in_front() {
        assert_eq!(hold(0, 2).unwrap(), "");
        for top in 0..6u8 {
            for front in 0..6u8 {
                let found = hold(top, front);
                if top == front || top == (front + 3) % 6 {
                    assert!(found.unwrap_err().to_string().contains("on top"));
                } else {
                    let held = Cube::solved().apply(found.unwrap());
                    assert_eq!((held.color(0, 1, 1), held.color(2, 1, 1)), (top, front));
                }
            }
        }
        // Yellow on top, green in front: the cube turned over
        assert_eq!(hold(3, 2).unwrap(), "x2 y2");
    }
}
