//! Moves as cubers write them (`notation.ts`)

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family { U, D, F, B, R, L, M, E, S, X, Y, Z }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Style { Face, Wide, Lower, Slice, Range }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move { pub family: Family, pub from: u32, pub to: u32, pub turns: u8, pub style: Style }

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedAlg { pub moves: Vec<Move>, pub invalid: Vec<String> }

impl Family {
    /// The letter as written: faces and slices in capitals, rotations in lower case
    pub fn letter(self) -> char {
        use Family::*;
        match self { U => 'U', D => 'D', F => 'F', B => 'B', R => 'R', L => 'L', M => 'M', E => 'E', S => 'S', X => 'x', Y => 'y', Z => 'z' }
    }

    fn of(letter: char) -> Option<Family> {
        use Family::*;
        Some(match letter { 'U' => U, 'D' => D, 'F' => F, 'B' => B, 'R' => R, 'L' => L, 'M' => M, 'E' => E, 'S' => S, 'x' => X, 'y' => Y, 'z' => Z, _ => return None })
    }
}

/// JavaScript's `\s`, which splits algorithms in ChronoCube
fn js_space(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}')
}

/// Leading ASCII digits of `chars[at..]`: their value (saturated), their value mod 4, how many
fn digits(chars: &[char], at: usize) -> (u64, u64, usize) {
    let (mut value, mut quarter, mut n) = (0u64, 0u64, 0);
    while let Some(d) = chars.get(at + n).and_then(|c| c.to_digit(10)) {
        value = value.saturating_mul(10).saturating_add(d as u64);
        quarter = (quarter * 10 + d as u64) % 4;
        n += 1;
    }
    (value, quarter, n)
}

/// One move token ("R", "U2'", "3Rw", "2-3R", "M'", "y2"), or None if it is not a move
pub fn parse_move(token: &str) -> Option<Move> {
    // Typographic quotes for primes; R'2 is sometimes written for R2'
    let mut chars: Vec<char> = token.chars().map(|c| if matches!(c, '’' | '‘' | '`' | '´') { '\'' } else { c }).collect();
    let tail = chars.iter().rev().take_while(|c| c.is_ascii_digit()).count();
    if tail > 0 && chars.len() >= tail + 2 && chars[chars.len() - tail - 1] == '\'' {
        let quote = chars.len() - tail - 1;
        chars.remove(quote);
        chars.push('\'');
    }
    // ^(?:(\d+)(?:-(\d+))?)?([UDFBRL]w?|[udfbrl]|[MESxyz])(\d*)('?)$
    let mut at = 0;
    let (first, _, n) = digits(&chars, 0);
    let first = (n > 0).then_some(first);
    at += n;
    let mut second = None;
    if first.is_some() && chars.get(at) == Some(&'-') {
        let (b, _, m) = digits(&chars, at + 1);
        if m == 0 {
            return None;
        }
        second = Some(b);
        at += 1 + m;
    }
    let letter = *chars.get(at)?;
    at += 1;
    let wide = letter.is_ascii_uppercase() && "UDFBRL".contains(letter) && chars.get(at) == Some(&'w');
    if wide {
        at += 1;
    }
    let lower = "udfbrl".contains(letter);
    let family = match letter {
        'U' | 'D' | 'F' | 'B' | 'R' | 'L' | 'u' | 'd' | 'f' | 'b' | 'r' | 'l' => Family::of(letter.to_ascii_uppercase())?,
        'M' | 'E' | 'S' | 'x' | 'y' | 'z' => Family::of(letter)?,
        _ => return None,
    };
    let (_, quarter, m) = digits(&chars, at);
    at += m;
    let prime = chars.get(at) == Some(&'\'');
    if prime {
        at += 1;
    }
    if at != chars.len() {
        return None;
    }
    let amount = if m > 0 { quarter } else { 1 };
    let turns = if prime { (4 - amount) % 4 } else { amount } as u8;
    if turns == 0 {
        return None;
    }
    if first == Some(0) || matches!((first, second), (Some(a), Some(b)) if b < a) {
        return None;
    }
    let layer = |v: u64| v.min(u32::MAX as u64) as u32;
    if matches!(family, Family::M | Family::E | Family::S | Family::X | Family::Y | Family::Z) {
        if first.is_some() {
            return None;
        }
        return Some(Move { family, from: 0, to: 0, turns, style: Style::Face });
    }
    if let (Some(a), Some(b)) = (first, second) {
        return Some(Move { family, from: layer(a), to: layer(b), turns, style: Style::Range });
    }
    if wide || lower {
        let style = if wide { Style::Wide } else { Style::Lower };
        return Some(Move { family, from: 1, to: first.map_or(2, layer), turns, style });
    }
    Some(match first {
        Some(a) => Move { family, from: layer(a), to: layer(a), turns, style: if a == 1 { Style::Face } else { Style::Slice } },
        None => Move { family, from: 1, to: 1, turns, style: Style::Face },
    })
}

pub fn format_move(m: &Move) -> String {
    let amount = match m.turns { 1 => "", 2 => "2", _ => "'" };
    let family = m.family.letter();
    let width = |to: u32| if to == 2 { String::new() } else { to.to_string() };
    match m.style {
        Style::Wide => format!("{}{family}w{amount}", width(m.to)),
        Style::Lower => format!("{}{}{amount}", width(m.to), family.to_ascii_lowercase()),
        Style::Slice => format!("{}{family}{amount}", m.from),
        Style::Range => format!("{}-{}{family}{amount}", m.from, m.to),
        Style::Face => format!("{family}{amount}"),
    }
}

/// Grouping parentheses are ignored: "(R U R' U')" reads like "R U R' U'"
pub fn tokenize(alg: &str) -> Vec<&str> {
    alg.split(|c: char| js_space(c) || c == '(' || c == ')').filter(|t| !t.is_empty()).collect()
}

pub fn parse_alg(alg: &str) -> ParsedAlg {
    let mut parsed = ParsedAlg::default();
    for token in tokenize(alg) {
        match parse_move(token) {
            Some(m) => parsed.moves.push(m),
            None => parsed.invalid.push(token.to_string()),
        }
    }
    parsed
}

pub fn format_alg(moves: &[Move]) -> String {
    moves.iter().map(format_move).collect::<Vec<_>>().join(" ")
}

pub fn invert_alg(alg: &str) -> String {
    let moves: Vec<Move> = parse_alg(alg).moves.into_iter().rev().map(|m| Move { turns: 4 - m.turns, ..m }).collect();
    format_alg(&moves)
}
