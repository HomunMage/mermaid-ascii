//! Box-drawing character sets and junction merging logic.
//!
//! Mirrors Python's renderers/charset.py.

/// Arms represent which directions a junction character connects to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Arms {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
}

impl Arms {
    pub fn new(up: bool, down: bool, left: bool, right: bool) -> Self {
        Self {
            up,
            down,
            left,
            right,
        }
    }
}

/// Unicode box-drawing character set.
pub struct BoxChars {
    pub h: char,       // ─
    pub v: char,       // │
    pub tl: char,      // ┌
    pub tr: char,      // ┐
    pub bl: char,      // └
    pub br: char,      // ┘
    pub t_down: char,  // ┬
    pub t_up: char,    // ┴
    pub t_right: char, // ├
    pub t_left: char,  // ┤
    pub cross: char,   // ┼
    pub arrow_r: char, // →
    pub arrow_l: char, // ←
    pub arrow_d: char, // ↓
    pub arrow_u: char, // ↑
}

impl BoxChars {
    pub fn unicode() -> Self {
        Self {
            h: '─',
            v: '│',
            tl: '┌',
            tr: '┐',
            bl: '└',
            br: '┘',
            t_down: '┬',
            t_up: '┴',
            t_right: '├',
            t_left: '┤',
            cross: '┼',
            arrow_r: '→',
            arrow_l: '←',
            arrow_d: '↓',
            arrow_u: '↑',
        }
    }

    pub fn ascii() -> Self {
        Self {
            h: '-',
            v: '|',
            tl: '+',
            tr: '+',
            bl: '+',
            br: '+',
            t_down: '+',
            t_up: '+',
            t_right: '+',
            t_left: '+',
            cross: '+',
            arrow_r: '>',
            arrow_l: '<',
            arrow_d: 'v',
            arrow_u: '^',
        }
    }
}

/// Merge two junction characters by combining their arms.
///
/// TODO: implement full junction merging in Phase 4.
pub fn merge_junction(existing: char, new: char, unicode: bool) -> char {
    let _ = (existing, unicode);
    new
}
