//! Box-drawing character sets and junction merging logic.
//!
//! Mirrors Python's renderers/charset.py.

// ─── CharSet ─────────────────────────────────────────────────────────────────

/// Which character set to use for box-drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CharSet {
    #[default]
    Unicode,
    Ascii,
}

// ─── BoxChars ─────────────────────────────────────────────────────────────────

/// Unicode or ASCII box-drawing character set.
pub struct BoxChars {
    pub top_left: char,
    pub top_right: char,
    pub bottom_left: char,
    pub bottom_right: char,
    pub horizontal: char,
    pub vertical: char,
    pub tee_right: char, // ├ left-T opening right
    pub tee_left: char,  // ┤ right-T opening left
    pub tee_down: char,  // ┬ top-T opening down
    pub tee_up: char,    // ┴ bottom-T opening up
    pub cross: char,     // ┼
    pub arrow_right: char,
    pub arrow_left: char,
    pub arrow_down: char,
    pub arrow_up: char,
}

impl BoxChars {
    pub fn unicode() -> Self {
        Self {
            top_left: '┌',
            top_right: '┐',
            bottom_left: '└',
            bottom_right: '┘',
            horizontal: '─',
            vertical: '│',
            tee_right: '├',
            tee_left: '┤',
            tee_down: '┬',
            tee_up: '┴',
            cross: '┼',
            arrow_right: '►',
            arrow_left: '◄',
            arrow_down: '▼',
            arrow_up: '▲',
        }
    }

    pub fn ascii() -> Self {
        Self {
            top_left: '+',
            top_right: '+',
            bottom_left: '+',
            bottom_right: '+',
            horizontal: '-',
            vertical: '|',
            tee_right: '+',
            tee_left: '+',
            tee_down: '+',
            tee_up: '+',
            cross: '+',
            arrow_right: '>',
            arrow_left: '<',
            arrow_down: 'v',
            arrow_up: '^',
        }
    }

    pub fn for_charset(cs: CharSet) -> Self {
        match cs {
            CharSet::Unicode => Self::unicode(),
            CharSet::Ascii => Self::ascii(),
        }
    }
}

// ─── Arms ────────────────────────────────────────────────────────────────────

/// Which arms of a junction cell are active.
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

    /// Decode a box-drawing character into its arms. Returns None for non-junction chars.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '─' | '-' => Some(Self::new(false, false, true, true)),
            '│' | '|' => Some(Self::new(true, true, false, false)),
            '┌' => Some(Self::new(false, true, false, true)),
            '┐' => Some(Self::new(false, true, true, false)),
            '└' => Some(Self::new(true, false, false, true)),
            '┘' => Some(Self::new(true, false, true, false)),
            '├' => Some(Self::new(true, true, false, true)),
            '┤' => Some(Self::new(true, true, true, false)),
            '┬' => Some(Self::new(false, true, true, true)),
            '┴' => Some(Self::new(true, false, true, true)),
            '┼' | '+' => Some(Self::new(true, true, true, true)),
            _ => None,
        }
    }

    /// Merge two Arms by OR-ing each direction.
    pub fn merge(self, other: Self) -> Self {
        Self {
            up: self.up || other.up,
            down: self.down || other.down,
            left: self.left || other.left,
            right: self.right || other.right,
        }
    }

    /// Convert Arms to the appropriate box-drawing character for the given CharSet.
    pub fn to_char(self, cs: CharSet) -> char {
        let bc = BoxChars::for_charset(cs);
        match (self.up, self.down, self.left, self.right) {
            (false, false, false, false) => ' ',
            (false, false, true, true) => bc.horizontal,
            (true, true, false, false) => bc.vertical,
            (false, true, false, true) => bc.top_left,
            (false, true, true, false) => bc.top_right,
            (true, false, false, true) => bc.bottom_left,
            (true, false, true, false) => bc.bottom_right,
            (true, true, false, true) => bc.tee_right,
            (true, true, true, false) => bc.tee_left,
            (false, true, true, true) => bc.tee_down,
            (true, false, true, true) => bc.tee_up,
            (true, true, true, true) => bc.cross,
            // Single-arm fallbacks
            (true, false, false, false) => bc.vertical,
            (false, true, false, false) => bc.vertical,
            (false, false, true, false) => bc.horizontal,
            (false, false, false, true) => bc.horizontal,
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/rust/test_renderers_charset.rs"]
mod tests;
