//! Canvas — 2D character grid for painting ASCII art.
//!
//! Mirrors Python's renderers/canvas.py.

use super::charset::{Arms, BoxChars, CharSet};

// ─── Rect ─────────────────────────────────────────────────────────────────────

/// A rectangle in character-grid coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub width: i64,
    pub height: i64,
}

impl Rect {
    pub fn new(x: i64, y: i64, width: i64, height: i64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn right(&self) -> i64 {
        self.x + self.width
    }

    pub fn bottom(&self) -> i64 {
        self.y + self.height
    }
}

// ─── Canvas ───────────────────────────────────────────────────────────────────

/// A 2D character grid used as a painting surface.
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub charset: CharSet,
    cells: Vec<Vec<char>>,
}

impl Canvas {
    pub fn new(width: usize, height: usize, charset: CharSet) -> Self {
        Self {
            width,
            height,
            charset,
            cells: vec![vec![' '; width]; height],
        }
    }

    pub fn get(&self, col: usize, row: usize) -> char {
        if row < self.height && col < self.width {
            self.cells[row][col]
        } else {
            ' '
        }
    }

    pub fn set(&mut self, col: usize, row: usize, ch: char) {
        if row < self.height && col < self.width {
            self.cells[row][col] = ch;
        }
    }

    /// Set a cell, merging junction characters if both old and new are box-drawing chars.
    pub fn set_merge(&mut self, col: usize, row: usize, ch: char) {
        if row >= self.height || col >= self.width {
            return;
        }
        let existing = self.cells[row][col];
        let ea = Arms::from_char(existing);
        let na = Arms::from_char(ch);
        if let (Some(e), Some(n)) = (ea, na) {
            self.cells[row][col] = e.merge(n).to_char(self.charset);
        } else {
            self.cells[row][col] = ch;
        }
    }

    /// Draw a horizontal line from x1 to x2 (inclusive) at row y.
    pub fn hline(&mut self, y: usize, x1: usize, x2: usize, ch: char) {
        let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
        for col in lo..=hi {
            self.set_merge(col, y, ch);
        }
    }

    /// Draw a vertical line from y1 to y2 (inclusive) at column x.
    pub fn vline(&mut self, x: usize, y1: usize, y2: usize, ch: char) {
        let (lo, hi) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
        for row in lo..=hi {
            self.set_merge(x, row, ch);
        }
    }

    /// Draw a box outline using box-drawing characters from BoxChars.
    pub fn draw_box(&mut self, rect: Rect, bc: &BoxChars) {
        if rect.width < 2 || rect.height < 2 {
            return;
        }
        let x0 = rect.x as usize;
        let y0 = rect.y as usize;
        let x1 = (rect.x + rect.width - 1) as usize;
        let y1 = (rect.y + rect.height - 1) as usize;
        self.set(x0, y0, bc.top_left);
        self.set(x1, y0, bc.top_right);
        self.set(x0, y1, bc.bottom_left);
        self.set(x1, y1, bc.bottom_right);
        for col in (x0 + 1)..x1 {
            self.set(col, y0, bc.horizontal);
            self.set(col, y1, bc.horizontal);
        }
        for row in (y0 + 1)..y1 {
            self.set(x0, row, bc.vertical);
            self.set(x1, row, bc.vertical);
        }
    }

    /// Write a string starting at (col, row).
    pub fn write_str(&mut self, col: usize, row: usize, s: &str) {
        for (i, ch) in s.chars().enumerate() {
            let c = col + i;
            if c >= self.width || row >= self.height {
                break;
            }
            self.cells[row][c] = ch;
        }
    }

    /// Render the canvas to a string, trimming trailing whitespace per line.
    pub fn render_to_string(&self) -> String {
        let mut lines: Vec<String> = self
            .cells
            .iter()
            .map(|row| row.iter().collect::<String>().trim_end().to_string())
            .collect();
        // Trim trailing empty lines
        while lines.last().map(|l: &String| l.is_empty()).unwrap_or(false) {
            lines.pop();
        }
        let mut out = lines.join("\n");
        out.push('\n');
        out
    }
}

impl std::fmt::Display for Canvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render_to_string())
    }
}

#[cfg(test)]
#[path = "../../../tests/rust/test_renderers_canvas.rs"]
mod tests;
