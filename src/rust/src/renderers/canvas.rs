//! Canvas — 2D character grid for painting ASCII art.
//!
//! Mirrors Python's renderers/canvas.py.

/// A 2D character grid used as a painting surface.
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    cells: Vec<Vec<char>>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![vec![' '; width]; height],
        }
    }

    pub fn set(&mut self, x: usize, y: usize, ch: char) {
        if y < self.height && x < self.width {
            self.cells[y][x] = ch;
        }
    }

    pub fn get(&self, x: usize, y: usize) -> char {
        if y < self.height && x < self.width {
            self.cells[y][x]
        } else {
            ' '
        }
    }

}

impl std::fmt::Display for Canvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self
            .cells
            .iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        write!(f, "{}", s)
    }
}
