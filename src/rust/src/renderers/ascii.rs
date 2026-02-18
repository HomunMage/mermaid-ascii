//! ASCII/Unicode renderer for mermaid-ascii.
//!
//! Mirrors Python's renderers/ascii.py.

use super::Renderer;
use crate::layout::graph::GraphIR;
use crate::layout::types::LayoutResult;

/// Renders a graph layout to ASCII/Unicode text using box-drawing characters.
pub struct AsciiRenderer {
    pub unicode: bool,
}

impl AsciiRenderer {
    pub fn new(unicode: bool) -> Self {
        Self { unicode }
    }
}

impl Renderer for AsciiRenderer {
    fn render(&self, _gir: &GraphIR, _layout: &LayoutResult) -> String {
        // TODO: implement in Phase 6
        String::new()
    }
}
