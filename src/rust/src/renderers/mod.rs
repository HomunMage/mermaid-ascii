//! Renderer registry and Renderer trait.
//!
//! Mirrors Python's renderers/base.py.

pub mod ascii;
pub mod canvas;
pub mod charset;

pub use ascii::AsciiRenderer;

use crate::layout::graph::GraphIR;
use crate::layout::types::LayoutResult;

/// Trait for diagram renderers.
pub trait Renderer {
    /// Render a laid-out graph to a string.
    fn render(&self, gir: &GraphIR, layout: &LayoutResult) -> String;
}
