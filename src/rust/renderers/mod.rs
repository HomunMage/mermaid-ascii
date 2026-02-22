//! Renderer registry and Renderer trait.
//!
//! Mirrors Python's renderers/base.py.

pub mod ascii;
pub mod canvas;
pub mod charset;
pub mod svg;

pub use ascii::AsciiRenderer;
pub use svg::SvgRenderer;

use crate::layout::types::LayoutResult;

/// Trait for diagram renderers.
///
/// Mirrors Python's Renderer protocol: render(result: LayoutResult) -> str
pub trait Renderer {
    /// Render a laid-out graph to a string.
    fn render(&self, layout: &LayoutResult) -> String;
}
