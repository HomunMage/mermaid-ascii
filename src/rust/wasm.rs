//! WASM bindings for mermaid-ascii.
//!
//! Exposes `render` and `renderWithOptions` to JavaScript via wasm-bindgen.

use wasm_bindgen::prelude::*;

/// Render Mermaid flowchart DSL to Unicode ASCII art with default settings.
#[wasm_bindgen]
pub fn render(src: &str) -> Result<String, JsError> {
    crate::render_dsl(src, true, 1, None).map_err(|e| JsError::new(&e))
}

/// Render Mermaid flowchart DSL with full control over options.
///
/// - `unicode`: true for Unicode box-drawing chars, false for plain ASCII
/// - `padding`: spaces inside node borders
/// - `direction`: "LR", "RL", "TD", "BT", or empty string for default
#[wasm_bindgen(js_name = "renderWithOptions")]
pub fn render_with_options(
    src: &str,
    unicode: bool,
    padding: usize,
    direction: &str,
) -> Result<String, JsError> {
    let dir = if direction.is_empty() {
        None
    } else {
        Some(direction)
    };
    crate::render_dsl(src, unicode, padding, dir).map_err(|e| JsError::new(&e))
}

/// Render Mermaid flowchart DSL to SVG string.
///
/// - `padding`: spaces inside node borders
/// - `direction`: "LR", "RL", "TD", "BT", or empty string for default
#[wasm_bindgen(js_name = "renderSvg")]
pub fn render_svg(src: &str, padding: usize, direction: &str) -> Result<String, JsError> {
    let dir = if direction.is_empty() {
        None
    } else {
        Some(direction)
    };
    crate::render_svg(src, padding, dir).map_err(|e| JsError::new(&e))
}
