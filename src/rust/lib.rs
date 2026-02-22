//! mermaid-ascii — Mermaid flowchart syntax to ASCII/Unicode text renderer.
//!
//! Public API: `render_dsl()`
//! Mirrors Python's api.py.

pub mod config;
pub mod layout;
pub mod parsers;
pub mod renderers;
pub mod syntax;

#[cfg(feature = "wasm")]
pub mod wasm;

use crate::config::RenderConfig;
use crate::layout::full_layout_with_config;
use crate::layout::graph::GraphIR;
use crate::parsers::parse;
use crate::renderers::{AsciiRenderer, Renderer, SvgRenderer};
use crate::syntax::types::{Direction, Graph as AstGraph};

/// Maps a direction string to the Direction enum.
///
/// Mirrors Python's `_DIRECTION_MAP` in api.py.
fn apply_direction(ast_graph: &mut AstGraph, direction: Option<&str>) -> Result<(), String> {
    let Some(dir) = direction else { return Ok(()) };
    let d = match dir.to_uppercase().as_str() {
        "LR" => Direction::LR,
        "RL" => Direction::RL,
        "TD" | "TB" => Direction::TD,
        "BT" => Direction::BT,
        other => {
            return Err(format!(
                "Unknown direction '{other}'; use LR, RL, TD, or BT"
            ));
        }
    };
    ast_graph.direction = d;
    Ok(())
}

/// Parse a Mermaid flowchart string and render it to ASCII/Unicode art.
///
/// Mirrors Python's `render_dsl()` in api.py.
pub fn render_dsl(
    src: &str,
    unicode: bool,
    padding: usize,
    direction: Option<&str>,
) -> Result<String, String> {
    let mut ast_graph = parse(src)?;
    apply_direction(&mut ast_graph, direction)?;
    let gir = GraphIR::from_ast(&ast_graph);
    if gir.node_count() == 0 && gir.subgraph_members.is_empty() {
        return Ok(String::new());
    }
    let config = RenderConfig {
        unicode,
        padding,
        direction_override: direction.map(str::to_owned),
    };
    let layout_result = full_layout_with_config(&gir, &config);
    let renderer = AsciiRenderer::new(unicode);
    Ok(renderer.render(&layout_result))
}

/// Parse a Mermaid flowchart string and render it to SVG.
///
/// Mirrors Python's `render_svg()` in api.py.
pub fn render_svg(src: &str, padding: usize, direction: Option<&str>) -> Result<String, String> {
    let mut ast_graph = parse(src)?;
    apply_direction(&mut ast_graph, direction)?;
    let gir = GraphIR::from_ast(&ast_graph);
    if gir.node_count() == 0 && gir.subgraph_members.is_empty() {
        return Ok(String::new());
    }
    let config = RenderConfig {
        unicode: true,
        padding,
        direction_override: direction.map(str::to_owned),
    };
    let layout_result = full_layout_with_config(&gir, &config);
    let renderer = SvgRenderer;
    Ok(renderer.render(&layout_result))
}
