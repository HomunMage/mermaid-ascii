//! mermaid-ascii — Mermaid flowchart syntax to ASCII/Unicode text renderer.
//!
//! .hom source files are compiled to .rs by build.rs → homunc into OUT_DIR.
//! Hand-written Rust helpers live in dep/.
//!
//! Modules (uncomment as .hom files compile cleanly with homunc):
//!   mod types;          // Direction, NodeShape, EdgeType, Node, Edge, Subgraph, Graph
//!   mod config;         // RenderConfig
//!   mod layout_types;   // Point, LayoutNode, RoutedEdge, LayoutResult
//!   mod charset;        // BoxChars, Arms, CharSet
//!   mod canvas;         // Rect, Canvas (2D char grid)
//!   mod parser;         // Cursor tokenizer + flowchart recursive descent
//!   mod pathfinder;     // A* orthogonal edge routing
//!   mod layout;         // Sugiyama 8-phase algorithm
//!   mod render;         // ASCII renderer 7 phases

// graph/ — hand-written Rust helper modules (petgraph wrapper + mutable state types)
#[path = "graph/mod.rs"]
pub mod graph;

// Generated .hom modules live in OUT_DIR. Uncomment as they compile cleanly.
// mod types { include!(concat!(env!("OUT_DIR"), "/types.rs")); }
// mod config { include!(concat!(env!("OUT_DIR"), "/config.rs")); }
// mod layout_types { include!(concat!(env!("OUT_DIR"), "/layout_types.rs")); }
// mod charset { include!(concat!(env!("OUT_DIR"), "/charset.rs")); }
// mod canvas { include!(concat!(env!("OUT_DIR"), "/canvas.rs")); }
// mod parser { include!(concat!(env!("OUT_DIR"), "/parser.rs")); }
// mod pathfinder { include!(concat!(env!("OUT_DIR"), "/pathfinder.rs")); }
// mod layout { include!(concat!(env!("OUT_DIR"), "/layout.rs")); }
// mod render { include!(concat!(env!("OUT_DIR"), "/render.rs")); }

/// Parse a Mermaid flowchart string and render it to ASCII/Unicode art.
///
/// Full implementation wires: parse → GraphIR → layout → renderer.
pub fn render_dsl(
    _src: &str,
    _unicode: bool,
    _padding: usize,
    _direction: Option<&str>,
) -> Result<String, String> {
    Err("not yet implemented — modules are being ported to .hom".to_string())
}
