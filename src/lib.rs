pub mod ast;
pub mod graph;
pub mod layout;
pub mod parser;
pub mod render;

/// Parse a DSL string and render it to a text graph.
///
/// Returns the rendered ASCII/Unicode string on success, or a parse/layout
/// error message on failure.
pub fn render_dsl(src: &str, unicode: bool) -> Result<String, String> {
    let ast_graph = parser::parse(src)?;
    let gir = graph::GraphIR::from_ast(&ast_graph);
    if gir.node_count() == 0 && gir.subgraph_members.is_empty() {
        return Ok(String::new());
    }
    let (layout_nodes, routed_edges) = layout::full_layout(&gir);
    Ok(render::render(&gir, &layout_nodes, &routed_edges, unicode))
}

/// Parse a DSL string and render it with a custom padding value.
pub fn render_dsl_padded(src: &str, unicode: bool, padding: usize) -> Result<String, String> {
    let ast_graph = parser::parse(src)?;
    let gir = graph::GraphIR::from_ast(&ast_graph);
    if gir.node_count() == 0 && gir.subgraph_members.is_empty() {
        return Ok(String::new());
    }
    let (layout_nodes, routed_edges) = layout::full_layout_with_padding(&gir, padding);
    Ok(render::render(&gir, &layout_nodes, &routed_edges, unicode))
}
