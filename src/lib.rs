//! mermaid-ascii — Mermaid flowchart syntax to ASCII/Unicode text renderer.
//!
//! .hom source files are compiled to .rs by build.rs → homunc into OUT_DIR.
//! Hand-written Rust helpers live in graph/.

#![allow(
    unused_variables,
    unused_mut,
    dead_code,
    unused_imports,
    unused_macros,
    unused_assignments
)]
#![allow(non_snake_case)]
#![allow(
    clippy::clone_on_copy,
    clippy::redundant_field_names,
    clippy::assign_op_pattern,
    clippy::no_effect,
    clippy::unused_unit,
    clippy::needless_return,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::ptr_arg,
    clippy::comparison_to_empty,
    clippy::redundant_clone,
    clippy::inherent_to_string,
    clippy::useless_conversion,
    clippy::unnecessary_to_owned,
    clippy::too_many_arguments,
    clippy::redundant_closure,
    clippy::bool_comparison,
    clippy::while_immutable_condition,
    clippy::identity_op,
    clippy::almost_swapped,
    clippy::needless_borrow,
    clippy::op_ref,
    clippy::iter_overeager_cloned
)]

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// ── Homun runtime (builtin + std + re + heap) ──────────────────────────────
#[macro_use]
pub mod runtime {
    include!(concat!(env!("OUT_DIR"), "/runtime.rs"));
}

// graph/ — hand-written Rust helper modules (petgraph wrapper + mutable state types)
#[path = "graph/mod.rs"]
pub mod graph;

// Generated .hom modules live in OUT_DIR.
mod types {
    use crate::runtime::*;
    include!(concat!(env!("OUT_DIR"), "/types.rs"));
}
mod config {
    use crate::runtime::*;
    include!(concat!(env!("OUT_DIR"), "/config.rs"));
}
mod layout_types {
    use crate::runtime::*;
    include!(concat!(env!("OUT_DIR"), "/layout_types.rs"));
}
mod charset {
    use crate::runtime::*;
    include!(concat!(env!("OUT_DIR"), "/charset.rs"));
}
mod canvas {
    use crate::runtime::*;
    include!(concat!(env!("OUT_DIR"), "/canvas.rs"));
}
mod parser {
    use crate::runtime::*;
    include!(concat!(env!("OUT_DIR"), "/parser.rs"));
}
mod pathfinder {
    use crate::runtime::*;
    include!(concat!(env!("OUT_DIR"), "/pathfinder.rs"));
}
mod layout {
    use crate::runtime::*;
    include!(concat!(env!("OUT_DIR"), "/layout.rs"));
}

// ── Rust-native parser (bypasses broken .hom parser due to .clone() semantics) ──

mod rust_parser {
    //! Recursive descent parser for Mermaid flowchart syntax.
    //! Produces the same types as the .hom parser module.
    use super::parser;

    struct Cursor {
        src: Vec<char>,
        pos: usize,
    }

    impl Cursor {
        fn new(s: &str) -> Self {
            Cursor {
                src: s.chars().collect(),
                pos: 0,
            }
        }
        fn eof(&self) -> bool {
            self.pos >= self.src.len()
        }
        fn peek_str(&self, s: &str) -> bool {
            let chars: Vec<char> = s.chars().collect();
            if self.pos + chars.len() > self.src.len() {
                return false;
            }
            for (i, ch) in chars.iter().enumerate() {
                if self.src[self.pos + i] != *ch {
                    return false;
                }
            }
            true
        }
        fn consume_str(&mut self, s: &str) -> bool {
            if self.peek_str(s) {
                self.pos += s.chars().count();
                true
            } else {
                false
            }
        }
        fn ch(&self) -> char {
            if self.eof() { '\0' } else { self.src[self.pos] }
        }
        fn skip_ws(&mut self) {
            loop {
                if self.pos < self.src.len() && (self.ch() == ' ' || self.ch() == '\t') {
                    self.pos += 1;
                } else if self.peek_str("%%") {
                    while self.pos < self.src.len() && self.ch() != '\n' {
                        self.pos += 1;
                    }
                } else {
                    break;
                }
            }
        }
        fn skip_ws_and_newlines(&mut self) {
            loop {
                if self.pos < self.src.len() && matches!(self.ch(), ' ' | '\t' | '\n' | '\r') {
                    self.pos += 1;
                } else if self.peek_str("%%") {
                    while self.pos < self.src.len() && self.ch() != '\n' {
                        self.pos += 1;
                    }
                } else {
                    break;
                }
            }
        }
        fn consume_newline(&mut self) -> bool {
            if self.peek_str("\r\n") {
                self.pos += 2;
                true
            } else if self.pos < self.src.len() && (self.ch() == '\n' || self.ch() == '\r') {
                self.pos += 1;
                true
            } else {
                false
            }
        }
        fn match_node_id(&mut self) -> String {
            let start = self.pos;
            if self.pos < self.src.len() && (self.ch().is_ascii_alphabetic() || self.ch() == '_') {
                self.pos += 1;
                while self.pos < self.src.len()
                    && (self.ch().is_ascii_alphanumeric() || self.ch() == '_' || self.ch() == '-')
                {
                    self.pos += 1;
                }
                // Backtrack trailing hyphens/dots/equals that could be edge connectors
                while self.pos > start + 1 && matches!(self.src[self.pos - 1], '-' | '.' | '=') {
                    self.pos -= 1;
                }
                self.src[start..self.pos].iter().collect()
            } else {
                String::new()
            }
        }
    }

    fn parse_direction(c: &mut Cursor) -> parser::Direction {
        if c.consume_str("TD") || c.consume_str("TB") {
            parser::Direction::TD
        } else if c.consume_str("LR") {
            parser::Direction::LR
        } else if c.consume_str("RL") {
            parser::Direction::RL
        } else if c.consume_str("BT") {
            parser::Direction::BT
        } else {
            parser::Direction::TD
        }
    }

    fn parse_quoted_string(c: &mut Cursor) -> String {
        c.pos += 1; // skip opening "
        let mut buf = String::new();
        while !c.eof() && c.ch() != '"' {
            if c.ch() == '\\' && c.pos + 1 < c.src.len() {
                let nxt = c.src[c.pos + 1];
                match nxt {
                    'n' => buf.push('\n'),
                    '"' => buf.push('"'),
                    '\\' => buf.push('\\'),
                    other => buf.push(other),
                }
                c.pos += 2;
            } else {
                buf.push(c.ch());
                c.pos += 1;
            }
        }
        if !c.eof() {
            c.pos += 1;
        } // skip closing "
        buf
    }

    fn parse_node_label(c: &mut Cursor, closers: &[char]) -> String {
        c.skip_ws();
        if !c.eof() && c.ch() == '"' {
            return parse_quoted_string(c);
        }
        let start = c.pos;
        while !c.eof() && !closers.contains(&c.ch()) && c.ch() != '\n' {
            c.pos += 1;
        }
        c.src[start..c.pos]
            .iter()
            .collect::<String>()
            .trim()
            .to_string()
    }

    fn parse_node_shape(c: &mut Cursor) -> (bool, parser::NodeShape, String) {
        if c.consume_str("((") {
            let label = parse_node_label(c, &[')']);
            c.consume_str("))");
            (true, parser::NodeShape::Circle, label)
        } else if c.consume_str("(") {
            let label = parse_node_label(c, &[')']);
            c.consume_str(")");
            (true, parser::NodeShape::Rounded, label)
        } else if c.consume_str("{") {
            let label = parse_node_label(c, &['}']);
            c.consume_str("}");
            (true, parser::NodeShape::Diamond, label)
        } else if c.consume_str("[") {
            let label = parse_node_label(c, &[']']);
            c.consume_str("]");
            (true, parser::NodeShape::Rectangle, label)
        } else {
            (false, parser::NodeShape::Rectangle, String::new())
        }
    }

    fn parse_node_ref(c: &mut Cursor) -> parser::Node {
        c.skip_ws();
        let id = c.match_node_id();
        if id.is_empty() {
            return parser::node_bare(String::new());
        }
        let (found, shape, label) = parse_node_shape(c);
        if found {
            parser::node_new(id.clone(), label, shape)
        } else {
            parser::node_bare(id)
        }
    }

    struct EdgeMatch {
        token: &'static str,
        etype: parser::EdgeType,
    }

    const EDGE_PATTERNS: &[EdgeMatch] = &[
        EdgeMatch {
            token: "<-.->",
            etype: parser::EdgeType::BidirDotted,
        },
        EdgeMatch {
            token: "<==>",
            etype: parser::EdgeType::BidirThick,
        },
        EdgeMatch {
            token: "<-->",
            etype: parser::EdgeType::BidirArrow,
        },
        EdgeMatch {
            token: "-.->",
            etype: parser::EdgeType::DottedArrow,
        },
        EdgeMatch {
            token: "==>",
            etype: parser::EdgeType::ThickArrow,
        },
        EdgeMatch {
            token: "-->",
            etype: parser::EdgeType::Arrow,
        },
        EdgeMatch {
            token: "-.-",
            etype: parser::EdgeType::DottedLine,
        },
        EdgeMatch {
            token: "===",
            etype: parser::EdgeType::ThickLine,
        },
        EdgeMatch {
            token: "---",
            etype: parser::EdgeType::Line,
        },
    ];

    fn parse_edge_connector(c: &mut Cursor) -> parser::EdgeType {
        c.skip_ws();
        for em in EDGE_PATTERNS {
            if c.consume_str(em.token) {
                return em.etype.clone();
            }
        }
        parser::EdgeType::None
    }

    fn parse_edge_label(c: &mut Cursor) -> String {
        c.skip_ws();
        if !c.consume_str("|") {
            return String::new();
        }
        let start = c.pos;
        while !c.eof() && c.ch() != '|' && c.ch() != '\n' {
            c.pos += 1;
        }
        let text: String = c.src[start..c.pos].iter().collect();
        c.consume_str("|");
        text.trim().to_string()
    }

    fn at_end_keyword(c: &Cursor) -> bool {
        if !c.peek_str("end") {
            return false;
        }
        let after = c.pos + 3;
        if after >= c.src.len() {
            return true;
        }
        let ch = c.src[after];
        !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    }

    fn parse_statement_into(
        c: &mut Cursor,
        nodes: &mut Vec<parser::Node>,
        edges: &mut Vec<parser::Edge>,
        subgraphs: &mut Vec<parser::Subgraph>,
    ) -> bool {
        c.skip_ws();
        if c.eof() {
            return false;
        }

        // Try subgraph
        let saved = c.pos;
        let sg = parse_subgraph_block(c);
        if !sg.name.is_empty() {
            subgraphs.push(sg);
            return true;
        }
        c.pos = saved;

        // Try edge statement
        let saved = c.pos;
        let src_node = parse_node_ref(c);
        if !src_node.id.is_empty() {
            let mut chain_segs: Vec<(parser::EdgeType, String, parser::Node)> = Vec::new();
            loop {
                let seg_saved = c.pos;
                let etype = parse_edge_connector(c);
                if etype == parser::EdgeType::None {
                    c.pos = seg_saved;
                    break;
                }
                let lbl = parse_edge_label(c);
                let tgt = parse_node_ref(c);
                if tgt.id.is_empty() {
                    c.pos = seg_saved;
                    break;
                }
                chain_segs.push((etype, lbl, tgt));
            }

            if !chain_segs.is_empty() {
                upsert_node(nodes, src_node.clone());
                let mut prev_id = src_node.id.clone();
                for (etype, lbl, tgt) in chain_segs {
                    let mut e = parser::edge_new(prev_id.clone(), tgt.id.clone(), etype);
                    e.label = lbl;
                    upsert_node(nodes, tgt.clone());
                    edges.push(e);
                    prev_id = tgt.id;
                }
                c.skip_ws();
                c.consume_newline();
                return true;
            }

            // Not an edge — try as bare node
            upsert_node(nodes, src_node);
            c.skip_ws();
            c.consume_newline();
            return true;
        }
        c.pos = saved;
        false
    }

    fn upsert_node(nodes: &mut Vec<parser::Node>, node: parser::Node) {
        if !nodes.iter().any(|n| n.id == node.id) {
            nodes.push(node);
        }
    }

    fn parse_subgraph_block(c: &mut Cursor) -> parser::Subgraph {
        let saved = c.pos;
        c.skip_ws();
        if !c.consume_str("subgraph") {
            c.pos = saved;
            return parser::subgraph_new(String::new());
        }
        // "subgraph" must be followed by non-identifier char
        if !c.eof() && (c.ch().is_ascii_alphanumeric() || c.ch() == '_' || c.ch() == '-') {
            c.pos = saved;
            return parser::subgraph_new(String::new());
        }

        c.skip_ws();
        // Parse name/label
        let name = if !c.eof() && c.ch() == '"' {
            parse_quoted_string(c)
        } else {
            let start = c.pos;
            while !c.eof() && c.ch() != '\n' && c.ch() != '\r' {
                c.pos += 1;
            }
            c.src[start..c.pos]
                .iter()
                .collect::<String>()
                .trim()
                .to_string()
        };
        c.skip_ws();
        c.consume_newline();

        let mut sg = parser::subgraph_new(name);

        // Optional "direction XX"
        let dir_saved = c.pos;
        c.skip_ws();
        if c.consume_str("direction") {
            c.skip_ws();
            sg.direction = parse_direction(c);
            c.skip_ws();
            c.consume_newline();
        } else {
            c.pos = dir_saved;
        }

        // Parse body
        while !c.eof() {
            c.skip_ws();
            if at_end_keyword(c) {
                c.pos += 3;
                c.skip_ws();
                c.consume_newline();
                break;
            }
            let ok = parse_statement_into(c, &mut sg.nodes, &mut sg.edges, &mut sg.subgraphs);
            if !ok {
                if !c.consume_newline() {
                    c.pos += 1;
                }
            }
        }
        sg
    }

    fn parse_header(c: &mut Cursor) -> parser::Direction {
        let saved = c.pos;
        c.skip_ws_and_newlines();
        let ok = c.consume_str("flowchart") || c.consume_str("graph");
        if !ok {
            c.pos = saved;
            return parser::Direction::TD;
        }
        c.skip_ws();
        let d = parse_direction(c);
        c.skip_ws();
        // skip optional trailing comment
        if c.peek_str("%%") {
            while !c.eof() && c.ch() != '\n' {
                c.pos += 1;
            }
        }
        c.skip_ws();
        c.consume_newline();
        d
    }

    pub fn parse_flowchart(src: &str) -> parser::Graph {
        let mut c = Cursor::new(src);
        let mut g = parser::graph_new();
        g.direction = parse_header(&mut c);

        while !c.eof() {
            c.skip_ws();
            if !c.eof() {
                if !c.consume_newline() {
                    let ok =
                        parse_statement_into(&mut c, &mut g.nodes, &mut g.edges, &mut g.subgraphs);
                    if !ok {
                        c.pos += 1;
                    }
                }
            }
        }
        g
    }
}

// ── Bridge: parser AST → graph::Graph ───────────────────────────────────────

fn ast_to_graph(parsed: &parser::Graph) -> graph::Graph {
    let mut g = graph::graph_new();

    fn shape_str(s: &parser::NodeShape) -> &'static str {
        match s {
            parser::NodeShape::Rectangle => "Rectangle",
            parser::NodeShape::Rounded => "Rounded",
            parser::NodeShape::Diamond => "Diamond",
            parser::NodeShape::Circle => "Circle",
        }
    }
    fn etype_str(e: &parser::EdgeType) -> &'static str {
        match e {
            parser::EdgeType::Arrow => "Arrow",
            parser::EdgeType::Line => "Line",
            parser::EdgeType::DottedArrow => "DottedArrow",
            parser::EdgeType::DottedLine => "DottedLine",
            parser::EdgeType::ThickArrow => "ThickArrow",
            parser::EdgeType::ThickLine => "ThickLine",
            parser::EdgeType::BidirArrow => "BidirArrow",
            parser::EdgeType::BidirDotted => "BidirDotted",
            parser::EdgeType::BidirThick => "BidirThick",
            parser::EdgeType::None => "Arrow",
        }
    }

    for node in &parsed.nodes {
        graph::graph_add_node(&mut g, &node.id, &node.label, shape_str(&node.shape), None);
    }
    for edge in &parsed.edges {
        let label = if edge.label.is_empty() {
            None
        } else {
            Some(edge.label.as_str())
        };
        graph::graph_add_edge(
            &mut g,
            &edge.from_id,
            &edge.to_id,
            etype_str(&edge.edge_type),
            label,
        );
    }

    fn add_sg(g: &mut graph::Graph, sg: &parser::Subgraph) {
        fn sh(s: &parser::NodeShape) -> &'static str {
            match s {
                parser::NodeShape::Rectangle => "Rectangle",
                parser::NodeShape::Rounded => "Rounded",
                parser::NodeShape::Diamond => "Diamond",
                parser::NodeShape::Circle => "Circle",
            }
        }
        fn et(e: &parser::EdgeType) -> &'static str {
            match e {
                parser::EdgeType::Arrow => "Arrow",
                parser::EdgeType::Line => "Line",
                parser::EdgeType::DottedArrow => "DottedArrow",
                parser::EdgeType::DottedLine => "DottedLine",
                parser::EdgeType::ThickArrow => "ThickArrow",
                parser::EdgeType::ThickLine => "ThickLine",
                parser::EdgeType::BidirArrow => "BidirArrow",
                parser::EdgeType::BidirDotted => "BidirDotted",
                parser::EdgeType::BidirThick => "BidirThick",
                parser::EdgeType::None => "Arrow",
            }
        }
        for node in &sg.nodes {
            graph::graph_add_node(g, &node.id, &node.label, sh(&node.shape), Some(&sg.name));
        }
        for edge in &sg.edges {
            let label = if edge.label.is_empty() {
                None
            } else {
                Some(edge.label.as_str())
            };
            graph::graph_add_edge(g, &edge.from_id, &edge.to_id, et(&edge.edge_type), label);
        }
        for nested in &sg.subgraphs {
            add_sg(g, nested);
        }
    }

    for sg in &parsed.subgraphs {
        add_sg(&mut g, sg);
    }

    g
}

// ── Rust-native Sugiyama layout pipeline ────────────────────────────────────
// All layout functions implemented in Rust to bypass broken .hom codegen
// (nested while loops generate shadow variables instead of reassignment).

use std::collections::{HashMap, HashSet, VecDeque};

/// Phase 1: Remove cycles by reversing back edges (DFS-based).
fn remove_cycles_rust(g: &graph::Graph) -> (graph::Graph, Vec<(String, String)>) {
    if graph::graph_is_dag(g) {
        return (graph::graph_copy(g), vec![]);
    }

    let nodes = graph::graph_nodes(g);
    let mut visited: HashSet<String> = HashSet::new();
    let mut on_stack: HashSet<String> = HashSet::new();
    let mut back_edges: Vec<(String, String)> = Vec::new();

    fn dfs(
        node: &str,
        g: &graph::Graph,
        visited: &mut HashSet<String>,
        on_stack: &mut HashSet<String>,
        back_edges: &mut Vec<(String, String)>,
    ) {
        visited.insert(node.to_string());
        on_stack.insert(node.to_string());
        for succ in graph::graph_successors(g, node) {
            if on_stack.contains(&succ) {
                back_edges.push((node.to_string(), succ));
            } else if !visited.contains(&succ) {
                dfs(&succ, g, visited, on_stack, back_edges);
            }
        }
        on_stack.remove(node);
    }

    for node in &nodes {
        if !visited.contains(node) {
            dfs(node, g, &mut visited, &mut on_stack, &mut back_edges);
        }
    }

    // Build new graph with back edges reversed
    let mut dag = graph::graph_new();
    for node in &nodes {
        let idx = g.node_index[node];
        let nd = &g.digraph[idx];
        graph::graph_add_node(
            &mut dag,
            &nd.id,
            &nd.label,
            &nd.shape,
            nd.subgraph.as_deref(),
        );
    }

    let back_set: HashSet<(String, String)> = back_edges.iter().cloned().collect();
    for eidx in g.digraph.edge_indices() {
        let (a, b) = g.digraph.edge_endpoints(eidx).unwrap();
        let from_id = g.digraph[a].id.clone();
        let to_id = g.digraph[b].id.clone();
        let ed = &g.digraph[eidx];
        if back_set.contains(&(from_id.clone(), to_id.clone())) {
            // Reverse this edge
            graph::graph_add_edge(
                &mut dag,
                &to_id,
                &from_id,
                &ed.edge_type,
                ed.label.as_deref(),
            );
        } else {
            graph::graph_add_edge(
                &mut dag,
                &from_id,
                &to_id,
                &ed.edge_type,
                ed.label.as_deref(),
            );
        }
    }

    (dag, back_edges)
}

/// Phase 2: Assign layers using longest-path method (topological order).
fn assign_layers_rust(g: &graph::Graph) -> HashMap<String, i32> {
    let topo = graph::graph_topo_sort(g).unwrap_or_else(|| graph::graph_nodes(g));
    let mut layers: HashMap<String, i32> = HashMap::new();
    for node in &topo {
        layers.insert(node.clone(), 0);
    }
    for node in &topo {
        let curr = layers[node];
        for succ in graph::graph_successors(g, node) {
            let succ_layer = layers.get(&succ).copied().unwrap_or(0);
            if succ_layer <= curr {
                layers.insert(succ, curr + 1);
            }
        }
    }
    layers
}

/// Phase 3-4: Build layer ordering (group nodes by layer, sort within layer).
fn build_ordering(g: &graph::Graph, layers: &HashMap<String, i32>) -> Vec<Vec<String>> {
    let max_layer = layers.values().max().copied().unwrap_or(0);
    let mut layer_groups: Vec<Vec<String>> = vec![vec![]; (max_layer + 1) as usize];
    for (id, &layer) in layers {
        if layer >= 0 && (layer as usize) < layer_groups.len() {
            layer_groups[layer as usize].push(id.clone());
        }
    }

    // Deterministic initial order before barycenter passes
    for group in &mut layer_groups {
        group.sort();
    }

    // Barycenter crossing minimization: order by average position of neighbors
    for _pass in 0..4 {
        // Forward pass: order layer[i] by average position of predecessors in layer[i-1]
        for li in 1..layer_groups.len() {
            let prev_positions: HashMap<String, f64> = layer_groups[li - 1]
                .iter()
                .enumerate()
                .map(|(i, id)| (id.clone(), i as f64))
                .collect();
            let mut scored: Vec<(String, f64)> = layer_groups[li]
                .iter()
                .map(|id| {
                    let preds = graph::graph_predecessors(g, id);
                    let positions: Vec<f64> = preds
                        .iter()
                        .filter_map(|p| prev_positions.get(p).copied())
                        .collect();
                    let avg = if positions.is_empty() {
                        0.0
                    } else {
                        positions.iter().sum::<f64>() / positions.len() as f64
                    };
                    (id.clone(), avg)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            layer_groups[li] = scored.into_iter().map(|(id, _)| id).collect();
        }
        // Backward pass: order layer[i] by average position of successors in layer[i+1]
        for li in (0..layer_groups.len().saturating_sub(1)).rev() {
            let next_positions: HashMap<String, f64> = layer_groups[li + 1]
                .iter()
                .enumerate()
                .map(|(i, id)| (id.clone(), i as f64))
                .collect();
            let mut scored: Vec<(String, f64)> = layer_groups[li]
                .iter()
                .map(|id| {
                    let succs = graph::graph_successors(g, id);
                    let positions: Vec<f64> = succs
                        .iter()
                        .filter_map(|s| next_positions.get(s).copied())
                        .collect();
                    let avg = if positions.is_empty() {
                        0.0
                    } else {
                        positions.iter().sum::<f64>() / positions.len() as f64
                    };
                    (id.clone(), avg)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            layer_groups[li] = scored.into_iter().map(|(id, _)| id).collect();
        }
    }

    layer_groups
}

/// Ensure first segment exits vertically (down) and last segment enters vertically (down).
/// Mirrors the legacy `ensure_vertical_endpoints` from the reference Sugiyama implementation.
/// This guarantees correct arrowhead direction (▼ for TD, ► for LR after transpose).
fn ensure_vertical_endpoints(wps: &mut Vec<(i32, i32)>) {
    if wps.len() < 2 {
        return;
    }
    // Fix last segment: if horizontal, lift the second-to-last point up one row
    // so the final segment becomes a short downward step.
    let n = wps.len();
    let (lx, ly) = wps[n - 1];
    let (px, py) = wps[n - 2];
    if ly == py && lx != px && py > 0 {
        let new_y = py - 1;
        wps[n - 2] = (px, new_y);
        wps.insert(n - 1, (lx, new_y));
    }
    // Fix first segment: if horizontal, push the second point down one row
    // so the initial segment becomes a short downward exit.
    if wps.len() >= 2 && wps[0].1 == wps[1].1 && wps[0].0 != wps[1].0 {
        let new_y = wps[0].1 + 1;
        let sx = wps[1].0;
        let x0 = wps[0].0;
        wps[1] = (sx, new_y);
        wps.insert(1, (x0, new_y));
    }
}

/// Collected edge info for routing.
struct EdgeInfo {
    from_id: String,
    to_id: String,
    edge_type: String,
    label: String,
}

/// Phase 5: Assign coordinates to nodes.
fn assign_coordinates_rust(
    g: &graph::Graph,
    ordering: &[Vec<String>],
    padding: i32,
    is_lr_or_rl: bool,
) -> graph::NodeLayoutList {
    let nll = graph::nll_new();
    // For LR/RL, swap h_gap and v_gap so that after transposing the visual
    // gaps match the expected output (h_gap becomes row-spacing, v_gap becomes col-spacing).
    let h_gap = if is_lr_or_rl { 3i32 } else { 4i32 };
    let v_gap = if is_lr_or_rl { 4i32 } else { 3i32 };
    let min_node_h = 3i32;

    let mut y_offset = 0i32;
    for (layer_idx, layer_nodes) in ordering.iter().enumerate() {
        let mut layer_max_h = min_node_h;
        // First pass: compute dimensions
        let mut dims: Vec<(i32, i32)> = Vec::new();
        for node_id in layer_nodes {
            let idx = g.node_index[node_id];
            let nd = &g.digraph[idx];
            let label_w = nd.label.lines().map(|l| l.len()).max().unwrap_or(0) as i32;
            let label_h = std::cmp::max(nd.label.lines().count() as i32, 1);
            let w_vis = std::cmp::max(label_w + 2 + 2 * padding, 5);
            let h_vis = std::cmp::max(label_h + 2, min_node_h);
            // For LR/RL: swap width and height in TD layout space so that after
            // transposing the coordinates, nodes appear with the correct aspect ratio.
            let (w, h) = if is_lr_or_rl {
                (h_vis, w_vis)
            } else {
                (w_vis, h_vis)
            };
            if h > layer_max_h {
                layer_max_h = h;
            }
            dims.push((w, h));
        }
        // Second pass: place nodes
        let mut x_offset = 0i32;
        for (i, node_id) in layer_nodes.iter().enumerate() {
            let (w, h) = dims[i];
            let idx = g.node_index[node_id];
            let nd = &g.digraph[idx];
            graph::nll_push(
                nll.clone(),
                node_id.clone(),
                layer_idx as i32,
                i as i32,
                x_offset,
                y_offset,
                w,
                h,
                nd.label.clone(),
                nd.shape.clone(),
            );
            x_offset += w + h_gap;
        }
        y_offset += layer_max_h + v_gap;
    }

    // Center layers: find max total width, then offset each layer to center
    let nlen = graph::nll_len(nll.clone());
    let max_layer = ordering.len() as i32;
    let mut layer_widths: Vec<i32> = vec![0; max_layer as usize];
    for i in 0..nlen {
        let li = graph::nll_get_layer(nll.clone(), i) as usize;
        let right_edge = graph::nll_get_x(nll.clone(), i) + graph::nll_get_width(nll.clone(), i);
        if right_edge > layer_widths[li] {
            layer_widths[li] = right_edge;
        }
    }
    let max_width = layer_widths.iter().max().copied().unwrap_or(0);
    for i in 0..nlen {
        let li = graph::nll_get_layer(nll.clone(), i) as usize;
        let shift = (max_width - layer_widths[li]) / 2;
        let old_x = graph::nll_get_x(nll.clone(), i);
        graph::nll_set_x(nll.clone(), i, old_x + shift);
    }

    nll
}

/// Phase 6: Route edges using A* pathfinding with fallback.
fn route_edges_rust(
    g: &graph::Graph,
    nodes: &graph::NodeLayoutList,
    reversed: &[(String, String)],
) -> graph::EdgeRouteList {
    let routes = graph::erl_new();
    let nn = graph::nll_len(nodes.clone());

    // Build occupancy grid
    let mut max_x: i32 = 40;
    let mut max_y: i32 = 10;
    for i in 0..nn {
        let rx = graph::nll_get_x(nodes.clone(), i) + graph::nll_get_width(nodes.clone(), i) + 10;
        let ry = graph::nll_get_y(nodes.clone(), i) + graph::nll_get_height(nodes.clone(), i) + 10;
        if rx > max_x {
            max_x = rx;
        }
        if ry > max_y {
            max_y = ry;
        }
    }

    let grid = pathfinder::grid_new(max_x, max_y);
    for i in 0..nn {
        pathfinder::grid_mark_blocked(
            grid.clone(),
            graph::nll_get_x(nodes.clone(), i),
            graph::nll_get_y(nodes.clone(), i),
            graph::nll_get_width(nodes.clone(), i),
            graph::nll_get_height(nodes.clone(), i),
        );
    }

    // Collect all edges with metadata
    let reversed_set: HashSet<(String, String)> = reversed.iter().cloned().collect();
    for eidx in g.digraph.edge_indices() {
        let (a, b) = g.digraph.edge_endpoints(eidx).unwrap();
        let from_id = g.digraph[a].id.clone();
        let to_id = g.digraph[b].id.clone();
        let ed = &g.digraph[eidx];
        if from_id == to_id {
            continue;
        }

        let is_rev = reversed_set.contains(&(from_id.clone(), to_id.clone()));
        let (vis_from, vis_to) = if is_rev {
            (to_id.clone(), from_id.clone())
        } else {
            (from_id.clone(), to_id.clone())
        };

        let from_idx = graph::nll_id_to_index(nodes.clone(), vis_from.clone());
        let to_idx = graph::nll_id_to_index(nodes.clone(), vis_to.clone());
        if from_idx < 0 || to_idx < 0 {
            continue;
        }

        let exit_x = graph::nll_get_x(nodes.clone(), from_idx)
            + graph::nll_get_width(nodes.clone(), from_idx) / 2;
        let exit_y = graph::nll_get_y(nodes.clone(), from_idx)
            + graph::nll_get_height(nodes.clone(), from_idx);
        let entry_x = graph::nll_get_x(nodes.clone(), to_idx)
            + graph::nll_get_width(nodes.clone(), to_idx) / 2;
        let entry_y = graph::nll_get_y(nodes.clone(), to_idx) - 1;

        let path = pathfinder::a_star(grid.clone(), exit_x, exit_y, entry_x, entry_y);
        let plen = graph::point_list_len(path.clone());

        let waypoints = if plen > 0 {
            pathfinder::simplify_path(path)
        } else {
            // Fallback: orthogonal L-path
            let wp = graph::point_list_new();
            let mid_y = (exit_y + entry_y) / 2;
            graph::point_list_push(wp.clone(), exit_x, exit_y);
            graph::point_list_push(wp.clone(), exit_x, mid_y);
            graph::point_list_push(wp.clone(), entry_x, mid_y);
            graph::point_list_push(wp.clone(), entry_x, entry_y);
            wp
        };

        // Extract to Vec, fix vertical endpoints, rebuild PointList
        let n_wp = graph::point_list_len(waypoints.clone());
        let mut wp_vec: Vec<(i32, i32)> = (0..n_wp)
            .map(|i| {
                let x = graph::point_list_get_x(waypoints.clone(), i);
                let y = graph::point_list_get_y(waypoints.clone(), i);
                (x, y)
            })
            .collect();
        ensure_vertical_endpoints(&mut wp_vec);
        let fixed_wp = graph::point_list_new();
        for (x, y) in wp_vec {
            graph::point_list_push(fixed_wp.clone(), x, y);
        }

        let label = ed.label.clone().unwrap_or_default();
        graph::erl_push(
            routes.clone(),
            vis_from,
            vis_to,
            label,
            ed.edge_type.clone(),
            fixed_wp,
        );
    }

    routes
}

// ── Canvas direct-mutation helpers ──────────────────────────────────────────
// canvas.hom functions take Canvas by value (.clone()), so mutations are lost.
// These helpers mutate c.cells directly via &mut Canvas.

fn cset(c: &mut canvas::Canvas, col: i32, row: i32, ch: String) {
    if row >= 0 && row < c.height && col >= 0 && col < c.width {
        c.cells[row as usize][col as usize] = ch;
    }
}

fn cget(c: &canvas::Canvas, col: i32, row: i32) -> String {
    if row >= 0 && row < c.height && col >= 0 && col < c.width {
        c.cells[row as usize][col as usize].clone()
    } else {
        " ".to_string()
    }
}

fn cset_merge(c: &mut canvas::Canvas, col: i32, row: i32, ch: String) {
    if row >= 0 && col >= 0 && row < c.height && col < c.width {
        let existing = c.cells[row as usize][col as usize].clone();
        let ea = canvas::arms_from_char(existing);
        let na = canvas::arms_from_char(ch.clone());
        if ea.valid && na.valid {
            let merged = canvas::arms_merge(ea, na);
            c.cells[row as usize][col as usize] = canvas::arms_to_char(merged, c.charset.clone());
        } else {
            c.cells[row as usize][col as usize] = ch;
        }
    }
}

fn cwrite_str(c: &mut canvas::Canvas, col: i32, row: i32, s: &str) {
    for (i, ch) in s.chars().enumerate() {
        let cc = col + i as i32;
        if cc >= 0 && cc < c.width && row >= 0 && row < c.height {
            c.cells[row as usize][cc as usize] = ch.to_string();
        }
    }
}

fn cdraw_box(c: &mut canvas::Canvas, x: i32, y: i32, w: i32, h: i32, bc: &canvas::BoxChars) {
    if w < 2 || h < 2 {
        return;
    }
    let x1 = x + w - 1;
    let y1 = y + h - 1;
    cset(c, x, y, bc.top_left.clone());
    cset(c, x1, y, bc.top_right.clone());
    cset(c, x, y1, bc.bottom_left.clone());
    cset(c, x1, y1, bc.bottom_right.clone());
    for col in (x + 1)..x1 {
        cset(c, col, y, bc.horizontal.clone());
        cset(c, col, y1, bc.horizontal.clone());
    }
    for row in (y + 1)..y1 {
        cset(c, x, row, bc.vertical.clone());
        cset(c, x1, row, bc.vertical.clone());
    }
}

// ── Renderer helpers ────────────────────────────────────────────────────────

fn paint_node(c: &mut canvas::Canvas, x: i32, y: i32, w: i32, h: i32, label: &str, shape: &str) {
    let cs = c.charset.clone();
    let bc = match shape {
        "Rounded" => canvas::box_chars_rounded(cs),
        "Diamond" => canvas::box_chars_diamond(cs),
        "Circle" => canvas::box_chars_circle(cs),
        _ => canvas::box_chars_for_charset(cs),
    };
    cdraw_box(c, x, y, w, h, &bc);

    let inner_w = std::cmp::max(0, w - 2);
    let lines: Vec<&str> = label.split('\n').collect();
    for (i, line) in lines.iter().enumerate() {
        let label_row = y + 1 + i as i32;
        let line_len = line.len() as i32;
        let pad = std::cmp::max(0, inner_w - line_len) / 2;
        let col_start = x + 1 + pad;
        cwrite_str(c, col_start, label_row, line);
    }
}

fn paint_edge(c: &mut canvas::Canvas, waypoints: &[(i32, i32)], edge_type: &str, label: &str) {
    if waypoints.len() < 2 {
        return;
    }

    let cs = c.charset.clone();
    let bc = canvas::box_chars_for_charset(cs.clone());

    let (h_ch, v_ch) = match edge_type {
        "ThickArrow" | "ThickLine" | "BidirThick" => ("═".to_string(), "║".to_string()),
        "DottedArrow" | "DottedLine" | "BidirDotted" => ("╌".to_string(), "╎".to_string()),
        _ => (bc.horizontal.clone(), bc.vertical.clone()),
    };

    for i in 0..waypoints.len() - 1 {
        let (x0, y0) = waypoints[i];
        let (x1, y1) = waypoints[i + 1];
        if y0 == y1 {
            for col in (x0.min(x1) + 1)..x0.max(x1) {
                cset_merge(c, col, y0, h_ch.clone());
            }
        } else if x0 == x1 {
            for row in (y0.min(y1) + 1)..y0.max(y1) {
                cset_merge(c, x0, row, v_ch.clone());
            }
        }
    }

    for i in 0..waypoints.len() {
        let (px, py) = waypoints[i];
        let mut arms = canvas::Arms {
            valid: true,
            up: false,
            down: false,
            left: false,
            right: false,
        };
        if i > 0 {
            let (prev_x, prev_y) = waypoints[i - 1];
            if prev_x < px {
                arms.left = true;
            } else if prev_x > px {
                arms.right = true;
            } else if prev_y < py {
                arms.up = true;
            } else if prev_y > py {
                arms.down = true;
            }
        }
        if i < waypoints.len() - 1 {
            let (nxt_x, nxt_y) = waypoints[i + 1];
            if nxt_x > px {
                arms.right = true;
            } else if nxt_x < px {
                arms.left = true;
            } else if nxt_y > py {
                arms.down = true;
            } else if nxt_y < py {
                arms.up = true;
            }
        }
        cset_merge(c, px, py, canvas::arms_to_char(arms, cs.clone()));
    }

    // Arrowheads
    let arrow_types = [
        "Arrow",
        "DottedArrow",
        "ThickArrow",
        "BidirArrow",
        "BidirDotted",
        "BidirThick",
    ];
    let bidir_types = ["BidirArrow", "BidirDotted", "BidirThick"];

    if arrow_types.contains(&edge_type) {
        let (last_x, last_y) = waypoints[waypoints.len() - 1];
        let (prev_x, prev_y) = waypoints[waypoints.len() - 2];
        let arrow = if last_y < prev_y {
            bc.arrow_up.clone()
        } else if last_y > prev_y {
            bc.arrow_down.clone()
        } else if last_x > prev_x {
            bc.arrow_right.clone()
        } else {
            bc.arrow_left.clone()
        };
        cset(c, last_x, last_y, arrow);
    }

    if bidir_types.contains(&edge_type) && waypoints.len() >= 2 {
        let (first_x, first_y) = waypoints[0];
        let (second_x, second_y) = waypoints[1];
        let arrow = if first_y < second_y {
            bc.arrow_up.clone()
        } else if first_y > second_y {
            bc.arrow_down.clone()
        } else if first_x > second_x {
            bc.arrow_right.clone()
        } else {
            bc.arrow_left.clone()
        };
        cset(c, first_x, first_y, arrow);
    }

    if !label.is_empty() && waypoints.len() >= 2 {
        let mid = waypoints.len() / 2;
        let (lx, ly) = waypoints[mid];
        let label_y = std::cmp::max(0, ly - 1);
        cwrite_str(c, lx, label_y, label);
    }
}

fn paint_exit_stubs(
    c: &mut canvas::Canvas,
    edges: &graph::EdgeRouteList,
    nodes: &graph::NodeLayoutList,
) {
    let cs = c.charset.clone();
    let en = graph::erl_len(edges.clone());

    for ei in 0..en {
        let from_id = graph::erl_get_from(edges.clone(), ei);
        let wpc = graph::erl_get_waypoint_count(edges.clone(), ei);
        if wpc < 1 {
            continue;
        }

        let from_idx = graph::nll_id_to_index(nodes.clone(), from_id);
        if from_idx < 0 {
            continue;
        }

        let nx = graph::nll_get_x(nodes.clone(), from_idx);
        let ny = graph::nll_get_y(nodes.clone(), from_idx);
        let nw = graph::nll_get_width(nodes.clone(), from_idx);
        let nh = graph::nll_get_height(nodes.clone(), from_idx);
        let center_x = nx + nw / 2;
        let center_y = ny + nh / 2;

        let first_wp_x = graph::erl_get_waypoint_x(edges.clone(), ei, 0);
        let first_wp_y = graph::erl_get_waypoint_y(edges.clone(), ei, 0);

        let (stub_x, stub_y, arm_dir) = if first_wp_y >= ny + nh {
            (center_x, ny + nh - 1, "down")
        } else if first_wp_y < ny {
            (center_x, ny, "up")
        } else if first_wp_x >= nx + nw {
            (nx + nw - 1, center_y, "right")
        } else if first_wp_x < nx {
            (nx, center_y, "left")
        } else {
            (center_x, ny + nh - 1, "down")
        };

        let existing = cget(c, stub_x, stub_y);
        let ea = canvas::arms_from_char(existing);
        if ea.valid {
            let mut merged = ea.clone();
            match arm_dir {
                "down" => merged.down = true,
                "up" => merged.up = true,
                "right" => merged.right = true,
                "left" => merged.left = true,
                _ => {}
            }
            cset(c, stub_x, stub_y, canvas::arms_to_char(merged, cs.clone()));
        }
    }
}

fn transpose_layout(nodes: &graph::NodeLayoutList, edges: &graph::EdgeRouteList) {
    for n in nodes.borrow_mut().iter_mut() {
        std::mem::swap(&mut n.x, &mut n.y);
        std::mem::swap(&mut n.width, &mut n.height);
    }
    for e in edges.borrow_mut().iter_mut() {
        for wp in e.waypoints.iter_mut() {
            std::mem::swap(&mut wp.0, &mut wp.1);
        }
    }
}

fn flip_vertical(s: &str) -> String {
    let remap = |c: char| -> char {
        match c {
            '▼' => '▲',
            '▲' => '▼',
            'v' => '^',
            '^' => 'v',
            '┌' => '└',
            '└' => '┌',
            '┐' => '┘',
            '┘' => '┐',
            '╭' => '╰',
            '╰' => '╭',
            '╮' => '╯',
            '╯' => '╮',
            '┬' => '┴',
            '┴' => '┬',
            other => other,
        }
    };
    let lines: Vec<&str> = s.trim_end_matches('\n').split('\n').collect();
    let flipped: Vec<String> = lines
        .iter()
        .rev()
        .map(|line| line.chars().map(remap).collect::<String>())
        .collect();
    flipped.join("\n") + "\n"
}

fn flip_horizontal(s: &str) -> String {
    let remap = |c: char| -> char {
        match c {
            '►' => '◄',
            '◄' => '►',
            '>' => '<',
            '<' => '>',
            '┌' => '┐',
            '┐' => '┌',
            '└' => '┘',
            '┘' => '└',
            '╭' => '╮',
            '╮' => '╭',
            '╰' => '╯',
            '╯' => '╰',
            '├' => '┤',
            '┤' => '├',
            other => other,
        }
    };
    let lines: Vec<&str> = s.trim_end_matches('\n').split('\n').collect();
    let max_w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let flipped: Vec<String> = lines
        .iter()
        .map(|line| {
            let mut chars: Vec<char> = line.chars().collect();
            while chars.len() < max_w {
                chars.push(' ');
            }
            chars.reverse();
            let remapped: String = chars.into_iter().map(remap).collect();
            remapped.trim_end().to_string()
        })
        .collect();
    flipped.join("\n") + "\n"
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Parse a Mermaid flowchart string and render it to ASCII/Unicode art.
pub fn render_dsl(
    src: &str,
    unicode: bool,
    padding: usize,
    _direction: Option<&str>,
) -> Result<String, String> {
    // Phase 0: Parse
    let parsed = rust_parser::parse_flowchart(src);
    if parsed.nodes.is_empty() && parsed.edges.is_empty() && parsed.subgraphs.is_empty() {
        return Ok(String::new());
    }

    let direction = match parsed.direction {
        parser::Direction::LR => "LR",
        parser::Direction::RL => "RL",
        parser::Direction::BT => "BT",
        _ => "TD",
    };

    // Phase 1: Build graph
    let g = ast_to_graph(&parsed);

    // Phase 2: Remove cycles + assign layers
    let (dag, reversed) = remove_cycles_rust(&g);
    let layers = assign_layers_rust(&dag);

    // Phase 3-4: Build layer ordering with crossing minimization
    let ordering = build_ordering(&dag, &layers);

    // Phase 5: Assign coordinates
    let is_lr_or_rl = direction == "LR" || direction == "RL";
    let nodes = assign_coordinates_rust(&dag, &ordering, padding as i32, is_lr_or_rl);

    // Phase 6: Route edges
    let routed = route_edges_rust(&g, &nodes, &reversed);

    // Transpose node positions and waypoints for LR/RL so the TD layout maps
    // to a horizontal visual arrangement.
    if is_lr_or_rl {
        transpose_layout(&nodes, &routed);
    }

    // Phase 7: Render to canvas
    let cs = if unicode {
        canvas::CharSet::Unicode
    } else {
        canvas::CharSet::Ascii
    };

    // Canvas dimensions
    let nn = graph::nll_len(nodes.clone());
    let en = graph::erl_len(routed.clone());
    let mut max_col: i32 = 40;
    let mut max_row: i32 = 10;
    for i in 0..nn {
        let r = graph::nll_get_x(nodes.clone(), i) + graph::nll_get_width(nodes.clone(), i) + 2;
        let b = graph::nll_get_y(nodes.clone(), i) + graph::nll_get_height(nodes.clone(), i) + 4;
        if r > max_col {
            max_col = r;
        }
        if b > max_row {
            max_row = b;
        }
    }
    for i in 0..en {
        let wpc = graph::erl_get_waypoint_count(routed.clone(), i);
        for j in 0..wpc {
            let wx = graph::erl_get_waypoint_x(routed.clone(), i, j) + 4;
            let wy = graph::erl_get_waypoint_y(routed.clone(), i, j) + 4;
            if wx > max_col {
                max_col = wx;
            }
            if wy > max_row {
                max_row = wy;
            }
        }
    }

    let mut c = canvas::canvas_new(max_col, max_row, cs);

    // Paint nodes
    for i in 0..nn {
        paint_node(
            &mut c,
            graph::nll_get_x(nodes.clone(), i),
            graph::nll_get_y(nodes.clone(), i),
            graph::nll_get_width(nodes.clone(), i),
            graph::nll_get_height(nodes.clone(), i),
            &graph::nll_get_label(nodes.clone(), i),
            &graph::nll_get_shape(nodes.clone(), i),
        );
    }

    // Paint edges
    for i in 0..en {
        let wpc = graph::erl_get_waypoint_count(routed.clone(), i);
        let mut wps: Vec<(i32, i32)> = Vec::new();
        for j in 0..wpc {
            wps.push((
                graph::erl_get_waypoint_x(routed.clone(), i, j),
                graph::erl_get_waypoint_y(routed.clone(), i, j),
            ));
        }
        paint_edge(
            &mut c,
            &wps,
            &graph::erl_get_etype(routed.clone(), i),
            &graph::erl_get_label(routed.clone(), i),
        );
    }

    // Paint exit stubs
    paint_exit_stubs(&mut c, &routed, &nodes);

    // Render canvas to string (implemented directly to avoid .hom codegen issues)
    let mut rendered = {
        let mut lines: Vec<String> = Vec::new();
        for row in &c.cells {
            let line: String = row.join("");
            lines.push(line.trim_end().to_string());
        }
        // Trim trailing empty lines
        while lines.last().map(|l| l.is_empty()).unwrap_or(false) {
            lines.pop();
        }
        lines.join("\n") + "\n"
    };

    // Direction transforms
    if direction == "BT" {
        rendered = flip_vertical(&rendered);
    } else if direction == "RL" {
        rendered = flip_horizontal(&rendered);
    }

    Ok(rendered)
}

// ── WASM bindings ───────────────────────────────────────────────────────────

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn render(src: &str) -> Result<String, JsError> {
    render_dsl(src, true, 1, None).map_err(|e| JsError::new(&e))
}

#[cfg(feature = "wasm")]
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
    render_dsl(src, unicode, padding, dir).map_err(|e| JsError::new(&e))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "renderSvg")]
pub fn render_svg(src: &str, padding: usize, direction: &str) -> Result<String, JsError> {
    // Render as Unicode, wrap in SVG <text> for now (full SVG renderer TODO)
    let dir = if direction.is_empty() {
        None
    } else {
        Some(direction)
    };
    let ascii = render_dsl(src, true, padding, dir).map_err(|e| JsError::new(&e))?;
    let lines: Vec<&str> = ascii.trim_end().split('\n').collect();
    let max_width = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let height = lines.len();
    let char_w: f64 = 9.6;
    let char_h: f64 = 20.0;
    let svg_w = (max_width as f64 * char_w + 40.0) as i64;
    let svg_h = (height as f64 * char_h + 40.0) as i64;
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">"#
    );
    svg.push_str(r##"<rect width="100%" height="100%" fill="#0d1117"/>"##);
    svg.push_str(r##"<text font-family="monospace" font-size="14" fill="#c9d1d9">"##);
    for (i, line) in lines.iter().enumerate() {
        let escaped = line
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        svg.push_str(&format!(
            r#"<tspan x="20" y="{}">{}</tspan>"#,
            20.0 + (i as f64 + 1.0) * char_h,
            escaped
        ));
    }
    svg.push_str("</text></svg>");
    Ok(svg)
}
