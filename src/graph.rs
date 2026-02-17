/// Graph IR — converts AST into a petgraph DiGraph for layout and analysis.
///
/// This module owns the canonical graph data structure used by all downstream
/// phases (layout, routing, rendering). It flattens subgraphs into the main
/// node/edge lists while preserving subgraph membership for later rendering.

use petgraph::algo::{is_cyclic_directed, toposort};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

use crate::ast;

// ─── Node data stored in the petgraph ────────────────────────────────────────

/// Data associated with each node in the DiGraph.
#[derive(Debug, Clone)]
pub struct NodeData {
    pub id: String,
    pub label: String,
    pub shape: ast::NodeShape,
    pub attrs: Vec<ast::Attr>,
    /// Which subgraph this node belongs to (None = top-level).
    pub subgraph: Option<String>,
}

// ─── Edge data stored in the petgraph ────────────────────────────────────────

/// Data associated with each edge in the DiGraph.
#[derive(Debug, Clone)]
pub struct EdgeData {
    pub edge_type: ast::EdgeType,
    pub label: Option<String>,
    pub attrs: Vec<ast::Attr>,
}

// ─── GraphIR ─────────────────────────────────────────────────────────────────

/// The graph intermediate representation built from an AST Graph.
///
/// Wraps a petgraph `DiGraph` and exposes helpers for topology queries.
pub struct GraphIR {
    pub digraph: DiGraph<NodeData, EdgeData>,
    /// Maps node id → NodeIndex for fast lookup during edge insertion.
    pub node_index: HashMap<String, NodeIndex>,
    /// Direction from the AST (used by layout).
    pub direction: ast::Direction,
    /// Subgraph membership: subgraph name → list of node ids.
    pub subgraph_members: Vec<(String, Vec<String>)>,
}

impl GraphIR {
    /// Build a GraphIR from an AST Graph.
    pub fn from_ast(ast_graph: &ast::Graph) -> Self {
        let mut digraph: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let mut node_index: HashMap<String, NodeIndex> = HashMap::new();
        let mut subgraph_members: Vec<(String, Vec<String>)> = Vec::new();

        // Helper closure: add a node if not already present.
        // Returns the NodeIndex for the given node id.
        let add_node = |digraph: &mut DiGraph<NodeData, EdgeData>,
                             node_index: &mut HashMap<String, NodeIndex>,
                             ast_node: &ast::Node,
                             subgraph_name: Option<&str>| {
            if !node_index.contains_key(&ast_node.id) {
                let data = NodeData {
                    id: ast_node.id.clone(),
                    label: ast_node.label.clone(),
                    shape: ast_node.shape.clone(),
                    attrs: ast_node.attrs.clone(),
                    subgraph: subgraph_name.map(|s| s.to_string()),
                };
                let idx = digraph.add_node(data);
                node_index.insert(ast_node.id.clone(), idx);
            }
        };

        // Process top-level nodes.
        for node in &ast_graph.nodes {
            add_node(&mut digraph, &mut node_index, node, None);
        }

        // Process subgraphs (flat — collect members for rendering).
        for sg in &ast_graph.subgraphs {
            collect_subgraph(sg, &mut digraph, &mut node_index, &mut subgraph_members);
        }

        // Add top-level edges. Implicitly add any nodes referenced that aren't
        // already present (can happen if edge references undeclared node ids).
        for edge in &ast_graph.edges {
            ensure_node(&mut digraph, &mut node_index, &edge.from);
            ensure_node(&mut digraph, &mut node_index, &edge.to);
            add_edge(&mut digraph, &node_index, edge);
        }

        // Add subgraph edges (all levels, recursively).
        for sg in &ast_graph.subgraphs {
            collect_subgraph_edges(sg, &mut digraph, &mut node_index);
        }

        GraphIR {
            digraph,
            node_index,
            direction: ast_graph.direction.clone(),
            subgraph_members,
        }
    }

    /// Returns true if the graph has no directed cycles (is a DAG).
    pub fn is_dag(&self) -> bool {
        !is_cyclic_directed(&self.digraph)
    }

    /// Topological order of node ids, if the graph is a DAG.
    /// Returns None if the graph has cycles.
    pub fn topological_order(&self) -> Option<Vec<String>> {
        match toposort(&self.digraph, None) {
            Ok(indices) => Some(
                indices
                    .iter()
                    .map(|&idx| self.digraph[idx].id.clone())
                    .collect(),
            ),
            Err(_) => None,
        }
    }

    /// Returns the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.digraph.node_count()
    }

    /// Returns the total number of edges.
    pub fn edge_count(&self) -> usize {
        self.digraph.edge_count()
    }

    /// Returns the in-degree of a node by id.
    pub fn in_degree(&self, node_id: &str) -> usize {
        self.node_index.get(node_id).map_or(0, |&idx| {
            self.digraph
                .neighbors_directed(idx, petgraph::Direction::Incoming)
                .count()
        })
    }

    /// Returns the out-degree of a node by id.
    pub fn out_degree(&self, node_id: &str) -> usize {
        self.node_index.get(node_id).map_or(0, |&idx| {
            self.digraph
                .neighbors_directed(idx, petgraph::Direction::Outgoing)
                .count()
        })
    }

    /// Returns an adjacency list: each node id mapped to its outgoing neighbor ids.
    pub fn adjacency_list(&self) -> Vec<(String, Vec<String>)> {
        let mut list: Vec<(String, Vec<String>)> = self
            .node_index
            .iter()
            .map(|(id, &idx)| {
                let neighbors: Vec<String> = self
                    .digraph
                    .neighbors_directed(idx, petgraph::Direction::Outgoing)
                    .map(|n| self.digraph[n].id.clone())
                    .collect();
                (id.clone(), neighbors)
            })
            .collect();
        // Sort for deterministic output.
        list.sort_by(|a, b| a.0.cmp(&b.0));
        list
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Recursively collect nodes from a subgraph (and nested subgraphs) into the DiGraph.
fn collect_subgraph(
    sg: &ast::Subgraph,
    digraph: &mut DiGraph<NodeData, EdgeData>,
    node_index: &mut HashMap<String, NodeIndex>,
    subgraph_members: &mut Vec<(String, Vec<String>)>,
) {
    let mut member_ids: Vec<String> = Vec::new();

    for node in &sg.nodes {
        if !node_index.contains_key(&node.id) {
            let data = NodeData {
                id: node.id.clone(),
                label: node.label.clone(),
                shape: node.shape.clone(),
                attrs: node.attrs.clone(),
                subgraph: Some(sg.name.clone()),
            };
            let idx = digraph.add_node(data);
            node_index.insert(node.id.clone(), idx);
        }
        member_ids.push(node.id.clone());
    }

    subgraph_members.push((sg.name.clone(), member_ids));

    for nested in &sg.subgraphs {
        collect_subgraph(nested, digraph, node_index, subgraph_members);
    }
}

/// Recursively add edges from a subgraph into the DiGraph.
fn collect_subgraph_edges(
    sg: &ast::Subgraph,
    digraph: &mut DiGraph<NodeData, EdgeData>,
    node_index: &mut HashMap<String, NodeIndex>,
) {
    for edge in &sg.edges {
        ensure_node(digraph, node_index, &edge.from);
        ensure_node(digraph, node_index, &edge.to);
        add_edge(digraph, node_index, edge);
    }
    for nested in &sg.subgraphs {
        collect_subgraph_edges(nested, digraph, node_index);
    }
}

/// Ensure a node with the given id exists in the DiGraph.
/// Creates a minimal placeholder node if not already present.
fn ensure_node(
    digraph: &mut DiGraph<NodeData, EdgeData>,
    node_index: &mut HashMap<String, NodeIndex>,
    node_id: &str,
) {
    if !node_index.contains_key(node_id) {
        let data = NodeData {
            id: node_id.to_string(),
            label: node_id.to_string(),
            shape: ast::NodeShape::Rectangle,
            attrs: Vec::new(),
            subgraph: None,
        };
        let idx = digraph.add_node(data);
        node_index.insert(node_id.to_string(), idx);
    }
}

/// Add an AST edge to the DiGraph. Both endpoint nodes must already exist.
fn add_edge(
    digraph: &mut DiGraph<NodeData, EdgeData>,
    node_index: &HashMap<String, NodeIndex>,
    edge: &ast::Edge,
) {
    let from_idx = node_index[&edge.from];
    let to_idx = node_index[&edge.to];
    let data = EdgeData {
        edge_type: edge.edge_type.clone(),
        label: edge.label.clone(),
        attrs: edge.attrs.clone(),
    };
    digraph.add_edge(from_idx, to_idx, data);
}
