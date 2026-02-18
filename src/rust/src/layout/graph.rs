//! GraphIR — intermediate representation wrapping petgraph DiGraph.
//!
//! Mirrors Python's layout/graph.py.

use petgraph::graph::DiGraph;

use crate::syntax::types::{Direction, Graph as AstGraph};

/// Node data stored in the petgraph DiGraph.
#[derive(Debug, Clone)]
pub struct NodeData {
    pub id: String,
    pub label: String,
    pub width: i64,
    pub height: i64,
}

/// Edge data stored in the petgraph DiGraph.
#[derive(Debug, Clone)]
pub struct EdgeData {
    pub label: Option<String>,
}

/// Graph intermediate representation.
///
/// Wraps petgraph DiGraph and adds Mermaid-specific metadata.
pub struct GraphIR {
    pub digraph: DiGraph<NodeData, EdgeData>,
    pub direction: Direction,
    /// Maps node id → petgraph NodeIndex.
    pub node_index: std::collections::HashMap<String, petgraph::graph::NodeIndex>,
    /// Maps subgraph name → list of member node ids.
    pub subgraph_members: std::collections::HashMap<String, Vec<String>>,
}

impl GraphIR {
    /// Build a GraphIR from the parsed AST.
    pub fn from_ast(_ast: &AstGraph) -> Self {
        // TODO: implement in Phase 3
        Self {
            digraph: DiGraph::new(),
            direction: Direction::TD,
            node_index: std::collections::HashMap::new(),
            subgraph_members: std::collections::HashMap::new(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.digraph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.digraph.edge_count()
    }

    pub fn is_dag(&self) -> bool {
        // TODO: implement in Phase 3
        true
    }

    pub fn topological_order(&self) -> Option<Vec<String>> {
        // TODO: implement in Phase 3
        None
    }

    pub fn in_degree(&self, _id: &str) -> usize {
        // TODO: implement in Phase 3
        0
    }

    pub fn out_degree(&self, _id: &str) -> usize {
        // TODO: implement in Phase 3
        0
    }

    pub fn adjacency_list(&self) -> Vec<(String, Vec<String>)> {
        // TODO: implement in Phase 3
        Vec::new()
    }
}
