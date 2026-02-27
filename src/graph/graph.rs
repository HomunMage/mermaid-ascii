// graph.rs — petgraph DiGraph wrapper with free-function API.
//
// Exposes a free-function style API that mirrors the .hom convention
// (no methods, only free functions operating on a plain struct).
//
// .hom modules import this via `use graph` and call:
//   graph_new(), graph_add_node(), graph_add_edge(), graph_topo_sort(), …
//
// Wraps petgraph::graph::DiGraph internally.

use std::collections::HashMap;

use petgraph::algo::{is_cyclic_directed, toposort};
use petgraph::graph::{DiGraph as PetGraph, NodeIndex};

// ── Data types ────────────────────────────────────────────────────────────────

/// Node metadata stored in the directed graph.
#[derive(Debug, Clone)]
pub struct NodeData {
    pub id: String,
    pub label: String,
    /// Shape name: "Rectangle", "Rounded", "Diamond", "Circle".
    pub shape: String,
    /// Subgraph this node belongs to, if any.
    pub subgraph: Option<String>,
}

/// Edge metadata stored in the directed graph.
#[derive(Debug, Clone)]
pub struct EdgeData {
    /// Edge type name: "Arrow", "Line", "DottedArrow", "DottedLine",
    /// "ThickArrow", "ThickLine", "BidirArrow", "BidirDotted", "BidirThick".
    pub edge_type: String,
    pub label: Option<String>,
}

/// Directed graph wrapper — the central data structure for layout phases.
///
/// Holds both the petgraph DiGraph (for topology algorithms) and a
/// `HashMap<String, NodeIndex>` for O(1) node lookup by id.
#[derive(Clone)]
pub struct Graph {
    pub digraph: PetGraph<NodeData, EdgeData>,
    /// Maps node id → petgraph NodeIndex.
    pub node_index: HashMap<String, NodeIndex>,
}

// ── Constructor ───────────────────────────────────────────────────────────────

/// Create a new, empty directed graph.
pub fn graph_new() -> Graph {
    Graph {
        digraph: PetGraph::new(),
        node_index: HashMap::new(),
    }
}

// ── Mutation ─────────────────────────────────────────────────────────────────

/// Add a node. No-op if a node with the same `id` already exists.
pub fn graph_add_node(
    g: &mut Graph,
    id: &str,
    label: &str,
    shape: &str,
    subgraph: Option<&str>,
) {
    if g.node_index.contains_key(id) {
        return;
    }
    let data = NodeData {
        id: id.to_string(),
        label: label.to_string(),
        shape: shape.to_string(),
        subgraph: subgraph.map(|s| s.to_string()),
    };
    let idx = g.digraph.add_node(data);
    g.node_index.insert(id.to_string(), idx);
}

/// Add a directed edge from `from_id` to `to_id`.
///
/// If either endpoint does not exist, a placeholder node is created with
/// `label = id` and `shape = "Rectangle"`.
pub fn graph_add_edge(
    g: &mut Graph,
    from_id: &str,
    to_id: &str,
    edge_type: &str,
    label: Option<&str>,
) {
    graph_ensure_node(g, from_id);
    graph_ensure_node(g, to_id);
    let from_idx = g.node_index[from_id];
    let to_idx = g.node_index[to_id];
    let data = EdgeData {
        edge_type: edge_type.to_string(),
        label: label.map(|l| l.to_string()),
    };
    g.digraph.add_edge(from_idx, to_idx, data);
}

/// Ensure a node exists. If absent, creates a Rectangle placeholder.
///
/// Exposed as `pub` for higher-level builder code (e.g., layout phases that
/// need to materialise implicit nodes referenced only in edges).
pub fn graph_ensure_node(g: &mut Graph, id: &str) {
    if !g.node_index.contains_key(id) {
        graph_add_node(g, id, id, "Rectangle", None);
    }
}

// ── Topology queries ─────────────────────────────────────────────────────────

/// Return the sorted list of successor (outgoing-neighbour) ids for `id`.
///
/// Returns an empty list if `id` is not present in the graph.
pub fn graph_successors(g: &Graph, id: &str) -> Vec<String> {
    match g.node_index.get(id) {
        None => vec![],
        Some(&idx) => {
            let mut result: Vec<String> = g
                .digraph
                .neighbors(idx)
                .map(|n| g.digraph[n].id.clone())
                .collect();
            result.sort();
            result
        }
    }
}

/// Return the sorted list of predecessor (incoming-neighbour) ids for `id`.
///
/// Returns an empty list if `id` is not present in the graph.
pub fn graph_predecessors(g: &Graph, id: &str) -> Vec<String> {
    match g.node_index.get(id) {
        None => vec![],
        Some(&idx) => {
            let mut result: Vec<String> = g
                .digraph
                .neighbors_directed(idx, petgraph::Direction::Incoming)
                .map(|n| g.digraph[n].id.clone())
                .collect();
            result.sort();
            result
        }
    }
}

/// Number of incoming edges for `id`. Returns 0 if the node is absent.
pub fn graph_in_degree(g: &Graph, id: &str) -> usize {
    match g.node_index.get(id) {
        None => 0,
        Some(&idx) => g
            .digraph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .count(),
    }
}

/// Number of outgoing edges for `id`. Returns 0 if the node is absent.
pub fn graph_out_degree(g: &Graph, id: &str) -> usize {
    match g.node_index.get(id) {
        None => 0,
        Some(&idx) => g
            .digraph
            .edges_directed(idx, petgraph::Direction::Outgoing)
            .count(),
    }
}

// ── Iteration ────────────────────────────────────────────────────────────────

/// Return all node ids, sorted alphabetically.
pub fn graph_nodes(g: &Graph) -> Vec<String> {
    let mut ids: Vec<String> = g
        .digraph
        .node_indices()
        .map(|idx| g.digraph[idx].id.clone())
        .collect();
    ids.sort();
    ids
}

/// Return all edges as `(from_id, to_id)` pairs, sorted lexicographically.
pub fn graph_edges(g: &Graph) -> Vec<(String, String)> {
    let mut edges: Vec<(String, String)> = g
        .digraph
        .edge_indices()
        .map(|eidx| {
            let (a, b) = g.digraph.edge_endpoints(eidx).unwrap();
            (g.digraph[a].id.clone(), g.digraph[b].id.clone())
        })
        .collect();
    edges.sort();
    edges
}

/// Total number of nodes.
pub fn graph_node_count(g: &Graph) -> usize {
    g.digraph.node_count()
}

/// Total number of edges.
pub fn graph_edge_count(g: &Graph) -> usize {
    g.digraph.edge_count()
}

// ── DAG algorithms ───────────────────────────────────────────────────────────

/// Returns `true` if the graph contains no directed cycles (i.e., is a DAG).
pub fn graph_is_dag(g: &Graph) -> bool {
    !is_cyclic_directed(&g.digraph)
}

/// Returns a topological ordering of node ids, or `None` if the graph has cycles.
///
/// The order respects all directed edges: if there is an edge A→B then A
/// appears before B in the returned list.
pub fn graph_topo_sort(g: &Graph) -> Option<Vec<String>> {
    match toposort(&g.digraph, None) {
        Ok(indices) => {
            let ids = indices
                .into_iter()
                .map(|idx| g.digraph[idx].id.clone())
                .collect();
            Some(ids)
        }
        Err(_) => None,
    }
}

// ── Utility ───────────────────────────────────────────────────────────────────

/// Return a deep copy of the graph (all nodes, edges, and the index map).
pub fn graph_copy(g: &Graph) -> Graph {
    g.clone()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let g = graph_new();
        assert_eq!(graph_node_count(&g), 0);
        assert_eq!(graph_edge_count(&g), 0);
        assert!(graph_is_dag(&g));
        assert_eq!(graph_topo_sort(&g), Some(vec![]));
    }

    #[test]
    fn test_add_node() {
        let mut g = graph_new();
        graph_add_node(&mut g, "A", "Node A", "Rectangle", None);
        assert_eq!(graph_node_count(&g), 1);
        assert_eq!(graph_nodes(&g), vec!["A"]);
    }

    #[test]
    fn test_add_node_dedup() {
        let mut g = graph_new();
        graph_add_node(&mut g, "A", "Node A", "Rectangle", None);
        graph_add_node(&mut g, "A", "duplicate", "Rounded", None);
        assert_eq!(graph_node_count(&g), 1);
        // Original data kept
        let idx = g.node_index["A"];
        assert_eq!(g.digraph[idx].label, "Node A");
    }

    #[test]
    fn test_add_edge_creates_placeholder_nodes() {
        let mut g = graph_new();
        graph_add_edge(&mut g, "A", "B", "Arrow", None);
        assert_eq!(graph_node_count(&g), 2);
        assert_eq!(graph_edge_count(&g), 1);
    }

    #[test]
    fn test_successors_and_predecessors() {
        let mut g = graph_new();
        graph_add_edge(&mut g, "A", "B", "Arrow", None);
        graph_add_edge(&mut g, "A", "C", "Arrow", None);

        assert_eq!(graph_successors(&g, "A"), vec!["B", "C"]);
        assert_eq!(graph_predecessors(&g, "B"), vec!["A"]);
        assert_eq!(graph_predecessors(&g, "A"), Vec::<String>::new());
    }

    #[test]
    fn test_successors_missing_node() {
        let g = graph_new();
        assert_eq!(graph_successors(&g, "X"), Vec::<String>::new());
        assert_eq!(graph_predecessors(&g, "X"), Vec::<String>::new());
    }

    #[test]
    fn test_in_and_out_degree() {
        let mut g = graph_new();
        graph_add_edge(&mut g, "A", "B", "Arrow", None);
        graph_add_edge(&mut g, "C", "B", "Arrow", None);

        assert_eq!(graph_out_degree(&g, "A"), 1);
        assert_eq!(graph_in_degree(&g, "B"), 2);
        assert_eq!(graph_in_degree(&g, "A"), 0);
        assert_eq!(graph_in_degree(&g, "missing"), 0);
        assert_eq!(graph_out_degree(&g, "missing"), 0);
    }

    #[test]
    fn test_edges_list() {
        let mut g = graph_new();
        graph_add_edge(&mut g, "A", "B", "Arrow", None);
        graph_add_edge(&mut g, "B", "C", "Line", Some("label"));
        let edges = graph_edges(&g);
        assert_eq!(
            edges,
            vec![
                ("A".to_string(), "B".to_string()),
                ("B".to_string(), "C".to_string()),
            ]
        );
    }

    #[test]
    fn test_nodes_sorted() {
        let mut g = graph_new();
        // Insert in reverse order; graph_nodes should sort alphabetically.
        graph_add_node(&mut g, "C", "C", "Rectangle", None);
        graph_add_node(&mut g, "A", "A", "Rectangle", None);
        graph_add_node(&mut g, "B", "B", "Rectangle", None);
        assert_eq!(graph_nodes(&g), vec!["A", "B", "C"]);
    }

    #[test]
    fn test_is_dag_true() {
        let mut g = graph_new();
        graph_add_edge(&mut g, "A", "B", "Arrow", None);
        graph_add_edge(&mut g, "B", "C", "Arrow", None);
        assert!(graph_is_dag(&g));
    }

    #[test]
    fn test_is_dag_false_with_cycle() {
        let mut g = graph_new();
        graph_add_edge(&mut g, "A", "B", "Arrow", None);
        graph_add_edge(&mut g, "B", "C", "Arrow", None);
        graph_add_edge(&mut g, "C", "A", "Arrow", None);
        assert!(!graph_is_dag(&g));
    }

    #[test]
    fn test_topo_sort_chain() {
        let mut g = graph_new();
        graph_add_node(&mut g, "A", "A", "Rectangle", None);
        graph_add_node(&mut g, "B", "B", "Rectangle", None);
        graph_add_node(&mut g, "C", "C", "Rectangle", None);
        graph_add_edge(&mut g, "A", "B", "Arrow", None);
        graph_add_edge(&mut g, "B", "C", "Arrow", None);
        let order = graph_topo_sort(&g).unwrap();
        let pos = |id: &str| order.iter().position(|x| x == id).unwrap();
        assert!(pos("A") < pos("B"));
        assert!(pos("B") < pos("C"));
    }

    #[test]
    fn test_topo_sort_returns_none_for_cycle() {
        let mut g = graph_new();
        graph_add_edge(&mut g, "A", "B", "Arrow", None);
        graph_add_edge(&mut g, "B", "A", "Arrow", None);
        assert!(graph_topo_sort(&g).is_none());
    }

    #[test]
    fn test_graph_copy_is_independent() {
        let mut g = graph_new();
        graph_add_node(&mut g, "X", "Node X", "Diamond", None);
        let mut g2 = graph_copy(&g);

        // Add to original; copy unaffected.
        graph_add_node(&mut g, "Y", "Node Y", "Rectangle", None);
        assert_eq!(graph_node_count(&g2), 1);
        assert_eq!(graph_nodes(&g2), vec!["X"]);

        // Add to copy; original unaffected.
        graph_add_node(&mut g2, "Z", "Node Z", "Rounded", None);
        assert_eq!(graph_node_count(&g), 2);
    }

    #[test]
    fn test_node_data_fields() {
        let mut g = graph_new();
        graph_add_node(&mut g, "A", "Label A", "Rounded", Some("mysubgraph"));
        let idx = g.node_index["A"];
        let data = &g.digraph[idx];
        assert_eq!(data.id, "A");
        assert_eq!(data.label, "Label A");
        assert_eq!(data.shape, "Rounded");
        assert_eq!(data.subgraph, Some("mysubgraph".to_string()));
    }

    #[test]
    fn test_edge_data_fields() {
        let mut g = graph_new();
        graph_add_edge(&mut g, "A", "B", "DottedArrow", Some("my label"));
        let eidx = g.digraph.edge_indices().next().unwrap();
        let data = &g.digraph[eidx];
        assert_eq!(data.edge_type, "DottedArrow");
        assert_eq!(data.label, Some("my label".to_string()));
    }

    #[test]
    fn test_ensure_node_creates_placeholder() {
        let mut g = graph_new();
        graph_ensure_node(&mut g, "implicit");
        assert_eq!(graph_node_count(&g), 1);
        let idx = g.node_index["implicit"];
        let data = &g.digraph[idx];
        assert_eq!(data.id, "implicit");
        assert_eq!(data.label, "implicit");
        assert_eq!(data.shape, "Rectangle");
        assert!(data.subgraph.is_none());
    }
}
