/// Layout module — Phase 3 of the Sugiyama pipeline.
///
/// This module implements the 4 phases of Sugiyama-style layout:
///   1. Cycle removal  (this file — greedy-FAS approach)
///   2. Layer assignment (rank each node)
///   3. Crossing minimization (barycenter heuristic)
///   4. Coordinate assignment (x/y positions)
///
/// The public entry point is `Layout::from_graph_ir()`.

use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

use crate::ast;
use crate::graph::{EdgeData, GraphIR, NodeData};

// ─── Cycle Removal (Greedy-FAS) ───────────────────────────────────────────────

/// Result of cycle removal: a set of edge indices that were reversed to make
/// the graph a DAG. These are "back-edges" in the original graph.
///
/// The caller can use this set to flip arrow directions in the rendering phase
/// (so the displayed arrow still points the "right" way visually).
pub struct CycleRemovalResult {
    /// Edges that were reversed. Their direction in the layout graph is flipped.
    pub reversed_edges: HashSet<EdgeIndex>,
}

/// Remove cycles from a copy of the DiGraph using the greedy-FAS heuristic.
///
/// The greedy-FAS (Feedback Arc Set) algorithm works as follows:
///   - Maintain a sequence `s1` (sinks) and `s2` (sources) being built left/right.
///   - Repeatedly:
///     1. Move all sinks to the front of s2.
///     2. Move all sources to the back of s1.
///     3. Of the remaining nodes, pick the one with max (out-degree - in-degree)
///        and add it to s1.
///   - The final ordering is: s1 + s2 (reversed).
///   - Any edge going against this ordering is a back-edge and is reversed.
///
/// Returns the modified graph and the set of reversed edge indices (relative to
/// the input graph's edge indices).
///
/// Reference: Eades, Lin, Smyth (1993) — "A fast and effective heuristic for
/// the feedback arc set problem", Information Processing Letters.
pub fn remove_cycles(
    graph: &DiGraph<NodeData, EdgeData>,
) -> (DiGraph<NodeData, EdgeData>, HashSet<EdgeIndex>) {
    let node_count = graph.node_count();
    if node_count == 0 {
        return (graph.clone(), HashSet::new());
    }

    // Build node ordering via greedy-FAS.
    let ordering = greedy_fas_ordering(graph);

    // Build position map: NodeIndex → position in the ordering.
    let mut position: HashMap<NodeIndex, usize> = HashMap::new();
    for (pos, &node) in ordering.iter().enumerate() {
        position.insert(node, pos);
    }

    // Identify back-edges: edges where source comes AFTER target in the ordering,
    // or self-loops (source == target). These must be reversed to break cycles.
    let mut reversed_edges: HashSet<EdgeIndex> = HashSet::new();
    for edge in graph.edge_references() {
        let is_self_loop = edge.source() == edge.target();
        let src_pos = position[&edge.source()];
        let tgt_pos = position[&edge.target()];
        if is_self_loop || src_pos > tgt_pos {
            reversed_edges.insert(edge.id());
        }
    }

    // Build the modified graph with back-edges reversed.
    let mut new_graph: DiGraph<NodeData, EdgeData> = DiGraph::new();

    // Add all nodes preserving NodeIndex mapping (petgraph assigns 0..N in order).
    let mut sorted_nodes: Vec<NodeIndex> = graph.node_indices().collect();
    sorted_nodes.sort();
    for &ni in &sorted_nodes {
        new_graph.add_node(graph[ni].clone());
    }

    // Add edges, reversing back-edges. Skip self-loops (they can't be forward
    // edges and reversing them still gives a self-loop — just remove them).
    for edge in graph.edge_references() {
        if edge.source() == edge.target() {
            // Self-loop is in reversed_edges; omit from the DAG entirely.
            continue;
        }
        let (src, tgt) = if reversed_edges.contains(&edge.id()) {
            (edge.target(), edge.source())
        } else {
            (edge.source(), edge.target())
        };
        new_graph.add_edge(src, tgt, edge.weight().clone());
    }

    (new_graph, reversed_edges)
}

/// Compute a node ordering using the greedy-FAS heuristic.
///
/// Returns a Vec<NodeIndex> ordering that minimizes back-edges.
/// Nodes earlier in the ordering should have outgoing edges going forward.
fn greedy_fas_ordering(graph: &DiGraph<NodeData, EdgeData>) -> Vec<NodeIndex> {
    let mut active: HashSet<NodeIndex> = graph.node_indices().collect();

    // Dynamic degree counters updated as nodes are removed.
    let mut out_deg: HashMap<NodeIndex, i64> = HashMap::new();
    let mut in_deg: HashMap<NodeIndex, i64> = HashMap::new();

    for ni in graph.node_indices() {
        out_deg.insert(
            ni,
            graph
                .neighbors_directed(ni, petgraph::Direction::Outgoing)
                .count() as i64,
        );
        in_deg.insert(
            ni,
            graph
                .neighbors_directed(ni, petgraph::Direction::Incoming)
                .count() as i64,
        );
    }

    // s1: nodes placed at the "left" (sources, high out-degree surplus)
    // s2: nodes placed at the "right" (sinks)
    let mut s1: Vec<NodeIndex> = Vec::new();
    let mut s2: Vec<NodeIndex> = Vec::new();

    while !active.is_empty() {
        // Step 1: Pull all sinks (out_deg == 0) into s2.
        loop {
            let sinks: Vec<NodeIndex> = active
                .iter()
                .copied()
                .filter(|&n| out_deg[&n] == 0)
                .collect();
            if sinks.is_empty() {
                break;
            }
            for sink in sinks {
                active.remove(&sink);
                s2.push(sink);
                for pred in graph.neighbors_directed(sink, petgraph::Direction::Incoming) {
                    if active.contains(&pred) {
                        *out_deg.get_mut(&pred).unwrap() -= 1;
                    }
                }
            }
        }

        // Step 2: Pull all sources (in_deg == 0) into s1.
        loop {
            let sources: Vec<NodeIndex> = active
                .iter()
                .copied()
                .filter(|&n| in_deg[&n] == 0)
                .collect();
            if sources.is_empty() {
                break;
            }
            for source in sources {
                active.remove(&source);
                s1.push(source);
                for succ in graph.neighbors_directed(source, petgraph::Direction::Outgoing) {
                    if active.contains(&succ) {
                        *in_deg.get_mut(&succ).unwrap() -= 1;
                    }
                }
            }
        }

        // Step 3: If nodes remain in cycles, pick max (out - in) node.
        if let Some(&best) = active.iter().max_by_key(|&&n| out_deg[&n] - in_deg[&n]) {
            active.remove(&best);
            s1.push(best);
            for succ in graph.neighbors_directed(best, petgraph::Direction::Outgoing) {
                if active.contains(&succ) {
                    *in_deg.get_mut(&succ).unwrap() -= 1;
                }
            }
            for pred in graph.neighbors_directed(best, petgraph::Direction::Incoming) {
                if active.contains(&pred) {
                    *out_deg.get_mut(&pred).unwrap() -= 1;
                }
            }
        }
    }

    // Final ordering: s1 + reversed(s2)
    s2.reverse();
    s1.extend(s2);
    s1
}

// ─── Layout IR ────────────────────────────────────────────────────────────────

/// A positioned node in the layout.
#[derive(Debug, Clone)]
pub struct LayoutNode {
    /// The original node id (from GraphIR).
    pub id: String,
    /// Layer (rank) — 0 is top/left depending on direction.
    pub layer: usize,
    /// Position within the layer (0-indexed).
    pub order: usize,
    /// Final x coordinate (character column).
    pub x: usize,
    /// Final y coordinate (character row).
    pub y: usize,
    /// Width in characters (includes borders).
    pub width: usize,
    /// Height in characters (includes borders).
    pub height: usize,
}

/// The result of the full layout algorithm for a graph.
pub struct Layout {
    /// Positioned nodes.
    pub nodes: Vec<LayoutNode>,
    /// Edge indices that were reversed during cycle removal.
    pub reversed_edges: HashSet<EdgeIndex>,
}

impl Layout {
    /// Run cycle removal on the GraphIR and return a partial Layout.
    /// Coordinates are not yet assigned (that is done in later phases).
    pub fn from_graph_ir(gir: &GraphIR) -> Self {
        let (_, reversed_edges) = remove_cycles(&gir.digraph);

        // Placeholder nodes without coordinates — future phases fill these in.
        let nodes: Vec<LayoutNode> = gir
            .digraph
            .node_indices()
            .map(|ni| {
                let data = &gir.digraph[ni];
                let label_len = data.label.len();
                LayoutNode {
                    id: data.id.clone(),
                    layer: 0,
                    order: 0,
                    x: 0,
                    y: 0,
                    width: label_len + 4, // "[ " + label + " ]"
                    height: 3,            // top border + text + bottom border
                }
            })
            .collect();

        Layout {
            nodes,
            reversed_edges,
        }
    }
}

// ─── Dummy Node Insertion ──────────────────────────────────────────────────────

/// The result of dummy node insertion for a single long edge.
///
/// When an edge spans more than one layer (i.e., `layer[tgt] - layer[src] > 1`),
/// it is replaced by a chain of dummy nodes — one per intermediate layer.
/// Each dummy node gets a synthetic id and is marked with `is_dummy = true`
/// in the augmented graph.
pub struct DummyEdge {
    /// Id of the original source node.
    pub original_src: String,
    /// Id of the original target node.
    pub original_tgt: String,
    /// Ids of the dummy nodes inserted along the path, in order from src to tgt.
    pub dummy_ids: Vec<String>,
    /// Original edge data (edge type, label, attrs) preserved for rendering.
    pub edge_data: EdgeData,
}

/// A graph augmented with dummy nodes for edges that span multiple layers.
///
/// After dummy node insertion, every edge in the augmented graph connects
/// nodes in adjacent layers (layer difference == 1). This is a pre-condition
/// for the crossing-minimisation and coordinate-assignment phases.
pub struct AugmentedGraph {
    /// The augmented DiGraph. Contains original nodes plus dummy nodes.
    /// Dummy nodes have ids starting with `"__dummy_"`.
    pub graph: DiGraph<NodeData, EdgeData>,
    /// Layer assignment for every node (original + dummy).
    pub layers: HashMap<String, usize>,
    /// Total number of layers.
    pub layer_count: usize,
    /// Information about each long edge that was broken up.
    pub dummy_edges: Vec<DummyEdge>,
}

/// Prefix used for dummy node ids. The rendering phase can detect dummy nodes
/// by checking `id.starts_with(DUMMY_PREFIX)`.
pub const DUMMY_PREFIX: &str = "__dummy_";

/// Insert dummy nodes into the cycle-free, layer-assigned graph.
///
/// For each edge (u → v) where `layer[v] - layer[u] > 1`, the edge is removed
/// and replaced by the chain:
///   u → d₁ → d₂ → … → dₖ → v
/// where each dᵢ lives in layer `layer[u] + i`.
///
/// # Arguments
/// * `dag`   — The cycle-free DiGraph produced by `remove_cycles`.
/// * `la`    — The layer assignment produced by `LayerAssignment::assign`.
///
/// # Returns
/// An `AugmentedGraph` where every edge connects adjacent-layer nodes.
pub fn insert_dummy_nodes(dag: &DiGraph<NodeData, EdgeData>, la: &LayerAssignment) -> AugmentedGraph {
    use crate::ast::NodeShape;

    // Clone the DAG — we will mutate it by removing long edges and adding dummies.
    let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();

    // Rebuild the graph from scratch so we control NodeIndex ordering.
    // Map original NodeIndex → new NodeIndex.
    let mut old_to_new: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    // Add all original nodes.
    let mut sorted_nodes: Vec<NodeIndex> = dag.node_indices().collect();
    sorted_nodes.sort();
    for &ni in &sorted_nodes {
        let new_ni = g.add_node(dag[ni].clone());
        old_to_new.insert(ni, new_ni);
    }

    // Layer map: node_id → layer (starts with original nodes, extended with dummies).
    let mut layers: HashMap<String, usize> = la.layers.clone();

    let mut dummy_edges: Vec<DummyEdge> = Vec::new();
    // Counter to generate unique dummy ids per long edge.
    let mut edge_counter: usize = 0;

    // Collect all edges upfront so we can iterate without borrowing `g` mutably.
    let all_edges: Vec<(NodeIndex, NodeIndex, EdgeData)> = dag
        .edge_references()
        .map(|e| (e.source(), e.target(), e.weight().clone()))
        .collect();

    for (src_old, tgt_old, edge_data) in all_edges {
        let src_new = old_to_new[&src_old];
        let tgt_new = old_to_new[&tgt_old];

        let src_id = g[src_new].id.clone();
        let tgt_id = g[tgt_new].id.clone();

        let src_layer = layers[&src_id];
        let tgt_layer = layers[&tgt_id];

        // Edges within the same layer or adjacent layers need no dummies.
        // (Same-layer edges are unusual but can arise from bidirectional edges
        //  after cycle removal; we keep them as-is.)
        let layer_diff = if tgt_layer > src_layer {
            tgt_layer - src_layer
        } else {
            // Edge goes "upward" (reversed back-edge in display); treat as span 1.
            1
        };

        if layer_diff <= 1 {
            // Adjacent-layer edge — copy as-is.
            g.add_edge(src_new, tgt_new, edge_data);
            continue;
        }

        // Long edge: replace with a chain of dummy nodes.
        // Each dummy id is "__dummy_{edge_counter}_{i}" for uniqueness.
        let steps = layer_diff - 1; // number of intermediate layers
        let this_edge = edge_counter;
        edge_counter += 1;

        let mut dummy_ids: Vec<String> = Vec::with_capacity(steps);
        let mut chain_prev = src_new;

        for i in 0..steps {
            let dummy_layer = src_layer + i + 1;
            let dummy_id = format!("{}{}_{}", DUMMY_PREFIX, this_edge, i);

            let dummy_data = NodeData {
                id: dummy_id.clone(),
                label: String::new(),
                shape: NodeShape::Rectangle,
                attrs: Vec::new(),
                subgraph: None,
            };
            let dummy_ni = g.add_node(dummy_data);
            layers.insert(dummy_id.clone(), dummy_layer);
            dummy_ids.push(dummy_id);

            // Edge from previous node to this dummy.
            let segment_edge = EdgeData {
                edge_type: edge_data.edge_type.clone(),
                label: None, // label only on the last segment (see below)
                attrs: Vec::new(),
            };
            g.add_edge(chain_prev, dummy_ni, segment_edge);
            chain_prev = dummy_ni;
        }

        // Final segment: dummy → original target, carry the label.
        let last_segment = EdgeData {
            edge_type: edge_data.edge_type.clone(),
            label: edge_data.label.clone(),
            attrs: edge_data.attrs.clone(),
        };
        g.add_edge(chain_prev, tgt_new, last_segment);

        dummy_edges.push(DummyEdge {
            original_src: src_id,
            original_tgt: tgt_id,
            dummy_ids,
            edge_data,
        });
    }

    let layer_count = layers.values().copied().max().unwrap_or(0) + 1;

    AugmentedGraph {
        graph: g,
        layers,
        layer_count,
        dummy_edges,
    }
}

// ─── Layer Assignment ─────────────────────────────────────────────────────────

/// Result of layer assignment: each node is assigned a layer (rank).
///
/// Layer 0 is the "first" layer (top for TD, left for LR).
/// This is computed on a cycle-free copy of the graph produced by `remove_cycles`.
pub struct LayerAssignment {
    /// Maps node id → layer index.
    pub layers: HashMap<String, usize>,
    /// Total number of layers.
    pub layer_count: usize,
    /// Edges that were reversed during cycle removal (for display).
    pub reversed_edges: HashSet<EdgeIndex>,
}

impl LayerAssignment {
    /// Assign layers to all nodes using fixed-point iteration.
    ///
    /// Algorithm: for each edge u→v, rank[v] = max(rank[v], rank[u] + 1).
    /// Repeat until stable. Runs in O(V * E) worst case, fast in practice.
    pub fn assign(gir: &GraphIR) -> Self {
        let (dag, reversed_edges) = remove_cycles(&gir.digraph);

        // Initialize all layers to 0.
        let mut layers: HashMap<String, usize> = gir
            .digraph
            .node_indices()
            .map(|ni| (gir.digraph[ni].id.clone(), 0usize))
            .collect();

        // Fixed-point iteration: propagate ranks along DAG edges.
        let mut changed = true;
        while changed {
            changed = false;
            for edge in dag.edge_references() {
                let src_id = &dag[edge.source()].id;
                let tgt_id = &dag[edge.target()].id;
                let src_rank = layers[src_id];
                let tgt_rank = layers[tgt_id];
                if tgt_rank < src_rank + 1 {
                    *layers.get_mut(tgt_id).unwrap() = src_rank + 1;
                    changed = true;
                }
            }
        }

        let layer_count = layers.values().copied().max().unwrap_or(0) + 1;

        LayerAssignment {
            layers,
            layer_count,
            reversed_edges,
        }
    }

    /// Format a human-readable report for the out/phase3/ files.
    pub fn format_report(&self, gir: &GraphIR) -> String {
        let mut out = String::new();

        out.push_str(&format!(
            "Reversed edges (cycle removal): {}\n",
            self.reversed_edges.len()
        ));
        out.push_str(&format!("Layer count: {}\n\n", self.layer_count));

        // Group nodes by layer.
        let mut by_layer: Vec<Vec<&str>> = vec![vec![]; self.layer_count];
        let mut node_ids: Vec<&str> = self.layers.keys().map(|s| s.as_str()).collect();
        node_ids.sort();
        for id in &node_ids {
            let layer = self.layers[*id];
            by_layer[layer].push(id);
        }

        for (i, nodes) in by_layer.iter().enumerate() {
            let mut sorted = nodes.clone();
            sorted.sort();
            out.push_str(&format!("  Layer {}: {}\n", i, sorted.join(", ")));
        }

        // Show which edges were reversed (by node id pairs).
        if !self.reversed_edges.is_empty() {
            out.push_str("\nReversed back-edges:\n");
            for &eidx in &self.reversed_edges {
                if let Some(edge) = gir.digraph.edge_endpoints(eidx) {
                    let src = &gir.digraph[edge.0].id;
                    let tgt = &gir.digraph[edge.1].id;
                    out.push_str(&format!("  {} ← {} (displayed reversed)\n", src, tgt));
                }
            }
        }

        out
    }
}

// ─── Crossing Minimization (Barycenter) ───────────────────────────────────────

/// Minimise edge crossings using the barycenter heuristic.
///
/// Takes the augmented graph (with dummy nodes) and returns an ordering for
/// each layer that reduces edge crossings. Multiple top-down + bottom-up passes
/// are run until the crossing count stops improving (or a pass limit is hit).
///
/// Returns a `Vec<Vec<String>>` — one inner vec per layer, in minimised order.
pub fn minimise_crossings(aug: &AugmentedGraph) -> Vec<Vec<String>> {
    let layer_count = aug.layer_count;

    // Initial ordering: group by layer, sort alphabetically for determinism.
    let mut ordering: Vec<Vec<String>> = vec![Vec::new(); layer_count];
    let mut ids: Vec<String> = aug.layers.keys().cloned().collect();
    ids.sort();
    for id in ids {
        let layer = aug.layers[&id];
        ordering[layer].push(id);
    }

    // Build id → NodeIndex for the augmented graph.
    let id_to_idx: HashMap<String, NodeIndex> = aug
        .graph
        .node_indices()
        .map(|ni| (aug.graph[ni].id.clone(), ni))
        .collect();

    let max_passes = 24;
    let mut best = count_crossings(&ordering, &id_to_idx, &aug.graph);

    for _pass in 0..max_passes {
        // Top-down sweep: use predecessor positions as barycenter weights.
        for layer_idx in 1..layer_count {
            // Clone the previous layer's ids so we can borrow ordering[layer_idx] mutably.
            let prev_ids: Vec<String> = ordering[layer_idx - 1].clone();
            let prev: HashMap<&str, f64> = prev_ids
                .iter()
                .enumerate()
                .map(|(i, id)| (id.as_str(), i as f64))
                .collect();
            ordering[layer_idx].sort_by(|a, b| {
                let wa = barycenter(a, &aug.graph, &id_to_idx, &prev, petgraph::Direction::Incoming);
                let wb = barycenter(b, &aug.graph, &id_to_idx, &prev, petgraph::Direction::Incoming);
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        // Bottom-up sweep: use successor positions as barycenter weights.
        for layer_idx in (0..layer_count.saturating_sub(1)).rev() {
            // Clone the next layer's ids to avoid simultaneous borrow conflict.
            let next_ids: Vec<String> = ordering[layer_idx + 1].clone();
            let next: HashMap<&str, f64> = next_ids
                .iter()
                .enumerate()
                .map(|(i, id)| (id.as_str(), i as f64))
                .collect();
            ordering[layer_idx].sort_by(|a, b| {
                let wa = barycenter(a, &aug.graph, &id_to_idx, &next, petgraph::Direction::Outgoing);
                let wb = barycenter(b, &aug.graph, &id_to_idx, &next, petgraph::Direction::Outgoing);
                wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let new = count_crossings(&ordering, &id_to_idx, &aug.graph);
        if new >= best {
            break;
        }
        best = new;
    }

    ordering
}

/// Average position of a node's neighbours in the adjacent layer (barycenter weight).
fn barycenter(
    node_id: &str,
    graph: &DiGraph<NodeData, EdgeData>,
    id_to_idx: &HashMap<String, NodeIndex>,
    neighbor_pos: &HashMap<&str, f64>,
    direction: petgraph::Direction,
) -> f64 {
    let Some(ni) = id_to_idx.get(node_id) else {
        return f64::MAX;
    };
    let positions: Vec<f64> = graph
        .neighbors_directed(*ni, direction)
        .filter_map(|nb| neighbor_pos.get(graph[nb].id.as_str()).copied())
        .collect();
    if positions.is_empty() {
        return f64::MAX;
    }
    positions.iter().sum::<f64>() / positions.len() as f64
}

/// Count edge crossings between consecutive layers (inversion count heuristic).
fn count_crossings(
    ordering: &[Vec<String>],
    id_to_idx: &HashMap<String, NodeIndex>,
    graph: &DiGraph<NodeData, EdgeData>,
) -> usize {
    let mut total = 0;
    for l in 0..ordering.len().saturating_sub(1) {
        let tgt_pos: HashMap<&str, usize> = ordering[l + 1]
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (sp, sid) in ordering[l].iter().enumerate() {
            if let Some(ni) = id_to_idx.get(sid) {
                for nb in graph.neighbors_directed(*ni, petgraph::Direction::Outgoing) {
                    if let Some(&tp) = tgt_pos.get(graph[nb].id.as_str()) {
                        edges.push((sp, tp));
                    }
                }
            }
        }
        for i in 0..edges.len() {
            for j in i + 1..edges.len() {
                if (edges[i].0 < edges[j].0 && edges[i].1 > edges[j].1)
                    || (edges[i].0 > edges[j].0 && edges[i].1 < edges[j].1)
                {
                    total += 1;
                }
            }
        }
    }
    total
}

// ─── Coordinate Assignment ────────────────────────────────────────────────────

/// Character-unit geometry constants (TD layout).
const NODE_PADDING: usize = 1; // spaces inside brackets on each side of label
const H_GAP: usize = 4;        // horizontal gap (chars) between nodes in same layer
const V_GAP: usize = 3;        // vertical gap (rows) between adjacent layers
const NODE_HEIGHT: usize = 3;  // top-border + text-row + bottom-border

/// Assign (x, y) character coordinates to every node in the augmented graph.
///
/// Layout is top-down (TD): x = column, y = row. The renderer transposes for LR.
/// Dummy nodes are given width 1 to minimise horizontal space consumption.
pub fn assign_coordinates(ordering: &[Vec<String>], aug: &AugmentedGraph) -> Vec<LayoutNode> {
    assign_coordinates_padded(ordering, aug, NODE_PADDING, &HashMap::new())
}

// ─── Edge Routing (Orthogonal) ────────────────────────────────────────────────

/// A 2D point in character coordinates (column, row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

/// A routed edge: the from/to node ids, an optional label, and the ordered
/// list of waypoints that form the orthogonal (H/V segments only) path.
///
/// The path goes:  exit point (on source node border)
///                  → [intermediate bend points through inter-layer gaps]
///                  → entry point (on target node border).
#[derive(Debug, Clone)]
pub struct RoutedEdge {
    pub from_id: String,
    pub to_id: String,
    /// Label carried from the original edge, if any.
    pub label: Option<String>,
    /// Edge type from the original edge (Arrow, BackArrow, etc.).
    pub edge_type: crate::ast::EdgeType,
    pub waypoints: Vec<Point>,
}

/// Route all edges in `gir` orthogonally through inter-layer gap spaces.
///
/// For each original edge u → v (from `gir`, not the augmented graph):
///   1. Exit point = bottom-centre of u's node box.
///   2. Entry point = top-centre of v's node box.
///   3. Route through the midpoint of each inter-layer gap between their layers,
///      using dummy node x-positions to align through intermediate layers.
///
/// Back-edges (reversed during cycle removal) are displayed with from/to
/// swapped so the arrowhead points in the intended direction.
///
/// Self-loops are skipped (they were removed during cycle removal).
pub fn route_edges(
    gir: &GraphIR,
    layout_nodes: &[LayoutNode],
    aug: &AugmentedGraph,
    reversed_edges: &HashSet<EdgeIndex>,
) -> Vec<RoutedEdge> {
    // Fast id → LayoutNode lookup.
    let node_map: HashMap<&str, &LayoutNode> =
        layout_nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    // Per-layer geometry: top-Y and bottom-Y of the tallest node in each layer.
    let layer_count = layout_nodes.iter().map(|n| n.layer).max().map(|m| m + 1).unwrap_or(0);
    let mut layer_top_y = vec![usize::MAX; layer_count.max(1)];
    let mut layer_bottom_y = vec![0usize; layer_count.max(1)];
    for n in layout_nodes {
        if n.y < layer_top_y[n.layer] {
            layer_top_y[n.layer] = n.y;
        }
        let bot = n.y + n.height;
        if bot > layer_bottom_y[n.layer] {
            layer_bottom_y[n.layer] = bot;
        }
    }

    // Build lookup: (original_src_id, original_tgt_id) → dummy x-positions per gap.
    // The dummy x is the horizontal centre of the dummy node for that gap.
    let mut dummy_xs_map: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for de in &aug.dummy_edges {
        let xs: Vec<usize> = de
            .dummy_ids
            .iter()
            .filter_map(|did| node_map.get(did.as_str()).map(|n| n.x + n.width / 2))
            .collect();
        dummy_xs_map.insert((de.original_src.clone(), de.original_tgt.clone()), xs);
    }

    let mut routes: Vec<RoutedEdge> = Vec::new();

    for edge in gir.digraph.edge_references() {
        // Self-loops were removed during cycle removal — skip.
        if edge.source() == edge.target() {
            continue;
        }

        let is_reversed = reversed_edges.contains(&edge.id());
        let edge_data = edge.weight();

        // Visual from/to: flip if this is a reversed back-edge.
        let (vis_from, vis_to) = if is_reversed {
            (
                gir.digraph[edge.target()].id.as_str(),
                gir.digraph[edge.source()].id.as_str(),
            )
        } else {
            (
                gir.digraph[edge.source()].id.as_str(),
                gir.digraph[edge.target()].id.as_str(),
            )
        };

        let Some(from_node) = node_map.get(vis_from) else { continue };
        let Some(to_node) = node_map.get(vis_to) else { continue };

        // Retrieve dummy x-positions for skip-level routing (empty for adjacent edges).
        let empty: Vec<usize> = Vec::new();
        let dummy_xs = dummy_xs_map
            .get(&(vis_from.to_string(), vis_to.to_string()))
            .unwrap_or(&empty);

        let waypoints = compute_orthogonal_waypoints(
            from_node,
            to_node,
            &layer_top_y,
            &layer_bottom_y,
            dummy_xs,
        );

        routes.push(RoutedEdge {
            from_id: vis_from.to_string(),
            to_id: vis_to.to_string(),
            label: edge_data.label.clone(),
            edge_type: edge_data.edge_type.clone(),
            waypoints,
        });
    }

    routes
}

/// Compute the orthogonal waypoints for a single edge from `from` to `to`.
///
/// Strategy (TD layout):
///   - Exit = bottom-centre of `from`.
///   - Entry = top-centre of `to`.
///   - For each inter-layer gap between from.layer and to.layer, drop a bend
///     point at the midpoint row of that gap. Use `dummy_xs[i]` for the x
///     position at gap i (from dummy node insertion), or fall back to exit_x /
///     entry_x for the first and last gaps respectively.
///   - Same-layer edges get a U-shape looping below the layer.
fn compute_orthogonal_waypoints(
    from: &LayoutNode,
    to: &LayoutNode,
    layer_top_y: &[usize],
    layer_bottom_y: &[usize],
    dummy_xs: &[usize],
) -> Vec<Point> {
    let exit_x = from.x + from.width / 2;
    let exit_y = from.y + from.height - 1; // bottom border row
    let entry_x = to.x + to.width / 2;
    let entry_y = to.y;                    // top border row

    let src_layer = from.layer;
    let tgt_layer = to.layer;

    // Same-layer: U-shape going below the layer.
    if src_layer == tgt_layer {
        let below_y = layer_bottom_y.get(src_layer).copied().unwrap_or(exit_y + 1) + V_GAP / 2;
        return vec![
            Point { x: exit_x, y: exit_y },
            Point { x: exit_x, y: below_y },
            Point { x: entry_x, y: below_y },
            Point { x: entry_x, y: entry_y },
        ];
    }

    let (low_layer, high_layer) = if src_layer < tgt_layer {
        (src_layer, tgt_layer)
    } else {
        (tgt_layer, src_layer)
    };

    let mut waypoints: Vec<Point> = vec![Point { x: exit_x, y: exit_y }];

    let gaps = high_layer - low_layer;
    for gap_idx in 0..gaps {
        let gap = low_layer + gap_idx;

        // Midpoint row of the inter-layer gap.
        let gap_start = layer_bottom_y.get(gap).copied().unwrap_or(exit_y + 1);
        let gap_end = layer_top_y.get(gap + 1).copied().unwrap_or(gap_start + V_GAP);
        let mid_y = gap_start + (gap_end.saturating_sub(gap_start)) / 2;

        // X at this gap: use dummy node centre if available, else interpolate.
        let gap_x = dummy_xs.get(gap_idx).copied().unwrap_or(if gap_idx == 0 { exit_x } else { entry_x });

        let last_wp = waypoints.last().unwrap();

        // Horizontal move (if needed), then vertical move to mid_y.
        if last_wp.x != gap_x {
            let last_y = last_wp.y;
            waypoints.push(Point { x: gap_x, y: last_y });
        }
        waypoints.push(Point { x: gap_x, y: mid_y });
    }

    // Final horizontal move to entry_x, then down to entry_y.
    let last_wp = waypoints.last().unwrap();
    if last_wp.x != entry_x {
        let last_y = last_wp.y;
        waypoints.push(Point { x: entry_x, y: last_y });
    }
    waypoints.push(Point { x: entry_x, y: entry_y });

    waypoints
}

// ─── Compound Node (Subgraph Collapse/Expand) ────────────────────────────────

/// Prefix for compound node ids (subgraphs collapsed into single nodes).
pub const COMPOUND_PREFIX: &str = "__sg_";

/// Gap between member nodes inside a subgraph.
const SG_INNER_GAP: usize = 1;
/// Padding between subgraph border and member nodes (left/right).
const SG_PAD_X: usize = 1;

/// Information about a collapsed subgraph (compound node).
pub struct CompoundInfo {
    pub sg_name: String,
    pub compound_id: String,
    pub member_ids: Vec<String>,
    pub member_widths: Vec<usize>,
    pub member_heights: Vec<usize>,
    pub max_member_height: usize,
    /// Optional description text shown below members in the subgraph box.
    pub description: Option<String>,
}

/// Collapse subgraphs into compound nodes for layout.
///
/// For each subgraph with members, replaces the member nodes with a single
/// compound node sized to contain all members horizontally. Cross-boundary
/// edges are redirected to the compound node. Internal edges are dropped.
fn collapse_subgraphs(gir: &GraphIR, padding: usize) -> (GraphIR, Vec<CompoundInfo>) {
    // Map member_id → subgraph name
    let mut member_to_sg: HashMap<&str, &str> = HashMap::new();
    let mut compounds: Vec<CompoundInfo> = Vec::new();

    for (sg_name, members) in &gir.subgraph_members {
        let compound_id = format!("{}{}", COMPOUND_PREFIX, sg_name);

        let mut member_widths = Vec::new();
        let mut member_heights = Vec::new();

        for mid in members {
            if let Some(&ni) = gir.node_index.get(mid) {
                let data = &gir.digraph[ni];
                let (max_line_w, line_count) = label_dimensions(&data.label);
                member_widths.push(max_line_w + 2 + 2 * padding);
                member_heights.push(2 + line_count);
            } else {
                member_widths.push(3 + 2 * padding);
                member_heights.push(NODE_HEIGHT);
            }
            member_to_sg.insert(mid.as_str(), sg_name.as_str());
        }

        let max_member_height = member_heights.iter().copied().max().unwrap_or(NODE_HEIGHT);

        let description = gir.subgraph_descriptions.get(sg_name).cloned();

        compounds.push(CompoundInfo {
            sg_name: sg_name.clone(),
            compound_id,
            member_ids: members.clone(),
            member_widths,
            member_heights,
            max_member_height,
            description,
        });
    }

    // Build sg_name → compound_id lookup
    let sg_to_compound: HashMap<&str, &str> = compounds
        .iter()
        .map(|c| (c.sg_name.as_str(), c.compound_id.as_str()))
        .collect();

    // Build the collapsed graph
    let mut new_digraph: DiGraph<NodeData, EdgeData> = DiGraph::new();
    let mut new_node_index: HashMap<String, NodeIndex> = HashMap::new();

    // Add non-member, non-subgraph-ref nodes (preserve original order)
    let mut sorted_nodes: Vec<NodeIndex> = gir.digraph.node_indices().collect();
    sorted_nodes.sort();
    for &ni in &sorted_nodes {
        let data = &gir.digraph[ni];
        // Skip member nodes (they're inside compound nodes)
        if member_to_sg.contains_key(data.id.as_str()) {
            continue;
        }
        // Skip placeholder nodes whose id matches a subgraph name
        // (created by ensure_node when a subgraph is used as edge endpoint)
        if sg_to_compound.contains_key(data.id.as_str()) {
            continue;
        }
        let idx = new_digraph.add_node(data.clone());
        new_node_index.insert(data.id.clone(), idx);
    }

    // Add compound nodes
    for ci in &compounds {
        let compound_data = NodeData {
            id: ci.compound_id.clone(),
            label: ci.sg_name.clone(),
            shape: ast::NodeShape::Rectangle,
            attrs: Vec::new(),
            subgraph: None,
        };
        let idx = new_digraph.add_node(compound_data);
        new_node_index.insert(ci.compound_id.clone(), idx);
    }

    // Resolve an edge endpoint: member → compound, sg_name → compound, else keep.
    let resolve_endpoint = |id: &str| -> String {
        if let Some(sg) = member_to_sg.get(id) {
            return sg_to_compound[sg].to_string();
        }
        if let Some(cid) = sg_to_compound.get(id) {
            return cid.to_string();
        }
        id.to_string()
    };

    // Add edges, redirecting member/subgraph endpoints to compound nodes
    let mut added_edges: HashSet<(String, String)> = HashSet::new();
    for edge in gir.digraph.edge_references() {
        let src_id = &gir.digraph[edge.source()].id;
        let tgt_id = &gir.digraph[edge.target()].id;

        let actual_src = resolve_endpoint(src_id);
        let actual_tgt = resolve_endpoint(tgt_id);

        // Both resolved to same compound node → internal edge, skip
        if actual_src == actual_tgt {
            continue;
        }

        // Avoid duplicate edges between same pair
        let key = (actual_src.clone(), actual_tgt.clone());
        if added_edges.contains(&key) {
            continue;
        }
        added_edges.insert(key);

        let from_idx = new_node_index[&actual_src];
        let to_idx = new_node_index[&actual_tgt];
        new_digraph.add_edge(from_idx, to_idx, edge.weight().clone());
    }

    let collapsed = GraphIR {
        digraph: new_digraph,
        node_index: new_node_index,
        direction: gir.direction.clone(),
        subgraph_members: Vec::new(),
        subgraph_descriptions: HashMap::new(),
    };

    (collapsed, compounds)
}

/// Compute dimension overrides for compound nodes: id → (width, height).
fn compute_compound_dimensions(compounds: &[CompoundInfo], padding: usize) -> HashMap<String, (usize, usize)> {
    let mut overrides = HashMap::new();
    for ci in compounds {
        let total_member_w: usize = ci.member_widths.iter().sum();
        let gaps = if ci.member_ids.len() > 1 {
            (ci.member_ids.len() - 1) * SG_INNER_GAP
        } else {
            0
        };
        let content_w = total_member_w + gaps;
        let title_w = ci.sg_name.len() + 4;
        let desc_w = ci.description.as_ref().map(|d| d.len() + 4).unwrap_or(0);
        let inner_w = content_w.max(title_w).max(desc_w);
        let width = 2 + 2 * SG_PAD_X + inner_w;

        // Description adds one row per line of text.
        let desc_rows = ci.description.as_ref().map(|_| 1).unwrap_or(0);

        // height = top border + title row + member height + desc rows + bottom border
        let height = if ci.member_ids.is_empty() {
            3 + desc_rows // border + title + desc + border
        } else {
            2 + 1 + ci.max_member_height + desc_rows // borders + title + members + desc
        };

        let _ = padding; // padding already factored into member widths
        overrides.insert(ci.compound_id.clone(), (width, height));
    }
    overrides
}

/// Expand compound nodes by adding member nodes positioned inside them.
fn expand_compound_nodes(
    layout_nodes: &[LayoutNode],
    compounds: &[CompoundInfo],
) -> Vec<LayoutNode> {
    let compound_map: HashMap<&str, &CompoundInfo> = compounds
        .iter()
        .map(|c| (c.compound_id.as_str(), c))
        .collect();

    let mut result: Vec<LayoutNode> = Vec::new();

    for ln in layout_nodes {
        result.push(ln.clone()); // keep compound node (for border rendering)

        if let Some(ci) = compound_map.get(ln.id.as_str()) {
            // Place member nodes horizontally inside compound
            let mut member_x = ln.x + 1 + SG_PAD_X; // border + padding
            let member_y = ln.y + 2; // border + title row

            for (i, mid) in ci.member_ids.iter().enumerate() {
                result.push(LayoutNode {
                    id: mid.clone(),
                    layer: ln.layer,
                    order: ln.order,
                    x: member_x,
                    y: member_y,
                    width: ci.member_widths[i],
                    height: ci.member_heights[i],
                });
                member_x += ci.member_widths[i] + SG_INNER_GAP;
            }
        }
    }

    result
}

// ─── Full Layout Pipeline ────────────────────────────────────────────────────

/// Run the full layout pipeline and return positioned nodes + routed edges.
///
/// Steps:
///   1. Layer assignment (includes cycle removal via greedy-FAS).
///   2. Dummy node insertion for skip-level edges.
///   3. Crossing minimisation (barycenter heuristic, multi-pass).
///   4. Coordinate assignment.
///   5. Edge routing (orthogonal, through inter-layer gap spaces).
pub fn full_layout(gir: &GraphIR) -> (Vec<LayoutNode>, Vec<RoutedEdge>) {
    full_layout_with_padding(gir, NODE_PADDING)
}

/// Like `full_layout` but allows the caller to control the node padding
/// (number of spaces inside the node border on each side of the label).
pub fn full_layout_with_padding(gir: &GraphIR, padding: usize) -> (Vec<LayoutNode>, Vec<RoutedEdge>) {
    let has_subgraphs = !gir.subgraph_members.is_empty();

    if !has_subgraphs {
        // No subgraphs — use original pipeline
        let la = LayerAssignment::assign(gir);
        let (dag, reversed_edges) = remove_cycles(&gir.digraph);
        let aug = insert_dummy_nodes(&dag, &la);
        let ordering = minimise_crossings(&aug);
        let layout_nodes = assign_coordinates_padded(&ordering, &aug, padding, &HashMap::new());
        let routed_edges = route_edges(gir, &layout_nodes, &aug, &reversed_edges);
        return (layout_nodes, routed_edges);
    }

    // Collapse subgraphs into compound nodes
    let (collapsed, compounds) = collapse_subgraphs(gir, padding);
    let dim_overrides = compute_compound_dimensions(&compounds, padding);

    // Run Sugiyama on collapsed graph
    let la = LayerAssignment::assign(&collapsed);
    let (dag, reversed_edges) = remove_cycles(&collapsed.digraph);
    let aug = insert_dummy_nodes(&dag, &la);
    let ordering = minimise_crossings(&aug);
    let layout_nodes = assign_coordinates_padded(&ordering, &aug, padding, &dim_overrides);

    // Expand compound nodes → add member nodes inside
    let expanded = expand_compound_nodes(&layout_nodes, &compounds);

    // Route edges using collapsed graph (edges reference compound node ids)
    let routed_edges = route_edges(&collapsed, &expanded, &aug, &reversed_edges);

    (expanded, routed_edges)
}

/// Compute (max_line_width, line_count) for a label that may contain newlines.
fn label_dimensions(label: &str) -> (usize, usize) {
    if label.is_empty() {
        return (0, 1);
    }
    let lines: Vec<&str> = label.split('\n').collect();
    let max_w = lines.iter().map(|l| l.len()).max().unwrap_or(0);
    (max_w, lines.len())
}

/// Internal: coordinate assignment with a caller-specified padding value.
///
/// `size_overrides` maps node id → (width, height) for compound nodes or
/// other nodes whose dimensions can't be computed from the label alone.
fn assign_coordinates_padded(
    ordering: &[Vec<String>],
    aug: &AugmentedGraph,
    padding: usize,
    size_overrides: &HashMap<String, (usize, usize)>,
) -> Vec<LayoutNode> {
    // Build label info map: id -> (max_line_width, line_count)
    let id_to_label_info: HashMap<&str, (usize, usize)> = aug
        .graph
        .node_indices()
        .map(|ni| (aug.graph[ni].id.as_str(), label_dimensions(&aug.graph[ni].label)))
        .collect();

    // Compute (width, height) for each node, respecting overrides.
    let node_dims = |id: &str| -> (usize, usize) {
        if let Some(&dims) = size_overrides.get(id) {
            return dims;
        }
        let (max_line_w, line_count) = id_to_label_info.get(id).copied().unwrap_or((0, 1));
        let is_dummy = max_line_w == 0 && id.starts_with(DUMMY_PREFIX);
        let width = if is_dummy { 1 } else { max_line_w + 2 + 2 * padding };
        let height = if is_dummy { NODE_HEIGHT } else { 2 + line_count };
        (width, height)
    };

    // First pass: compute per-layer max height.
    let mut layer_max_height: Vec<usize> = vec![NODE_HEIGHT; ordering.len()];
    for (layer_idx, layer_nodes) in ordering.iter().enumerate() {
        for id in layer_nodes {
            let (_, h) = node_dims(id);
            if h > layer_max_height[layer_idx] {
                layer_max_height[layer_idx] = h;
            }
        }
    }

    // Compute layer Y offsets using actual max heights.
    let layer_y: Vec<usize> = {
        let mut y = 0;
        layer_max_height
            .iter()
            .map(|&h| {
                let top = y;
                y += h + V_GAP;
                top
            })
            .collect()
    };

    // Compute total width per layer for centering.
    let layer_total_widths: Vec<usize> = ordering
        .iter()
        .map(|layer_nodes| {
            let w_sum: usize = layer_nodes.iter().map(|id| node_dims(id).0).sum();
            let gaps = if layer_nodes.len() > 1 { (layer_nodes.len() - 1) * H_GAP } else { 0 };
            w_sum + gaps
        })
        .collect();
    let max_layer_w = layer_total_widths.iter().copied().max().unwrap_or(0);
    let center_col = max_layer_w / 2;

    let mut nodes: Vec<LayoutNode> = Vec::new();
    for (layer_idx, layer_nodes) in ordering.iter().enumerate() {
        // Center this layer's midpoint on center_col.
        let offset = center_col.saturating_sub(layer_total_widths[layer_idx] / 2);
        let mut x = offset;
        for (order, id) in layer_nodes.iter().enumerate() {
            let (width, height) = node_dims(id);
            nodes.push(LayoutNode {
                id: id.clone(),
                layer: layer_idx,
                order,
                x,
                y: layer_y[layer_idx],
                width,
                height,
            });
            x += width + H_GAP;
        }
    }

    // ── Barycenter refinement: align layers with parent/child centers ──
    //
    // After the initial center-based placement, shift each layer so the
    // average center of its nodes aligns with the average center of their
    // parents (top-down) or children (bottom-up).  Only small shifts
    // (≤ H_GAP) are applied to correct integer-rounding misalignment
    // without causing large displacements.

    let node_idx: HashMap<String, usize> = nodes.iter().enumerate()
        .map(|(i, n)| (n.id.clone(), i))
        .collect();

    // Top-down pass: align each layer under its parent centers.
    for layer_idx in 1..ordering.len() {
        let mut sum_child: isize = 0;
        let mut sum_parent: isize = 0;
        let mut count: usize = 0;

        for id in &ordering[layer_idx] {
            let ni = node_idx[id.as_str()];
            let child_center = (nodes[ni].x + nodes[ni].width / 2) as isize;

            for edge in aug.graph.edge_references() {
                let src_id = &aug.graph[edge.source()].id;
                let tgt_id = &aug.graph[edge.target()].id;
                if tgt_id == id && !src_id.starts_with(DUMMY_PREFIX) {
                    if let Some(&pi) = node_idx.get(src_id.as_str()) {
                        if nodes[pi].layer + 1 == layer_idx {
                            let parent_center = (nodes[pi].x + nodes[pi].width / 2) as isize;
                            sum_child += child_center;
                            sum_parent += parent_center;
                            count += 1;
                        }
                    }
                }
            }
        }

        if count == 0 { continue; }
        let shift = sum_parent / count as isize - sum_child / count as isize;
        if shift.abs() > H_GAP as isize { continue; }

        for id in &ordering[layer_idx] {
            let ni = node_idx[id.as_str()];
            nodes[ni].x = (nodes[ni].x as isize + shift).max(0) as usize;
        }
    }

    // Bottom-up pass: align each layer over its child centers.
    for layer_idx in (0..ordering.len().saturating_sub(1)).rev() {
        let mut sum_node: isize = 0;
        let mut sum_child: isize = 0;
        let mut count: usize = 0;

        for id in &ordering[layer_idx] {
            let ni = node_idx[id.as_str()];
            let node_center = (nodes[ni].x + nodes[ni].width / 2) as isize;

            for edge in aug.graph.edge_references() {
                let src_id = &aug.graph[edge.source()].id;
                let tgt_id = &aug.graph[edge.target()].id;
                if src_id == id && !tgt_id.starts_with(DUMMY_PREFIX) {
                    if let Some(&ci) = node_idx.get(tgt_id.as_str()) {
                        if nodes[ci].layer == layer_idx + 1 {
                            let child_center = (nodes[ci].x + nodes[ci].width / 2) as isize;
                            sum_node += node_center;
                            sum_child += child_center;
                            count += 1;
                        }
                    }
                }
            }
        }

        if count == 0 { continue; }
        let shift = sum_child / count as isize - sum_node / count as isize;
        if shift.abs() > H_GAP as isize { continue; }

        for id in &ordering[layer_idx] {
            let ni = node_idx[id.as_str()];
            nodes[ni].x = (nodes[ni].x as isize + shift).max(0) as usize;
        }
    }

    // Normalize: shift everything so the leftmost node starts at x=0.
    let min_x = nodes.iter().map(|n| n.x).min().unwrap_or(0);
    if min_x > 0 {
        for n in &mut nodes {
            n.x -= min_x;
        }
    }

    nodes
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{self, NodeShape};

    fn make_node(id: &str) -> NodeData {
        NodeData {
            id: id.to_string(),
            label: id.to_string(),
            shape: NodeShape::Rectangle,
            attrs: vec![],
            subgraph: None,
        }
    }

    fn make_edge() -> EdgeData {
        EdgeData {
            edge_type: ast::EdgeType::Arrow,
            label: None,
            attrs: vec![],
        }
    }

    #[test]
    fn test_dag_has_no_reversed_edges() {
        // A → B → C  (simple DAG, no cycles)
        let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let a = g.add_node(make_node("A"));
        let b = g.add_node(make_node("B"));
        let c = g.add_node(make_node("C"));
        g.add_edge(a, b, make_edge());
        g.add_edge(b, c, make_edge());

        let (_, reversed) = remove_cycles(&g);
        assert!(
            reversed.is_empty(),
            "DAG should have no reversed edges, got: {:?}",
            reversed
        );
    }

    #[test]
    fn test_single_cycle_reversed() {
        // A → B → A  (2-cycle)
        let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let a = g.add_node(make_node("A"));
        let b = g.add_node(make_node("B"));
        g.add_edge(a, b, make_edge());
        g.add_edge(b, a, make_edge());

        let (result, reversed) = remove_cycles(&g);
        assert_eq!(reversed.len(), 1, "Should reverse exactly one edge");
        assert!(
            !petgraph::algo::is_cyclic_directed(&result),
            "Result should be a DAG"
        );
    }

    #[test]
    fn test_self_loop_reversed() {
        // A → A  (self-loop)
        let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let a = g.add_node(make_node("A"));
        g.add_edge(a, a, make_edge());

        let (result, reversed) = remove_cycles(&g);
        assert_eq!(reversed.len(), 1, "Self-loop should be counted as reversed");
        assert!(
            !petgraph::algo::is_cyclic_directed(&result),
            "Result should be a DAG"
        );
    }

    #[test]
    fn test_complex_cycle() {
        // A → B → C → A  (3-cycle) plus D → B
        let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let a = g.add_node(make_node("A"));
        let b = g.add_node(make_node("B"));
        let c = g.add_node(make_node("C"));
        let d = g.add_node(make_node("D"));
        g.add_edge(a, b, make_edge());
        g.add_edge(b, c, make_edge());
        g.add_edge(c, a, make_edge()); // back-edge
        g.add_edge(d, b, make_edge());

        let (result, _) = remove_cycles(&g);
        assert!(
            !petgraph::algo::is_cyclic_directed(&result),
            "Result should be a DAG"
        );
    }

    #[test]
    fn test_empty_graph() {
        let g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let (result, reversed) = remove_cycles(&g);
        assert_eq!(result.node_count(), 0);
        assert!(reversed.is_empty());
    }

    // ─── insert_dummy_nodes tests ──────────────────────────────────────────

    /// Build a minimal LayerAssignment from a map of id→layer.
    fn make_layer_assignment(layers: HashMap<String, usize>) -> LayerAssignment {
        let layer_count = layers.values().copied().max().unwrap_or(0) + 1;
        LayerAssignment {
            layers,
            layer_count,
            reversed_edges: HashSet::new(),
        }
    }

    #[test]
    fn test_adjacent_edge_no_dummy() {
        // A(0) → B(1): span 1 — no dummy needed
        let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let a = g.add_node(make_node("A"));
        let b = g.add_node(make_node("B"));
        g.add_edge(a, b, make_edge());

        let mut layers = HashMap::new();
        layers.insert("A".to_string(), 0);
        layers.insert("B".to_string(), 1);
        let la = make_layer_assignment(layers);

        let aug = insert_dummy_nodes(&g, &la);

        // Should have exactly 2 nodes (A and B) and 1 edge.
        assert_eq!(aug.graph.node_count(), 2, "No dummy nodes should be added");
        assert_eq!(aug.graph.edge_count(), 1);
        assert!(aug.dummy_edges.is_empty());
    }

    #[test]
    fn test_long_edge_one_dummy() {
        // A(0) → C(2): span 2 — one dummy in layer 1
        let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let a = g.add_node(make_node("A"));
        let c = g.add_node(make_node("C"));
        g.add_edge(a, c, make_edge());

        let mut layers = HashMap::new();
        layers.insert("A".to_string(), 0);
        layers.insert("C".to_string(), 2);
        let la = make_layer_assignment(layers);

        let aug = insert_dummy_nodes(&g, &la);

        // 3 nodes total: A, C, and 1 dummy
        assert_eq!(aug.graph.node_count(), 3, "Should have 1 dummy node");
        // 2 edges: A→dummy and dummy→C
        assert_eq!(aug.graph.edge_count(), 2);
        assert_eq!(aug.dummy_edges.len(), 1);

        let de = &aug.dummy_edges[0];
        assert_eq!(de.original_src, "A");
        assert_eq!(de.original_tgt, "C");
        assert_eq!(de.dummy_ids.len(), 1);

        // The dummy node should be in layer 1
        let dummy_id = &de.dummy_ids[0];
        assert_eq!(aug.layers[dummy_id], 1);

        // All edges must connect adjacent layers
        for edge in aug.graph.edge_references() {
            let src_id = &aug.graph[edge.source()].id;
            let tgt_id = &aug.graph[edge.target()].id;
            let src_layer = aug.layers[src_id];
            let tgt_layer = aug.layers[tgt_id];
            assert!(
                tgt_layer >= src_layer && tgt_layer - src_layer <= 1,
                "Edge {}→{} spans {} layers (expected ≤1)",
                src_id, tgt_id, tgt_layer.saturating_sub(src_layer)
            );
        }
    }

    #[test]
    fn test_long_edge_two_dummies() {
        // A(0) → D(3): span 3 — two dummies in layers 1 and 2
        let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let a = g.add_node(make_node("A"));
        let d = g.add_node(make_node("D"));
        g.add_edge(a, d, make_edge());

        let mut layers = HashMap::new();
        layers.insert("A".to_string(), 0);
        layers.insert("D".to_string(), 3);
        let la = make_layer_assignment(layers);

        let aug = insert_dummy_nodes(&g, &la);

        // 4 nodes: A, D, dummy_0, dummy_1
        assert_eq!(aug.graph.node_count(), 4);
        // 3 edges: A→d0, d0→d1, d1→D
        assert_eq!(aug.graph.edge_count(), 3);
        assert_eq!(aug.dummy_edges.len(), 1);

        let de = &aug.dummy_edges[0];
        assert_eq!(de.dummy_ids.len(), 2);
        assert_eq!(aug.layers[&de.dummy_ids[0]], 1);
        assert_eq!(aug.layers[&de.dummy_ids[1]], 2);
    }

    #[test]
    fn test_multiple_long_edges_independent() {
        // A(0) → C(2) and B(0) → D(2): two independent long edges
        let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let a = g.add_node(make_node("A"));
        let b = g.add_node(make_node("B"));
        let c = g.add_node(make_node("C"));
        let d = g.add_node(make_node("D"));
        g.add_edge(a, c, make_edge());
        g.add_edge(b, d, make_edge());

        let mut layers = HashMap::new();
        layers.insert("A".to_string(), 0);
        layers.insert("B".to_string(), 0);
        layers.insert("C".to_string(), 2);
        layers.insert("D".to_string(), 2);
        let la = make_layer_assignment(layers);

        let aug = insert_dummy_nodes(&g, &la);

        // 6 nodes: A, B, C, D + 2 dummies
        assert_eq!(aug.graph.node_count(), 6);
        // 4 edges: A→d0, d0→C, B→d1, d1→D
        assert_eq!(aug.graph.edge_count(), 4);
        assert_eq!(aug.dummy_edges.len(), 2);
    }

    #[test]
    fn test_mixed_short_and_long_edges() {
        // A(0) → B(1) [short] and A(0) → C(2) [long]
        let mut g: DiGraph<NodeData, EdgeData> = DiGraph::new();
        let a = g.add_node(make_node("A"));
        let b = g.add_node(make_node("B"));
        let c = g.add_node(make_node("C"));
        g.add_edge(a, b, make_edge());
        g.add_edge(a, c, make_edge());

        let mut layers = HashMap::new();
        layers.insert("A".to_string(), 0);
        layers.insert("B".to_string(), 1);
        layers.insert("C".to_string(), 2);
        let la = make_layer_assignment(layers);

        let aug = insert_dummy_nodes(&g, &la);

        // 4 nodes: A, B, C, 1 dummy
        assert_eq!(aug.graph.node_count(), 4);
        // 3 edges: A→B (short), A→dummy, dummy→C
        assert_eq!(aug.graph.edge_count(), 3);
        assert_eq!(aug.dummy_edges.len(), 1);

        // Verify all edges are adjacent-layer
        for edge in aug.graph.edge_references() {
            let src_id = &aug.graph[edge.source()].id;
            let tgt_id = &aug.graph[edge.target()].id;
            let diff = aug.layers[tgt_id].saturating_sub(aug.layers[src_id]);
            assert!(diff <= 1, "Edge {}→{} not adjacent-layer", src_id, tgt_id);
        }
    }
}
