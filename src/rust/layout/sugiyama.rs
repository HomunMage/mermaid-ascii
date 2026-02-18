//! Sugiyama layered graph layout algorithm.
//!
//! Mirrors Python's layout/sugiyama.py.
//!
//! Phases:
//!   1. Cycle removal (greedy-FAS)
//!   2. Layer assignment
//!   3. Dummy node insertion
//!   4. Crossing minimisation (barycenter)
//!   5. Coordinate assignment
//!   6. Edge routing (orthogonal)
//!   7. Subgraph collapse/expand

use std::collections::{HashMap, HashSet};

use petgraph::graph::{DiGraph, NodeIndex};

use super::graph::{EdgeData, GraphIR, NodeData};
use super::types::{COMPOUND_PREFIX, DUMMY_PREFIX, LayoutNode, LayoutResult, Point, RoutedEdge};
use crate::syntax::types::{Direction, EdgeType, NodeShape};

// ─── Geometry constants ──────────────────────────────────────────────────────

pub const NODE_PADDING: i64 = 1;
pub const H_GAP: i64 = 4;
pub const V_GAP: i64 = 3;
pub const NODE_HEIGHT: i64 = 3;

// ─── Mini-graph helpers ───────────────────────────────────────────────────────

/// Lightweight graph representation used inside Sugiyama (avoids petgraph complexity).
/// Maps node_id → (successors, predecessors).
pub struct AdjGraph {
    /// All node ids.
    nodes: Vec<String>,
    /// node_id → successors
    successors: HashMap<String, Vec<String>>,
    /// node_id → predecessors
    predecessors: HashMap<String, Vec<String>>,
    /// Edges with data: (src, tgt, edge_data)
    edges: Vec<(String, String, Option<EdgeData>)>,
}

impl AdjGraph {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            successors: HashMap::new(),
            predecessors: HashMap::new(),
            edges: Vec::new(),
        }
    }

    fn add_node(&mut self, id: &str, data: NodeData) {
        if !self.successors.contains_key(id) {
            self.nodes.push(id.to_string());
            self.successors.insert(id.to_string(), Vec::new());
            self.predecessors.insert(id.to_string(), Vec::new());
            let _ = data; // data stored separately in node_data map
        }
    }

    fn add_edge(&mut self, src: &str, tgt: &str, data: Option<EdgeData>) {
        self.successors
            .entry(src.to_string())
            .or_default()
            .push(tgt.to_string());
        self.predecessors
            .entry(tgt.to_string())
            .or_default()
            .push(src.to_string());
        self.edges.push((src.to_string(), tgt.to_string(), data));
    }

    fn out_degree(&self, id: &str) -> usize {
        self.successors.get(id).map(|v| v.len()).unwrap_or(0)
    }

    fn in_degree(&self, id: &str) -> usize {
        self.predecessors.get(id).map(|v| v.len()).unwrap_or(0)
    }

    fn successors_of(&self, id: &str) -> &[String] {
        self.successors.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn predecessors_of(&self, id: &str) -> &[String] {
        self.predecessors
            .get(id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Build an AdjGraph from a petgraph DiGraph (for cycle removal + layer assignment).
fn petgraph_to_adj(
    gir_digraph: &DiGraph<NodeData, EdgeData>,
) -> (AdjGraph, HashMap<String, NodeData>) {
    let mut ag = AdjGraph::new();
    let mut node_data_map: HashMap<String, NodeData> = HashMap::new();

    for idx in gir_digraph.node_indices() {
        let data = gir_digraph[idx].clone();
        let id = data.id.clone();
        node_data_map.insert(id.clone(), data.clone());
        ag.add_node(&id, data);
    }

    for eidx in gir_digraph.edge_indices() {
        let (src_idx, tgt_idx) = gir_digraph.edge_endpoints(eidx).unwrap();
        let src = gir_digraph[src_idx].id.clone();
        let tgt = gir_digraph[tgt_idx].id.clone();
        let edge_data = gir_digraph[eidx].clone();
        ag.add_edge(&src, &tgt, Some(edge_data));
    }

    (ag, node_data_map)
}

// ─── Cycle Removal (Greedy-FAS) ─────────────────────────────────────────────

/// Compute a node ordering using the greedy-FAS heuristic.
fn greedy_fas_ordering(ag: &AdjGraph) -> Vec<String> {
    let mut active: HashSet<String> = ag.nodes.iter().cloned().collect();
    let mut out_deg: HashMap<String, i64> = HashMap::new();
    let mut in_deg: HashMap<String, i64> = HashMap::new();

    for node in &ag.nodes {
        out_deg.insert(node.clone(), ag.out_degree(node) as i64);
        in_deg.insert(node.clone(), ag.in_degree(node) as i64);
    }

    let mut s1: Vec<String> = Vec::new();
    let mut s2: Vec<String> = Vec::new();

    while !active.is_empty() {
        let mut changed = true;
        while changed {
            changed = false;
            let sinks: Vec<String> = active
                .iter()
                .filter(|n| *out_deg.get(*n).unwrap_or(&0) == 0)
                .cloned()
                .collect();
            if !sinks.is_empty() {
                changed = true;
                for sink in &sinks {
                    active.remove(sink);
                    s2.push(sink.clone());
                    for pred in ag.predecessors_of(sink) {
                        if active.contains(pred) {
                            *out_deg.entry(pred.clone()).or_insert(0) -= 1;
                        }
                    }
                }
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            let sources: Vec<String> = active
                .iter()
                .filter(|n| *in_deg.get(*n).unwrap_or(&0) == 0)
                .cloned()
                .collect();
            if !sources.is_empty() {
                changed = true;
                for source in &sources {
                    active.remove(source);
                    s1.push(source.clone());
                    for succ in ag.successors_of(source) {
                        if active.contains(succ) {
                            *in_deg.entry(succ.clone()).or_insert(0) -= 1;
                        }
                    }
                }
            }
        }

        if !active.is_empty() {
            let best = active
                .iter()
                .max_by_key(|n| {
                    out_deg.get(*n).copied().unwrap_or(0) - in_deg.get(*n).copied().unwrap_or(0)
                })
                .unwrap()
                .clone();
            active.remove(&best);
            s1.push(best.clone());
            for succ in ag.successors_of(&best).to_vec() {
                if active.contains(&succ) {
                    *in_deg.entry(succ).or_insert(0) -= 1;
                }
            }
            for pred in ag.predecessors_of(&best).to_vec() {
                if active.contains(&pred) {
                    *out_deg.entry(pred).or_insert(0) -= 1;
                }
            }
        }
    }

    s2.reverse();
    s1.extend(s2);
    s1
}

/// Remove cycles using greedy-FAS. Returns (dag as AdjGraph, reversed_edges, node_data_map).
fn remove_cycles(
    ag: &AdjGraph,
    node_data_map: &HashMap<String, NodeData>,
) -> (AdjGraph, HashSet<(String, String)>) {
    if ag.nodes.is_empty() {
        return (AdjGraph::new(), HashSet::new());
    }

    let ordering = greedy_fas_ordering(ag);
    let position: HashMap<String, usize> = ordering
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    let mut reversed_edges: HashSet<(String, String)> = HashSet::new();
    for (src, tgt, _) in &ag.edges {
        let is_self_loop = src == tgt;
        if is_self_loop {
            reversed_edges.insert((src.clone(), tgt.clone()));
            continue;
        }
        let src_pos = position.get(src).copied().unwrap_or(0);
        let tgt_pos = position.get(tgt).copied().unwrap_or(0);
        if src_pos > tgt_pos {
            reversed_edges.insert((src.clone(), tgt.clone()));
        }
    }

    let mut dag = AdjGraph::new();
    for node_id in &ag.nodes {
        let data = node_data_map
            .get(node_id)
            .cloned()
            .unwrap_or_else(|| NodeData {
                id: node_id.clone(),
                label: node_id.clone(),
                shape: NodeShape::Rectangle,
                attrs: Vec::new(),
                subgraph: None,
            });
        dag.add_node(node_id, data);
    }

    for (src, tgt, edge_data) in &ag.edges {
        if src == tgt {
            continue;
        }
        if reversed_edges.contains(&(src.clone(), tgt.clone())) {
            dag.add_edge(tgt, src, edge_data.clone());
        } else {
            dag.add_edge(src, tgt, edge_data.clone());
        }
    }

    (dag, reversed_edges)
}

// ─── Layer Assignment ────────────────────────────────────────────────────────

pub struct LayerAssignment {
    pub layers: HashMap<String, usize>,
    pub layer_count: usize,
    pub reversed_edges: HashSet<(String, String)>,
}

impl LayerAssignment {
    pub fn assign(gir: &GraphIR) -> Self {
        let (ag, node_data_map) = petgraph_to_adj(&gir.digraph);
        let (dag, reversed_edges) = remove_cycles(&ag, &node_data_map);

        let mut layers: HashMap<String, usize> = dag.nodes.iter().map(|n| (n.clone(), 0)).collect();

        let mut changed = true;
        while changed {
            changed = false;
            for (src, tgt, _) in &dag.edges {
                let src_layer = *layers.get(src).unwrap_or(&0);
                let tgt_layer = layers.entry(tgt.clone()).or_insert(0);
                if *tgt_layer < src_layer + 1 {
                    *tgt_layer = src_layer + 1;
                    changed = true;
                }
            }
        }

        let layer_count = if layers.is_empty() {
            1
        } else {
            layers.values().copied().max().unwrap_or(0) + 1
        };

        Self {
            layers,
            layer_count,
            reversed_edges,
        }
    }

    /// Build a LayerAssignment from a collapsed GraphIR (used for subgraphs).
    pub fn assign_from_adj(ag: &AdjGraph, node_data_map: &HashMap<String, NodeData>) -> Self {
        let (dag, reversed_edges) = remove_cycles(ag, node_data_map);

        let mut layers: HashMap<String, usize> = dag.nodes.iter().map(|n| (n.clone(), 0)).collect();

        let mut changed = true;
        while changed {
            changed = false;
            for (src, tgt, _) in &dag.edges {
                let src_layer = *layers.get(src).unwrap_or(&0);
                let tgt_layer = layers.entry(tgt.clone()).or_insert(0);
                if *tgt_layer < src_layer + 1 {
                    *tgt_layer = src_layer + 1;
                    changed = true;
                }
            }
        }

        let layer_count = if layers.is_empty() {
            1
        } else {
            layers.values().copied().max().unwrap_or(0) + 1
        };

        Self {
            layers,
            layer_count,
            reversed_edges,
        }
    }
}

// ─── Dummy Node Insertion ────────────────────────────────────────────────────

pub struct DummyEdge {
    pub original_src: String,
    pub original_tgt: String,
    pub dummy_ids: Vec<String>,
    pub edge_data: Option<EdgeData>,
}

pub struct AugmentedGraph {
    pub ag: AdjGraph,
    pub node_data: HashMap<String, NodeData>,
    pub layers: HashMap<String, usize>,
    pub layer_count: usize,
    pub dummy_edges: Vec<DummyEdge>,
}

pub fn insert_dummy_nodes(
    dag: AdjGraph,
    dag_node_data: HashMap<String, NodeData>,
    la: &LayerAssignment,
) -> AugmentedGraph {
    let mut new_ag = AdjGraph::new();
    let mut new_node_data: HashMap<String, NodeData> = HashMap::new();

    // Copy all original nodes
    for node_id in &dag.nodes {
        let data = dag_node_data
            .get(node_id)
            .cloned()
            .unwrap_or_else(|| NodeData {
                id: node_id.clone(),
                label: node_id.clone(),
                shape: NodeShape::Rectangle,
                attrs: Vec::new(),
                subgraph: None,
            });
        new_ag.add_node(node_id, data.clone());
        new_node_data.insert(node_id.clone(), data);
    }

    let mut layers: HashMap<String, usize> = la.layers.clone();
    let mut dummy_edges: Vec<DummyEdge> = Vec::new();
    let mut edge_counter: usize = 0;

    // Snapshot edges before modifying
    let all_edges: Vec<(String, String, Option<EdgeData>)> = dag.edges.clone();

    for (src_id, tgt_id, edge_data) in all_edges {
        let src_layer = *layers.get(&src_id).unwrap_or(&0);
        let tgt_layer = *layers.get(&tgt_id).unwrap_or(&0);
        let layer_diff = if tgt_layer > src_layer {
            tgt_layer - src_layer
        } else {
            1
        };

        if layer_diff <= 1 {
            new_ag.add_edge(&src_id, &tgt_id, edge_data);
            continue;
        }

        let steps = layer_diff - 1;
        let this_edge = edge_counter;
        edge_counter += 1;

        let mut dummy_ids: Vec<String> = Vec::new();
        let mut chain_prev = src_id.clone();

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
            new_ag.add_node(&dummy_id, dummy_data.clone());
            new_node_data.insert(dummy_id.clone(), dummy_data);
            layers.insert(dummy_id.clone(), dummy_layer);
            dummy_ids.push(dummy_id.clone());

            let segment_edge = EdgeData {
                edge_type: edge_data
                    .as_ref()
                    .map(|e| e.edge_type.clone())
                    .unwrap_or(EdgeType::Arrow),
                label: None,
                attrs: Vec::new(),
            };
            new_ag.add_edge(&chain_prev, &dummy_id, Some(segment_edge));
            chain_prev = dummy_id;
        }

        let last_segment = EdgeData {
            edge_type: edge_data
                .as_ref()
                .map(|e| e.edge_type.clone())
                .unwrap_or(EdgeType::Arrow),
            label: edge_data.as_ref().and_then(|e| e.label.clone()),
            attrs: edge_data
                .as_ref()
                .map(|e| e.attrs.clone())
                .unwrap_or_default(),
        };
        new_ag.add_edge(&chain_prev, &tgt_id, Some(last_segment));

        dummy_edges.push(DummyEdge {
            original_src: src_id,
            original_tgt: tgt_id,
            dummy_ids,
            edge_data,
        });
    }

    let layer_count = if layers.is_empty() {
        1
    } else {
        layers.values().copied().max().unwrap_or(0) + 1
    };

    AugmentedGraph {
        ag: new_ag,
        node_data: new_node_data,
        layers,
        layer_count,
        dummy_edges,
    }
}

// ─── Crossing Minimization ───────────────────────────────────────────────────

fn barycenter(
    node_id: &str,
    ag: &AdjGraph,
    neighbor_pos: &HashMap<String, f64>,
    direction: &str,
) -> f64 {
    let neighbors: Vec<&str> = if direction == "incoming" {
        ag.predecessors_of(node_id)
            .iter()
            .map(|s| s.as_str())
            .collect()
    } else {
        ag.successors_of(node_id)
            .iter()
            .map(|s| s.as_str())
            .collect()
    };
    let positions: Vec<f64> = neighbors
        .iter()
        .filter_map(|nb| neighbor_pos.get(*nb).copied())
        .collect();
    if positions.is_empty() {
        f64::INFINITY
    } else {
        positions.iter().sum::<f64>() / positions.len() as f64
    }
}

fn count_crossings(ordering: &[Vec<String>], ag: &AdjGraph) -> usize {
    let mut total = 0usize;
    for l_idx in 0..ordering.len().saturating_sub(1) {
        let tgt_pos: HashMap<&str, usize> = ordering[l_idx + 1]
            .iter()
            .enumerate()
            .map(|(i, nid)| (nid.as_str(), i))
            .collect();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (sp, src_id) in ordering[l_idx].iter().enumerate() {
            for nb in ag.successors_of(src_id) {
                if let Some(&tp) = tgt_pos.get(nb.as_str()) {
                    edges.push((sp, tp));
                }
            }
        }
        for i in 0..edges.len() {
            for j in (i + 1)..edges.len() {
                let (ei0, ei1) = edges[i];
                let (ej0, ej1) = edges[j];
                if (ei0 < ej0 && ei1 > ej1) || (ei0 > ej0 && ei1 < ej1) {
                    total += 1;
                }
            }
        }
    }
    total
}

pub fn minimise_crossings(aug: &AugmentedGraph) -> Vec<Vec<String>> {
    let layer_count = aug.layer_count;
    let mut ordering: Vec<Vec<String>> = vec![Vec::new(); layer_count];

    let mut sorted_nodes: Vec<&str> = aug.ag.nodes.iter().map(|s| s.as_str()).collect();
    sorted_nodes.sort();
    for node_id in sorted_nodes {
        let layer = *aug.layers.get(node_id).unwrap_or(&0);
        if layer < ordering.len() {
            ordering[layer].push(node_id.to_string());
        }
    }

    let max_passes = 24;
    let mut best = count_crossings(&ordering, &aug.ag);

    for _pass in 0..max_passes {
        for layer_idx in 1..layer_count {
            let prev_ids = ordering[layer_idx - 1].clone();
            let prev: HashMap<String, f64> = prev_ids
                .iter()
                .enumerate()
                .map(|(i, nid)| (nid.clone(), i as f64))
                .collect();
            ordering[layer_idx].sort_by(|a, b| {
                let ba = barycenter(a, &aug.ag, &prev, "incoming");
                let bb = barycenter(b, &aug.ag, &prev, "incoming");
                ba.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for layer_idx in (0..layer_count.saturating_sub(1)).rev() {
            let next_ids = ordering[layer_idx + 1].clone();
            let nxt: HashMap<String, f64> = next_ids
                .iter()
                .enumerate()
                .map(|(i, nid)| (nid.clone(), i as f64))
                .collect();
            ordering[layer_idx].sort_by(|a, b| {
                let ba = barycenter(a, &aug.ag, &nxt, "outgoing");
                let bb = barycenter(b, &aug.ag, &nxt, "outgoing");
                ba.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let new_crossings = count_crossings(&ordering, &aug.ag);
        if new_crossings >= best {
            break;
        }
        best = new_crossings;
    }

    ordering
}

// ─── Coordinate Assignment ───────────────────────────────────────────────────

fn label_dimensions(label: &str) -> (i64, i64) {
    if label.is_empty() {
        return (0, 1);
    }
    let lines: Vec<&str> = label.split('\n').collect();
    let max_w = lines.iter().map(|l| l.len()).max().unwrap_or(0) as i64;
    (max_w, lines.len() as i64)
}

pub fn assign_coordinates_padded(
    ordering: &[Vec<String>],
    aug: &AugmentedGraph,
    padding: i64,
    size_overrides: &HashMap<String, (i64, i64)>,
    direction: &Direction,
) -> Vec<LayoutNode> {
    let is_lr_or_rl = matches!(direction, Direction::LR | Direction::RL);
    let h_gap = if is_lr_or_rl { V_GAP } else { H_GAP };
    let v_gap = if is_lr_or_rl { H_GAP } else { V_GAP };

    // Build label info and metadata maps
    let mut id_to_label_info: HashMap<String, (i64, i64)> = HashMap::new();
    let mut id_to_meta: HashMap<String, (String, NodeShape)> = HashMap::new();
    for node_id in &aug.ag.nodes {
        if let Some(nd) = aug.node_data.get(node_id) {
            id_to_label_info.insert(node_id.clone(), label_dimensions(&nd.label));
            id_to_meta.insert(node_id.clone(), (nd.label.clone(), nd.shape.clone()));
        } else {
            id_to_label_info.insert(node_id.clone(), (node_id.len() as i64, 1));
            id_to_meta.insert(node_id.clone(), (node_id.clone(), NodeShape::Rectangle));
        }
    }

    let node_dims = |node_id: &str| -> (i64, i64) {
        if let Some(&(w, h)) = size_overrides.get(node_id) {
            return if is_lr_or_rl { (h, w) } else { (w, h) };
        }
        let (max_line_w, line_count) = id_to_label_info.get(node_id).copied().unwrap_or((0, 1));
        let is_dummy = max_line_w == 0 && node_id.starts_with(DUMMY_PREFIX);
        let width = if is_dummy {
            1
        } else {
            max_line_w + 2 + 2 * padding
        };
        let height = if is_dummy {
            NODE_HEIGHT
        } else {
            2 + line_count
        };
        if is_lr_or_rl {
            (height, width)
        } else {
            (width, height)
        }
    };

    // Compute layer max heights
    let mut layer_max_height: Vec<i64> = vec![NODE_HEIGHT; ordering.len()];
    for (layer_idx, layer_nodes) in ordering.iter().enumerate() {
        for node_id in layer_nodes {
            let (_, h) = node_dims(node_id);
            if h > layer_max_height[layer_idx] {
                layer_max_height[layer_idx] = h;
            }
        }
    }

    // Compute y positions per layer
    let mut layer_y: Vec<i64> = Vec::new();
    let mut y: i64 = 0;
    for &h in &layer_max_height {
        layer_y.push(y);
        y += h + v_gap;
    }

    // Compute total widths per layer
    let mut layer_total_widths: Vec<i64> = Vec::new();
    for layer_nodes in ordering {
        let w_sum: i64 = layer_nodes.iter().map(|nid| node_dims(nid).0).sum();
        let gaps = if layer_nodes.len() > 1 {
            (layer_nodes.len() as i64 - 1) * h_gap
        } else {
            0
        };
        layer_total_widths.push(w_sum + gaps);
    }

    let max_layer_w = layer_total_widths.iter().copied().max().unwrap_or(0);
    let center_col = max_layer_w / 2;

    let mut nodes: Vec<LayoutNode> = Vec::new();
    for (layer_idx, layer_nodes) in ordering.iter().enumerate() {
        let offset = (center_col - layer_total_widths[layer_idx] / 2).max(0);
        let mut x = offset;
        for (order, node_id) in layer_nodes.iter().enumerate() {
            let (width, height) = node_dims(node_id);
            let (label, shape) = id_to_meta
                .get(node_id)
                .cloned()
                .unwrap_or_else(|| (node_id.clone(), NodeShape::Rectangle));
            nodes.push(LayoutNode {
                id: node_id.clone(),
                layer: layer_idx,
                order,
                x,
                y: layer_y[layer_idx],
                width,
                height,
                label,
                shape,
            });
            x += width + h_gap;
        }
    }

    // Barycenter refinement — forward pass (child aligns to parent)
    let mut node_idx: HashMap<String, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.clone(), i))
        .collect();

    #[allow(clippy::needless_range_loop)]
    for layer_idx in 1..ordering.len() {
        let mut sum_child: i64 = 0;
        let mut sum_parent: i64 = 0;
        let mut count: i64 = 0;
        for node_id in &ordering[layer_idx] {
            if let Some(&ni) = node_idx.get(node_id) {
                let child_center = nodes[ni].x + nodes[ni].width / 2;
                for (src, tgt, _) in &aug.ag.edges {
                    if tgt == node_id
                        && !src.starts_with(DUMMY_PREFIX)
                        && node_idx.contains_key(src)
                    {
                        let pi = node_idx[src];
                        if nodes[pi].layer + 1 == layer_idx {
                            let parent_center = nodes[pi].x + nodes[pi].width / 2;
                            sum_child += child_center;
                            sum_parent += parent_center;
                            count += 1;
                        }
                    }
                }
            }
        }
        if count == 0 {
            continue;
        }
        let shift = sum_parent / count - sum_child / count;
        if shift.abs() > h_gap {
            continue;
        }
        for node_id in &ordering[layer_idx] {
            if let Some(&ni) = node_idx.get(node_id) {
                nodes[ni].x = (nodes[ni].x + shift).max(0);
            }
        }
    }

    // Barycenter refinement — backward pass (parent aligns to child)
    for layer_idx in (0..ordering.len().saturating_sub(1)).rev() {
        let mut sum_node: i64 = 0;
        let mut sum_child: i64 = 0;
        let mut count: i64 = 0;
        for node_id in &ordering[layer_idx] {
            if let Some(&ni) = node_idx.get(node_id) {
                let node_center = nodes[ni].x + nodes[ni].width / 2;
                for (src, tgt, _) in &aug.ag.edges {
                    if src == node_id
                        && !tgt.starts_with(DUMMY_PREFIX)
                        && node_idx.contains_key(tgt)
                    {
                        let ci = node_idx[tgt];
                        if nodes[ci].layer == layer_idx + 1 {
                            let child_center = nodes[ci].x + nodes[ci].width / 2;
                            sum_node += node_center;
                            sum_child += child_center;
                            count += 1;
                        }
                    }
                }
            }
        }
        if count == 0 {
            continue;
        }
        let shift = sum_child / count - sum_node / count;
        if shift.abs() > h_gap {
            continue;
        }
        for node_id in &ordering[layer_idx] {
            if let Some(&ni) = node_idx.get(node_id) {
                nodes[ni].x = (nodes[ni].x + shift).max(0);
            }
        }
    }

    // Re-build node_idx after potential moves, then normalize min_x to 0
    node_idx = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.clone(), i))
        .collect();
    let _ = node_idx; // suppress unused warning — was built for refinement above

    if !nodes.is_empty() {
        let min_x = nodes.iter().map(|n| n.x).min().unwrap_or(0);
        if min_x > 0 {
            for n in &mut nodes {
                n.x -= min_x;
            }
        }
    }

    nodes
}

// ─── Edge Routing ────────────────────────────────────────────────────────────

fn compute_orthogonal_waypoints(
    from_node: &LayoutNode,
    to_node: &LayoutNode,
    layer_top_y: &[i64],
    layer_bottom_y: &[i64],
    dummy_xs: &[i64],
) -> Vec<Point> {
    let exit_x = from_node.x + from_node.width / 2;
    let exit_y = from_node.y + from_node.height - 1;
    let entry_x = to_node.x + to_node.width / 2;
    let entry_y = to_node.y;

    let src_layer = from_node.layer;
    let tgt_layer = to_node.layer;

    if src_layer == tgt_layer {
        let bot = layer_bottom_y.get(src_layer).copied().unwrap_or(exit_y + 1);
        let below_y = bot + V_GAP / 2;
        return vec![
            Point::new(exit_x, exit_y),
            Point::new(exit_x, below_y),
            Point::new(entry_x, below_y),
            Point::new(entry_x, entry_y),
        ];
    }

    let low_layer = src_layer.min(tgt_layer);
    let high_layer = src_layer.max(tgt_layer);

    let mut waypoints = vec![Point::new(exit_x, exit_y)];

    let gaps = high_layer - low_layer;
    for gap_idx in 0..gaps {
        let gap = low_layer + gap_idx;
        let gap_start = layer_bottom_y.get(gap).copied().unwrap_or(exit_y + 1);
        let gap_end = layer_top_y
            .get(gap + 1)
            .copied()
            .unwrap_or(gap_start + V_GAP);
        let mid_y = gap_start + (gap_end - gap_start).max(0) / 2;

        let gap_x = if gap_idx < dummy_xs.len() {
            dummy_xs[gap_idx]
        } else if gap_idx == 0 {
            exit_x
        } else {
            entry_x
        };

        let last_wp = waypoints.last().unwrap().clone();
        if last_wp.x != gap_x {
            waypoints.push(Point::new(gap_x, last_wp.y));
        }
        waypoints.push(Point::new(gap_x, mid_y));
    }

    let last_wp = waypoints.last().unwrap().clone();
    if last_wp.x != entry_x {
        waypoints.push(Point::new(entry_x, last_wp.y));
    }
    waypoints.push(Point::new(entry_x, entry_y));

    waypoints
}

pub fn route_edges(
    gir: &GraphIR,
    layout_nodes: &[LayoutNode],
    aug: &AugmentedGraph,
    reversed_edges: &HashSet<(String, String)>,
) -> Vec<RoutedEdge> {
    let node_map: HashMap<&str, &LayoutNode> =
        layout_nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    let layer_count = layout_nodes
        .iter()
        .map(|n| n.layer)
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);
    let mut layer_top_y: Vec<i64> = vec![i64::MAX; layer_count.max(1)];
    let mut layer_bottom_y: Vec<i64> = vec![0i64; layer_count.max(1)];
    for n in layout_nodes {
        if n.y < layer_top_y[n.layer] {
            layer_top_y[n.layer] = n.y;
        }
        let bot = n.y + n.height;
        if bot > layer_bottom_y[n.layer] {
            layer_bottom_y[n.layer] = bot;
        }
    }

    // Build dummy_xs_map from aug dummy edges
    let mut dummy_xs_map: HashMap<(String, String), Vec<i64>> = HashMap::new();
    for de in &aug.dummy_edges {
        let xs: Vec<i64> = de
            .dummy_ids
            .iter()
            .filter_map(|did| node_map.get(did.as_str()))
            .map(|ln| ln.x + ln.width / 2)
            .collect();
        dummy_xs_map.insert((de.original_src.clone(), de.original_tgt.clone()), xs);
    }

    let mut routes: Vec<RoutedEdge> = Vec::new();

    for eidx in gir.digraph.edge_indices() {
        let (src_idx, tgt_idx) = gir.digraph.edge_endpoints(eidx).unwrap();
        let src = gir.digraph[src_idx].id.clone();
        let tgt = gir.digraph[tgt_idx].id.clone();

        if src == tgt {
            continue;
        }

        let edge_data = &gir.digraph[eidx];
        let is_reversed = reversed_edges.contains(&(src.clone(), tgt.clone()));

        let (vis_from, vis_to) = if is_reversed {
            (tgt.clone(), src.clone())
        } else {
            (src.clone(), tgt.clone())
        };

        let from_node = match node_map.get(vis_from.as_str()) {
            Some(n) => n,
            None => continue,
        };
        let to_node = match node_map.get(vis_to.as_str()) {
            Some(n) => n,
            None => continue,
        };

        let empty_xs = Vec::new();
        let dummy_xs = dummy_xs_map
            .get(&(vis_from.clone(), vis_to.clone()))
            .unwrap_or(&empty_xs);

        let waypoints = compute_orthogonal_waypoints(
            from_node,
            to_node,
            &layer_top_y,
            &layer_bottom_y,
            dummy_xs,
        );

        routes.push(RoutedEdge {
            from_id: vis_from,
            to_id: vis_to,
            label: edge_data.label.clone(),
            edge_type: edge_data.edge_type.clone(),
            waypoints,
        });
    }

    routes
}

// ─── Compound Node (Subgraph Collapse/Expand) ────────────────────────────────

const SG_INNER_GAP: i64 = 1;
const SG_PAD_X: i64 = 1;

pub struct CompoundInfo {
    pub sg_name: String,
    pub compound_id: String,
    pub member_ids: Vec<String>,
    pub member_widths: Vec<i64>,
    pub member_heights: Vec<i64>,
    pub max_member_height: i64,
    pub description: Option<String>,
    pub member_labels: Vec<String>,
    pub member_shapes: Vec<NodeShape>,
}

/// Collapse subgraphs into compound nodes. Returns (collapsed AdjGraph, its node_data, compounds).
pub fn collapse_subgraphs(
    gir: &GraphIR,
    padding: i64,
) -> (AdjGraph, HashMap<String, NodeData>, Vec<CompoundInfo>) {
    let mut member_to_sg: HashMap<String, String> = HashMap::new();
    let mut compounds: Vec<CompoundInfo> = Vec::new();

    for (sg_name, members) in &gir.subgraph_members {
        let compound_id = format!("{}{}", COMPOUND_PREFIX, sg_name);
        let mut member_widths: Vec<i64> = Vec::new();
        let mut member_heights: Vec<i64> = Vec::new();
        let mut member_labels: Vec<String> = Vec::new();
        let mut member_shapes: Vec<NodeShape> = Vec::new();

        for mid in members {
            if let Some(idx) = gir.node_index.get(mid) {
                let data = &gir.digraph[*idx];
                let (max_line_w, line_count) = label_dimensions(&data.label);
                member_widths.push(max_line_w + 2 + 2 * padding);
                member_heights.push(2 + line_count);
                member_labels.push(data.label.clone());
                member_shapes.push(data.shape.clone());
            } else {
                member_widths.push(3 + 2 * padding);
                member_heights.push(NODE_HEIGHT);
                member_labels.push(mid.clone());
                member_shapes.push(NodeShape::Rectangle);
            }
            member_to_sg.insert(mid.clone(), sg_name.clone());
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
            member_labels,
            member_shapes,
        });
    }

    let sg_to_compound: HashMap<String, String> = compounds
        .iter()
        .map(|c| (c.sg_name.clone(), c.compound_id.clone()))
        .collect();

    let resolve_endpoint = |node_id: &str| -> String {
        if let Some(sg) = member_to_sg.get(node_id) {
            return sg_to_compound[sg].clone();
        }
        if let Some(cid) = sg_to_compound.get(node_id) {
            return cid.clone();
        }
        node_id.to_string()
    };

    let mut new_ag = AdjGraph::new();
    let mut new_node_data: HashMap<String, NodeData> = HashMap::new();

    // Add non-member, non-subgraph nodes
    for idx in gir.digraph.node_indices() {
        let data = &gir.digraph[idx];
        let id = &data.id;
        if member_to_sg.contains_key(id) {
            continue;
        }
        if sg_to_compound.contains_key(id.as_str()) {
            continue;
        }
        new_ag.add_node(id, data.clone());
        new_node_data.insert(id.clone(), data.clone());
    }

    // Add compound nodes
    for ci in &compounds {
        let compound_data = NodeData {
            id: ci.compound_id.clone(),
            label: ci.sg_name.clone(),
            shape: NodeShape::Rectangle,
            attrs: Vec::new(),
            subgraph: None,
        };
        new_ag.add_node(&ci.compound_id, compound_data.clone());
        new_node_data.insert(ci.compound_id.clone(), compound_data);
    }

    // Add deduplicated edges
    let mut added_edges: HashSet<(String, String)> = HashSet::new();
    for eidx in gir.digraph.edge_indices() {
        let (src_idx, tgt_idx) = gir.digraph.edge_endpoints(eidx).unwrap();
        let src = &gir.digraph[src_idx].id;
        let tgt = &gir.digraph[tgt_idx].id;
        let actual_src = resolve_endpoint(src);
        let actual_tgt = resolve_endpoint(tgt);
        if actual_src == actual_tgt {
            continue;
        }
        let key = (actual_src.clone(), actual_tgt.clone());
        if added_edges.contains(&key) {
            continue;
        }
        added_edges.insert(key);
        let edge_data = gir.digraph[eidx].clone();
        new_ag.add_edge(&actual_src, &actual_tgt, Some(edge_data));
    }

    (new_ag, new_node_data, compounds)
}

pub fn compute_compound_dimensions(
    compounds: &[CompoundInfo],
    _padding: i64,
) -> HashMap<String, (i64, i64)> {
    let mut overrides: HashMap<String, (i64, i64)> = HashMap::new();
    for ci in compounds {
        let total_member_w: i64 = ci.member_widths.iter().sum();
        let gaps = if ci.member_ids.len() > 1 {
            (ci.member_ids.len() as i64 - 1) * SG_INNER_GAP
        } else {
            0
        };
        let content_w = total_member_w + gaps;
        let title_w = ci.sg_name.len() as i64 + 4;
        let desc_w = ci
            .description
            .as_ref()
            .map(|d| d.len() as i64 + 4)
            .unwrap_or(0);
        let inner_w = content_w.max(title_w).max(desc_w);
        let width = 2 + 2 * SG_PAD_X + inner_w;
        let desc_rows = if ci.description.is_some() { 1 } else { 0 };
        let height = if ci.member_ids.is_empty() {
            3 + desc_rows
        } else {
            2 + 1 + ci.max_member_height + desc_rows
        };
        overrides.insert(ci.compound_id.clone(), (width, height));
    }
    overrides
}

pub fn expand_compound_nodes(
    layout_nodes: Vec<LayoutNode>,
    compounds: &[CompoundInfo],
) -> Vec<LayoutNode> {
    let compound_map: HashMap<&str, &CompoundInfo> = compounds
        .iter()
        .map(|c| (c.compound_id.as_str(), c))
        .collect();
    let mut result: Vec<LayoutNode> = Vec::new();

    for ln in layout_nodes {
        let ci_opt = compound_map.get(ln.id.as_str()).copied();
        result.push(ln.clone());
        if let Some(ci) = ci_opt {
            let mut member_x = ln.x + 1 + SG_PAD_X;
            let member_y = ln.y + 2;
            for (i, mid) in ci.member_ids.iter().enumerate() {
                let label = ci
                    .member_labels
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| mid.clone());
                let shape = ci
                    .member_shapes
                    .get(i)
                    .cloned()
                    .unwrap_or(NodeShape::Rectangle);
                result.push(LayoutNode {
                    id: mid.clone(),
                    layer: ln.layer,
                    order: ln.order,
                    x: member_x,
                    y: member_y,
                    width: ci.member_widths.get(i).copied().unwrap_or(3),
                    height: ci.member_heights.get(i).copied().unwrap_or(NODE_HEIGHT),
                    label,
                    shape,
                });
                member_x += ci.member_widths.get(i).copied().unwrap_or(3) + SG_INNER_GAP;
            }
        }
    }

    result
}

// ─── SugiyamaLayout Engine ───────────────────────────────────────────────────

/// Sugiyama layered layout engine.
pub struct SugiyamaLayout;

impl SugiyamaLayout {
    /// Run the full Sugiyama layout pipeline on the given GraphIR.
    pub fn layout(gir: &GraphIR, padding: i64) -> LayoutResult {
        let has_subgraphs = !gir.subgraph_members.is_empty();

        if !has_subgraphs {
            let la = LayerAssignment::assign(gir);
            let (ag, node_data_map) = petgraph_to_adj(&gir.digraph);
            let (dag, _) = remove_cycles(&ag, &node_data_map);
            let dag_node_data = dag
                .nodes
                .iter()
                .map(|n| {
                    (
                        n.clone(),
                        node_data_map.get(n).cloned().unwrap_or_else(|| NodeData {
                            id: n.clone(),
                            label: n.clone(),
                            shape: NodeShape::Rectangle,
                            attrs: Vec::new(),
                            subgraph: None,
                        }),
                    )
                })
                .collect();
            let aug = insert_dummy_nodes(dag, dag_node_data, &la);
            let ordering = minimise_crossings(&aug);
            let layout_nodes = assign_coordinates_padded(
                &ordering,
                &aug,
                padding,
                &HashMap::new(),
                &gir.direction,
            );
            let routed_edges = route_edges(gir, &layout_nodes, &aug, &la.reversed_edges);

            return LayoutResult {
                nodes: layout_nodes,
                edges: routed_edges,
                direction: gir.direction.clone(),
                subgraph_members: gir.subgraph_members.clone(),
                subgraph_descriptions: gir.subgraph_descriptions.clone(),
            };
        }

        // Subgraph path
        let (collapsed_ag, collapsed_node_data, compounds) = collapse_subgraphs(gir, padding);
        let dim_overrides = compute_compound_dimensions(&compounds, padding);

        let la = LayerAssignment::assign_from_adj(&collapsed_ag, &collapsed_node_data);
        let (dag, _) = remove_cycles(&collapsed_ag, &collapsed_node_data);
        let dag_node_data: HashMap<String, NodeData> = dag
            .nodes
            .iter()
            .map(|n| {
                (
                    n.clone(),
                    collapsed_node_data
                        .get(n)
                        .cloned()
                        .unwrap_or_else(|| NodeData {
                            id: n.clone(),
                            label: n.clone(),
                            shape: NodeShape::Rectangle,
                            attrs: Vec::new(),
                            subgraph: None,
                        }),
                )
            })
            .collect();
        let aug = insert_dummy_nodes(dag, dag_node_data, &la);
        let ordering = minimise_crossings(&aug);
        let layout_nodes =
            assign_coordinates_padded(&ordering, &aug, padding, &dim_overrides, &gir.direction);
        let expanded = expand_compound_nodes(layout_nodes, &compounds);

        // Route edges using collapsed graph as source of truth
        // We need a temporary GraphIR-like structure for the collapsed graph
        // For subgraphs, route_edges expects the collapsed gir
        // Build a fake GraphIR for routing
        let collapsed_gir = build_collapsed_gir(gir, &collapsed_ag, &collapsed_node_data);
        let routed_edges = route_edges(&collapsed_gir, &expanded, &aug, &la.reversed_edges);

        LayoutResult {
            nodes: expanded,
            edges: routed_edges,
            direction: gir.direction.clone(),
            subgraph_members: gir.subgraph_members.clone(),
            subgraph_descriptions: gir.subgraph_descriptions.clone(),
        }
    }
}

/// Build a petgraph-based GraphIR from collapsed AdjGraph data (for edge routing).
fn build_collapsed_gir(
    original: &GraphIR,
    collapsed_ag: &AdjGraph,
    collapsed_node_data: &HashMap<String, NodeData>,
) -> GraphIR {
    use petgraph::graph::DiGraph as PetGraph;

    let mut digraph: PetGraph<NodeData, EdgeData> = PetGraph::new();
    let mut node_index: HashMap<String, NodeIndex> = HashMap::new();

    for node_id in &collapsed_ag.nodes {
        let data = collapsed_node_data
            .get(node_id)
            .cloned()
            .unwrap_or_else(|| NodeData {
                id: node_id.clone(),
                label: node_id.clone(),
                shape: NodeShape::Rectangle,
                attrs: Vec::new(),
                subgraph: None,
            });
        let idx = digraph.add_node(data);
        node_index.insert(node_id.clone(), idx);
    }

    for (src, tgt, edge_data_opt) in &collapsed_ag.edges {
        let src_idx = node_index[src];
        let tgt_idx = node_index[tgt];
        let edge_data = edge_data_opt.clone().unwrap_or(EdgeData {
            edge_type: EdgeType::Arrow,
            label: None,
            attrs: Vec::new(),
        });
        digraph.add_edge(src_idx, tgt_idx, edge_data);
    }

    GraphIR {
        digraph,
        direction: original.direction.clone(),
        node_index,
        subgraph_members: Vec::new(),
        subgraph_descriptions: std::collections::HashMap::new(),
    }
}

#[cfg(test)]
#[path = "../../../tests/rust/test_layout_sugiyama.rs"]
mod tests;
