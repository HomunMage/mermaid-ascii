// dep/layout_state.rs — Mutable state primitives for Sugiyama layout Phases 1–4.
//
// This companion file provides interior-mutable data structures needed by
// layout.hom.  The .hom codegen wraps all variable arguments with `.clone()`.
// For plain HashMap / HashSet / Vec, `.clone()` makes a deep copy — mutations
// to the copy are lost.  Using Rc<RefCell<...>> means `.clone()` is a cheap
// reference-count bump that still points to the SAME underlying data, so
// mutations are visible through all clones.
//
// IMPORTANT: uses fully-qualified `std::rc::Rc` / `std::cell::RefCell` rather
// than `use` statements to avoid E0252 "defined multiple times" when multiple
// dep .rs files are inlined together by the homunc build system.
//
// This file depends on Graph / graph_* functions from dep/graph.rs.
// When compiled as part of the mermaid-hom crate, graph.rs is always included
// first (via `use graph` in layout.hom).  No standalone unit tests are provided
// here — the algorithms are exercised through tests/test_layout.hom.
//
// ─── Exported types ─────────────────────────────────────────────────────────
//
//   DegMap      = Rc<RefCell<HashMap<String, i32>>>
//     deg_map_new()            -> DegMap
//     deg_map_set(dm, id, v)
//     deg_map_get(dm, id)      -> i32   (0 if absent)
//     deg_map_dec(dm, id)              (decrement by 1; floor at 0)
//     deg_map_max(dm)          -> i32   (max value; 0 if empty)
//     deg_map_copy(dm)         -> DegMap  (deep-copy, independent clone)
//
//   NodeSet     = Rc<RefCell<HashSet<String>>>
//     node_set_from_str_list(sl) -> NodeSet
//     node_set_remove(ns, id)
//     node_set_contains(ns, id)  -> bool
//     node_set_len(ns)            -> i32
//
//   StrList     = Rc<RefCell<Vec<String>>>
//     str_list_new()             -> StrList
//     str_list_push(sl, s)
//     str_list_len(sl)           -> i32
//     str_list_get(sl, idx)      -> String
//     str_list_extend_reversed(dst, src)  // append reversed(src) to dst
//
//   EdgePairList = Rc<RefCell<Vec<(String, String)>>>
//     edge_pair_list_new()       -> EdgePairList
//     edge_pair_list_add(epl, src, tgt)
//     edge_pair_list_contains(epl, src, tgt) -> bool
//     edge_pair_list_len(epl)    -> i32
//     edge_pair_list_get_src(epl, idx) -> String
//     edge_pair_list_get_tgt(epl, idx) -> String
//
//   EdgeInfoList = Rc<RefCell<Vec<(String, String, String, String)>>>
//     (from_id, to_id, edge_type, label)  — label="" means no label
//     edge_info_len(el)          -> i32
//     edge_info_src(el, idx)     -> String
//     edge_info_tgt(el, idx)     -> String
//     edge_info_etype(el, idx)   -> String
//     edge_info_label(el, idx)   -> String
//
//   PosMap      = Rc<RefCell<HashMap<String, i32>>>
//     pos_map_from_str_list(ordering) -> PosMap
//     pos_map_get(pm, id)        -> i32   (-1 if absent)
//
//   MutableGraph = Rc<RefCell<Graph>>
//     mgraph_new()               -> MutableGraph
//     mgraph_add_node_full(mg, id, label, shape)
//     mgraph_add_edge_full(mg, from, to, etype, label)  // label="" → None
//     mgraph_build(mg)           -> Graph
//
//   Graph wrappers (accept Graph by value — matches .hom's .clone() convention)
//     gw_node_count(g)           -> i32
//     gw_nodes(g)                -> StrList
//     gw_out_degree(g, id)       -> i32
//     gw_in_degree(g, id)        -> i32
//     gw_successors(g, id)       -> StrList
//     gw_predecessors(g, id)     -> StrList
//     gw_node_label(g, id)       -> String
//     gw_node_shape(g, id)       -> String
//     gw_copy(g)                 -> Graph
//     gw_edges_full(g)           -> EdgeInfoList
//
//   FAS helpers (encapsulate the set-membership scan)
//     fas_sinks(active, out_deg) -> StrList
//     fas_sources(active, in_deg) -> StrList
//     fas_best_node(active, out_deg, in_deg) -> String
//
//   DummyEdgeList = Rc<RefCell<Vec<DummyEdgeInfo>>>
//     (one entry per multi-layer edge that was split by insert_dummy_nodes)
//     dummy_edge_list_new()                              -> DummyEdgeList
//     dummy_edge_list_add(del, orig_src, orig_tgt, ids, etype, label)
//     dummy_edge_list_len(del)                           -> i32
//     dummy_edge_list_orig_src(del, idx)                 -> String
//     dummy_edge_list_orig_tgt(del, idx)                 -> String
//     dummy_edge_list_dummy_ids(del, idx)                -> StrList
//     dummy_edge_list_etype(del, idx)                    -> String
//     dummy_edge_list_label(del, idx)                    -> String
//
//   OrderingList = Rc<RefCell<Vec<StrList>>>
//     (Phase 4: 2D layer ordering; outer index = layer, inner StrList = node IDs)
//     ordering_new(layer_count: i32)           -> OrderingList
//     ordering_push(ol, layer_idx: i32, id)
//     ordering_layer_count(ol)                 -> i32
//     ordering_get_layer(ol, idx: i32)         -> StrList
//     ordering_set_layer(ol, idx: i32, layer)
//     ordering_count_crossings(ol, g)          -> i32
//
//   FloatMap = Rc<RefCell<HashMap<String, f32>>>
//     (Phase 4: barycenter position lookup)
//     float_map_new()                          -> FloatMap
//     float_map_from_str_list(sl)              -> FloatMap
//     float_map_get_or_inf(fm, id)             -> f32   (f32::MAX if absent)
//
//   Barycenter sort helpers (Phase 4)
//     sort_layer_by_barycenter_incoming(layer, g, neighbor_pos) -> StrList
//     sort_layer_by_barycenter_outgoing(layer, g, neighbor_pos) -> StrList
//
//   DegMap helper
//     deg_map_sorted_keys(dm)                  -> StrList   (sorted alphabetically)

use std::collections::HashSet;

// ── DegMap ────────────────────────────────────────────────────────────────────

pub type DegMap = std::rc::Rc<std::cell::RefCell<HashMap<String, i32>>>;

pub fn deg_map_new() -> DegMap {
    std::rc::Rc::new(std::cell::RefCell::new(HashMap::new()))
}

pub fn deg_map_set(dm: DegMap, id: String, val: i32) {
    dm.borrow_mut().insert(id, val);
}

pub fn deg_map_get(dm: DegMap, id: String) -> i32 {
    *dm.borrow().get(&id).unwrap_or(&0)
}

/// Decrement the degree for `id` by 1 (floor at 0).
pub fn deg_map_dec(dm: DegMap, id: String) {
    let mut map = dm.borrow_mut();
    if let Some(v) = map.get_mut(&id) {
        if *v > 0 {
            *v -= 1;
        }
    }
}

/// Return the maximum value in the map, or 0 if the map is empty.
/// Used by assign_layers to compute layer_count = max_layer + 1.
pub fn deg_map_max(dm: DegMap) -> i32 {
    dm.borrow().values().copied().max().unwrap_or(0)
}

/// Return an independent deep copy of `dm` (new Rc<RefCell<...>> with cloned data).
/// Used by insert_dummy_nodes to copy la.layers before adding dummy-node entries.
pub fn deg_map_copy(dm: DegMap) -> DegMap {
    let snapshot: HashMap<String, i32> = dm.borrow().clone();
    std::rc::Rc::new(std::cell::RefCell::new(snapshot))
}

// ── NodeSet ───────────────────────────────────────────────────────────────────

pub type NodeSet = std::rc::Rc<std::cell::RefCell<HashSet<String>>>;

/// Build a NodeSet pre-populated from all elements of a StrList.
pub fn node_set_from_str_list(
    sl: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
) -> NodeSet {
    let set: HashSet<String> = sl.borrow().iter().cloned().collect();
    std::rc::Rc::new(std::cell::RefCell::new(set))
}

pub fn node_set_remove(ns: NodeSet, id: String) {
    ns.borrow_mut().remove(&id);
}

pub fn node_set_contains(ns: NodeSet, id: String) -> bool {
    ns.borrow().contains(&id)
}

pub fn node_set_len(ns: NodeSet) -> i32 {
    ns.borrow().len() as i32
}

// ── StrList ───────────────────────────────────────────────────────────────────

pub type StrList = std::rc::Rc<std::cell::RefCell<Vec<String>>>;

pub fn str_list_new() -> StrList {
    std::rc::Rc::new(std::cell::RefCell::new(Vec::new()))
}

pub fn str_list_push(sl: StrList, s: String) {
    sl.borrow_mut().push(s);
}

pub fn str_list_len(sl: StrList) -> i32 {
    sl.borrow().len() as i32
}

pub fn str_list_get(sl: StrList, idx: i32) -> String {
    sl.borrow()[idx as usize].clone()
}

/// Append a reversed copy of `src` onto the end of `dst`.
pub fn str_list_extend_reversed(dst: StrList, src: StrList) {
    let rev: Vec<String> = src.borrow().iter().cloned().rev().collect();
    dst.borrow_mut().extend(rev);
}

// ── EdgePairList ──────────────────────────────────────────────────────────────

pub type EdgePairList = std::rc::Rc<std::cell::RefCell<Vec<(String, String)>>>;

pub fn edge_pair_list_new() -> EdgePairList {
    std::rc::Rc::new(std::cell::RefCell::new(Vec::new()))
}

pub fn edge_pair_list_add(epl: EdgePairList, src: String, tgt: String) {
    epl.borrow_mut().push((src, tgt));
}

pub fn edge_pair_list_contains(epl: EdgePairList, src: String, tgt: String) -> bool {
    epl.borrow().contains(&(src, tgt))
}

pub fn edge_pair_list_len(epl: EdgePairList) -> i32 {
    epl.borrow().len() as i32
}

pub fn edge_pair_list_get_src(epl: EdgePairList, idx: i32) -> String {
    epl.borrow()[idx as usize].0.clone()
}

pub fn edge_pair_list_get_tgt(epl: EdgePairList, idx: i32) -> String {
    epl.borrow()[idx as usize].1.clone()
}

// ── EdgeInfoList ──────────────────────────────────────────────────────────────
// Stores (from_id, to_id, edge_type, label) tuples.
// label == "" means no label (the original EdgeData.label was None).

pub type EdgeInfoList =
    std::rc::Rc<std::cell::RefCell<Vec<(String, String, String, String)>>>;

pub fn edge_info_len(el: EdgeInfoList) -> i32 {
    el.borrow().len() as i32
}

pub fn edge_info_src(el: EdgeInfoList, idx: i32) -> String {
    el.borrow()[idx as usize].0.clone()
}

pub fn edge_info_tgt(el: EdgeInfoList, idx: i32) -> String {
    el.borrow()[idx as usize].1.clone()
}

pub fn edge_info_etype(el: EdgeInfoList, idx: i32) -> String {
    el.borrow()[idx as usize].2.clone()
}

pub fn edge_info_label(el: EdgeInfoList, idx: i32) -> String {
    el.borrow()[idx as usize].3.clone()
}

// ── PosMap ────────────────────────────────────────────────────────────────────

pub type PosMap = std::rc::Rc<std::cell::RefCell<HashMap<String, i32>>>;

/// Build a PosMap from a StrList ordering: position[ordering[i]] = i.
pub fn pos_map_from_str_list(
    ordering: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
) -> PosMap {
    let map: HashMap<String, i32> = ordering
        .borrow()
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), i as i32))
        .collect();
    std::rc::Rc::new(std::cell::RefCell::new(map))
}

/// Return the position of `id` in the ordering, or -1 if absent.
pub fn pos_map_get(pm: PosMap, id: String) -> i32 {
    *pm.borrow().get(&id).unwrap_or(&-1)
}

// ── MutableGraph ──────────────────────────────────────────────────────────────
// Wraps Graph in Rc<RefCell<...>> so .hom's clone-based calling convention
// can mutate it without losing changes.
//
// Graph is defined in dep/graph.rs and is in scope when layout_state is
// inlined via `use layout_state` inside a .hom module that also has `use graph`.

pub type MutableGraph = std::rc::Rc<std::cell::RefCell<Graph>>;

pub fn mgraph_new() -> MutableGraph {
    std::rc::Rc::new(std::cell::RefCell::new(graph_new()))
}

/// Add a node to the mutable graph (no-op if already present).
pub fn mgraph_add_node_full(mg: MutableGraph, id: String, label: String, shape: String) {
    graph_add_node(&mut mg.borrow_mut(), &id, &label, &shape, None);
}

/// Add an edge to the mutable graph; label="" means no label.
pub fn mgraph_add_edge_full(
    mg: MutableGraph,
    from: String,
    to: String,
    etype: String,
    label: String,
) {
    let label_opt: Option<&str> = if label.is_empty() { None } else { Some(&label) };
    graph_add_edge(&mut mg.borrow_mut(), &from, &to, &etype, label_opt);
}

/// Extract the final Graph from a MutableGraph (clones the inner value).
pub fn mgraph_build(mg: MutableGraph) -> Graph {
    mg.borrow().clone()
}

// ── Graph wrappers (value-based) ─────────────────────────────────────────────
// These accept `Graph` by value rather than `&Graph` so that the .hom codegen's
// implicit `.clone()` on every argument produces a value of the correct type.

pub fn gw_node_count(g: Graph) -> i32 {
    graph_node_count(&g) as i32
}

pub fn gw_nodes(g: Graph) -> StrList {
    let ids = graph_nodes(&g);
    std::rc::Rc::new(std::cell::RefCell::new(ids))
}

pub fn gw_out_degree(g: Graph, id: String) -> i32 {
    graph_out_degree(&g, &id) as i32
}

pub fn gw_in_degree(g: Graph, id: String) -> i32 {
    graph_in_degree(&g, &id) as i32
}

pub fn gw_successors(g: Graph, id: String) -> StrList {
    let ids = graph_successors(&g, &id);
    std::rc::Rc::new(std::cell::RefCell::new(ids))
}

pub fn gw_predecessors(g: Graph, id: String) -> StrList {
    let ids = graph_predecessors(&g, &id);
    std::rc::Rc::new(std::cell::RefCell::new(ids))
}

pub fn gw_node_label(g: Graph, id: String) -> String {
    match g.node_index.get(&id) {
        None => id.clone(),
        Some(&idx) => g.digraph[idx].label.clone(),
    }
}

pub fn gw_node_shape(g: Graph, id: String) -> String {
    match g.node_index.get(&id) {
        None => "Rectangle".to_string(),
        Some(&idx) => g.digraph[idx].shape.clone(),
    }
}

pub fn gw_copy(g: Graph) -> Graph {
    graph_copy(&g)
}

/// Return all edges as an EdgeInfoList: (from_id, to_id, edge_type, label).
/// label="" when the original EdgeData.label was None.
pub fn gw_edges_full(g: Graph) -> EdgeInfoList {
    let mut v: Vec<(String, String, String, String)> = Vec::new();
    for eidx in g.digraph.edge_indices() {
        let (from_idx, to_idx) = g.digraph.edge_endpoints(eidx).unwrap();
        let from_id = g.digraph[from_idx].id.clone();
        let to_id = g.digraph[to_idx].id.clone();
        let data = &g.digraph[eidx];
        let etype = data.edge_type.clone();
        let label = data.label.clone().unwrap_or_default();
        v.push((from_id, to_id, etype, label));
    }
    std::rc::Rc::new(std::cell::RefCell::new(v))
}

// ── FAS helpers ───────────────────────────────────────────────────────────────

/// Return a StrList of all nodes in `active` whose out-degree is 0.
pub fn fas_sinks(active: NodeSet, out_deg: DegMap) -> StrList {
    let active_ref = active.borrow();
    let deg_ref = out_deg.borrow();
    let sinks: Vec<String> = active_ref
        .iter()
        .filter(|id| *deg_ref.get(*id).unwrap_or(&0) == 0)
        .cloned()
        .collect();
    std::rc::Rc::new(std::cell::RefCell::new(sinks))
}

/// Return a StrList of all nodes in `active` whose in-degree is 0.
pub fn fas_sources(active: NodeSet, in_deg: DegMap) -> StrList {
    let active_ref = active.borrow();
    let deg_ref = in_deg.borrow();
    let sources: Vec<String> = active_ref
        .iter()
        .filter(|id| *deg_ref.get(*id).unwrap_or(&0) == 0)
        .cloned()
        .collect();
    std::rc::Rc::new(std::cell::RefCell::new(sources))
}

/// Return the node in `active` with the highest (out_deg − in_deg) score.
/// Returns "" if active is empty.
pub fn fas_best_node(active: NodeSet, out_deg: DegMap, in_deg: DegMap) -> String {
    let active_ref = active.borrow();
    let od = out_deg.borrow();
    let id = in_deg.borrow();
    let mut best_id = String::new();
    let mut best_score = i32::MIN;
    for node_id in active_ref.iter() {
        let score = od.get(node_id).copied().unwrap_or(0)
            - id.get(node_id).copied().unwrap_or(0);
        if best_id.is_empty() || score > best_score {
            best_score = score;
            best_id = node_id.clone();
        }
    }
    best_id
}

// ── DummyEdgeList ─────────────────────────────────────────────────────────────
// Stores information about multi-layer edge replacements produced by Phase 3
// (insert_dummy_nodes).  Each entry records the original endpoints, the list
// of dummy node IDs inserted between them, and the edge metadata.
//
// The dummy_ids field captures a snapshot (Vec<String>) at add-time so that
// further mutations to the StrList passed in do not affect stored data.

pub struct DummyEdgeInfo {
    pub original_src: String,
    pub original_tgt: String,
    /// IDs of dummy nodes inserted between original_src and original_tgt,
    /// in order from src-side to tgt-side.
    pub dummy_ids: Vec<String>,
    pub edge_etype: String,
    /// "" means no label (the original edge had label = None).
    pub edge_label: String,
}

pub type DummyEdgeList =
    std::rc::Rc<std::cell::RefCell<Vec<DummyEdgeInfo>>>;

pub fn dummy_edge_list_new() -> DummyEdgeList {
    std::rc::Rc::new(std::cell::RefCell::new(Vec::new()))
}

/// Append a new DummyEdgeInfo entry.
/// `ids` is snapshot-copied at call time so the caller can reuse the StrList.
pub fn dummy_edge_list_add(
    del: DummyEdgeList,
    orig_src: String,
    orig_tgt: String,
    ids: std::rc::Rc<std::cell::RefCell<Vec<String>>>,
    etype: String,
    label: String,
) {
    let ids_snapshot: Vec<String> = ids.borrow().clone();
    del.borrow_mut().push(DummyEdgeInfo {
        original_src: orig_src,
        original_tgt: orig_tgt,
        dummy_ids: ids_snapshot,
        edge_etype: etype,
        edge_label: label,
    });
}

pub fn dummy_edge_list_len(del: DummyEdgeList) -> i32 {
    del.borrow().len() as i32
}

pub fn dummy_edge_list_orig_src(del: DummyEdgeList, idx: i32) -> String {
    del.borrow()[idx as usize].original_src.clone()
}

pub fn dummy_edge_list_orig_tgt(del: DummyEdgeList, idx: i32) -> String {
    del.borrow()[idx as usize].original_tgt.clone()
}

/// Return the dummy node IDs for the entry at `idx` as a new StrList.
pub fn dummy_edge_list_dummy_ids(del: DummyEdgeList, idx: i32) -> StrList {
    let ids = del.borrow()[idx as usize].dummy_ids.clone();
    std::rc::Rc::new(std::cell::RefCell::new(ids))
}

pub fn dummy_edge_list_etype(del: DummyEdgeList, idx: i32) -> String {
    del.borrow()[idx as usize].edge_etype.clone()
}

pub fn dummy_edge_list_label(del: DummyEdgeList, idx: i32) -> String {
    del.borrow()[idx as usize].edge_label.clone()
}

// ── OrderingList ──────────────────────────────────────────────────────────────
// Phase 4: 2D layer ordering for crossing minimisation.
//
// Outer Vec is indexed by layer (0-based).  Each element is a StrList holding
// the node IDs assigned to that layer, in the current left-to-right ordering.
//
// Using Rc<RefCell<Vec<StrList>>> (rather than Vec<Vec<String>>) means that
// ordering_get_layer() returns an Rc clone — mutations made via
// ordering_set_layer() replace the inner StrList pointer, staying visible
// through all callers.

pub type OrderingList = std::rc::Rc<std::cell::RefCell<Vec<StrList>>>;

/// Create an OrderingList with `layer_count` empty layers.
pub fn ordering_new(layer_count: i32) -> OrderingList {
    let layers: Vec<StrList> = (0..layer_count.max(0) as usize)
        .map(|_| std::rc::Rc::new(std::cell::RefCell::new(Vec::new())))
        .collect();
    std::rc::Rc::new(std::cell::RefCell::new(layers))
}

/// Append `node_id` to the layer at `layer_idx`.
pub fn ordering_push(ol: OrderingList, layer_idx: i32, node_id: String) {
    let layers = ol.borrow();
    layers[layer_idx as usize].borrow_mut().push(node_id);
}

/// Return the number of layers.
pub fn ordering_layer_count(ol: OrderingList) -> i32 {
    ol.borrow().len() as i32
}

/// Return the StrList for layer `idx` (Rc clone — shares the underlying Vec).
pub fn ordering_get_layer(ol: OrderingList, idx: i32) -> StrList {
    ol.borrow()[idx as usize].clone()
}

/// Replace the StrList for layer `idx` with `layer`.
pub fn ordering_set_layer(ol: OrderingList, idx: i32, layer: StrList) {
    ol.borrow_mut()[idx as usize] = layer;
}

/// Count edge crossings between consecutive layers.
///
/// For each pair of adjacent layers (l, l+1) finds all edges between them
/// and counts inversions — pairs of edges (ei, ej) where
/// ei.src < ej.src but ei.tgt > ej.tgt (or vice versa).
pub fn ordering_count_crossings(ol: OrderingList, g: Graph) -> i32 {
    let layers = ol.borrow();
    let layer_count = layers.len();
    let mut total: i32 = 0;

    for l_idx in 0..layer_count.saturating_sub(1) {
        // Build position map for the next layer.
        let tgt_layer = layers[l_idx + 1].borrow();
        let tgt_pos: HashMap<String, i32> = tgt_layer
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i as i32))
            .collect();

        // Collect (src_position, tgt_position) for all inter-layer edges.
        let src_layer = layers[l_idx].borrow();
        let mut edges: Vec<(i32, i32)> = Vec::new();

        for (sp, src_id) in src_layer.iter().enumerate() {
            if let Some(&src_idx) = g.node_index.get(src_id.as_str()) {
                for nb_idx in g.digraph.neighbors(src_idx) {
                    let nb_id = &g.digraph[nb_idx].id;
                    if let Some(&tp) = tgt_pos.get(nb_id.as_str()) {
                        edges.push((sp as i32, tp));
                    }
                }
            }
        }

        // Count inversions in the edge list.
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

// ── FloatMap ──────────────────────────────────────────────────────────────────
// Phase 4: f32-valued HashMap for barycenter position lookups.

pub type FloatMap = std::rc::Rc<std::cell::RefCell<HashMap<String, f32>>>;

pub fn float_map_new() -> FloatMap {
    std::rc::Rc::new(std::cell::RefCell::new(HashMap::new()))
}

/// Build a FloatMap from a StrList: position[id] = index as f32.
pub fn float_map_from_str_list(sl: StrList) -> FloatMap {
    let map: HashMap<String, f32> = sl
        .borrow()
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), i as f32))
        .collect();
    std::rc::Rc::new(std::cell::RefCell::new(map))
}

/// Return the value for `id`, or f32::MAX (used as "infinity") if absent.
pub fn float_map_get_or_inf(fm: FloatMap, id: String) -> f32 {
    *fm.borrow().get(&id).unwrap_or(&f32::MAX)
}

// ── Barycenter helpers ────────────────────────────────────────────────────────
// Internal helpers — not exposed to .hom, used by sort_layer_* functions.

/// Compute barycenter of `node_id`'s incoming neighbours' positions.
/// Returns f32::MAX if the node is absent from `g` or has no positioned
/// predecessors in `neighbor_pos`.
fn _barycenter_incoming(
    node_id: &str,
    g: &Graph,
    neighbor_pos: &HashMap<String, f32>,
) -> f32 {
    match g.node_index.get(node_id) {
        None => f32::MAX,
        Some(&idx) => {
            let positions: Vec<f32> = g
                .digraph
                .neighbors_directed(idx, petgraph::Direction::Incoming)
                .filter_map(|nb_idx| {
                    neighbor_pos
                        .get(g.digraph[nb_idx].id.as_str())
                        .copied()
                })
                .collect();
            if positions.is_empty() {
                f32::MAX
            } else {
                positions.iter().sum::<f32>() / positions.len() as f32
            }
        }
    }
}

/// Compute barycenter of `node_id`'s outgoing neighbours' positions.
/// Returns f32::MAX if the node is absent from `g` or has no positioned
/// successors in `neighbor_pos`.
fn _barycenter_outgoing(
    node_id: &str,
    g: &Graph,
    neighbor_pos: &HashMap<String, f32>,
) -> f32 {
    match g.node_index.get(node_id) {
        None => f32::MAX,
        Some(&idx) => {
            let positions: Vec<f32> = g
                .digraph
                .neighbors(idx) // outgoing by default for DiGraph
                .filter_map(|nb_idx| {
                    neighbor_pos
                        .get(g.digraph[nb_idx].id.as_str())
                        .copied()
                })
                .collect();
            if positions.is_empty() {
                f32::MAX
            } else {
                positions.iter().sum::<f32>() / positions.len() as f32
            }
        }
    }
}

/// Sort a copy of `layer` by barycenter of incoming neighbours in `neighbor_pos`.
/// Nodes with no positioned predecessors sort last (barycenter = f32::MAX).
pub fn sort_layer_by_barycenter_incoming(
    layer: StrList,
    g: Graph,
    neighbor_pos: FloatMap,
) -> StrList {
    let mut v: Vec<String> = layer.borrow().clone();
    let pos = neighbor_pos.borrow();
    v.sort_by(|a, b| {
        let fa = _barycenter_incoming(a.as_str(), &g, &pos);
        let fb = _barycenter_incoming(b.as_str(), &g, &pos);
        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
    });
    std::rc::Rc::new(std::cell::RefCell::new(v))
}

/// Sort a copy of `layer` by barycenter of outgoing neighbours in `neighbor_pos`.
/// Nodes with no positioned successors sort last (barycenter = f32::MAX).
pub fn sort_layer_by_barycenter_outgoing(
    layer: StrList,
    g: Graph,
    neighbor_pos: FloatMap,
) -> StrList {
    let mut v: Vec<String> = layer.borrow().clone();
    let pos = neighbor_pos.borrow();
    v.sort_by(|a, b| {
        let fa = _barycenter_outgoing(a.as_str(), &g, &pos);
        let fb = _barycenter_outgoing(b.as_str(), &g, &pos);
        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
    });
    std::rc::Rc::new(std::cell::RefCell::new(v))
}

// ── DegMap sorted keys ────────────────────────────────────────────────────────

/// Return all keys in `dm`, sorted alphabetically.
/// Used by minimise_crossings to produce a deterministic initial ordering.
pub fn deg_map_sorted_keys(dm: DegMap) -> StrList {
    let mut keys: Vec<String> = dm.borrow().keys().cloned().collect();
    keys.sort();
    std::rc::Rc::new(std::cell::RefCell::new(keys))
}
