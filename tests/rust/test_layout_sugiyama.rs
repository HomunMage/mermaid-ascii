use super::*;
use crate::syntax::types::{Edge, EdgeType, Graph, Node};

fn make_gir(edges: Vec<(&str, &str)>) -> GraphIR {
    let ast_edges: Vec<Edge> = edges
        .iter()
        .map(|(a, b)| Edge::new(*a, *b, EdgeType::Arrow))
        .collect();
    let g = Graph {
        direction: Direction::TD,
        nodes: Vec::new(),
        edges: ast_edges,
        subgraphs: Vec::new(),
    };
    GraphIR::from_ast(&g)
}

fn make_gir_nodes(nodes: Vec<&str>, edges: Vec<(&str, &str)>) -> GraphIR {
    let ast_nodes: Vec<Node> = nodes.iter().map(|n| Node::bare(*n)).collect();
    let ast_edges: Vec<Edge> = edges
        .iter()
        .map(|(a, b)| Edge::new(*a, *b, EdgeType::Arrow))
        .collect();
    let g = Graph {
        direction: Direction::TD,
        nodes: ast_nodes,
        edges: ast_edges,
        subgraphs: Vec::new(),
    };
    GraphIR::from_ast(&g)
}

fn make_gir_with_edge_type(edges: Vec<(&str, &str, EdgeType)>) -> GraphIR {
    use crate::syntax::types::{Edge, Graph};
    let ast_edges: Vec<Edge> = edges
        .iter()
        .map(|(a, b, t)| Edge::new(*a, *b, t.clone()))
        .collect();
    let g = Graph {
        direction: Direction::TD,
        nodes: Vec::new(),
        edges: ast_edges,
        subgraphs: Vec::new(),
    };
    GraphIR::from_ast(&g)
}

// ── Layer Assignment ─────────────────────────────────────────────────────

#[test]
fn test_layer_assignment_single_node() {
    let gir = make_gir_nodes(vec!["A"], vec![]);
    let la = LayerAssignment::assign(&gir);
    assert_eq!(la.layers["A"], 0);
    assert_eq!(la.layer_count, 1);
}

#[test]
fn test_layer_assignment_chain() {
    let gir = make_gir(vec![("A", "B"), ("B", "C")]);
    let la = LayerAssignment::assign(&gir);
    assert!(la.layers["A"] < la.layers["B"]);
    assert!(la.layers["B"] < la.layers["C"]);
    assert_eq!(la.layer_count, 3);
}

#[test]
fn test_layer_assignment_parallel() {
    // A -> C, B -> C: A and B should both be layer 0 (or same layer)
    let gir = make_gir(vec![("A", "C"), ("B", "C")]);
    let la = LayerAssignment::assign(&gir);
    assert!(la.layers["A"] < la.layers["C"]);
    assert!(la.layers["B"] < la.layers["C"]);
}

#[test]
fn test_layer_assignment_empty_graph() {
    let gir = make_gir(vec![]);
    let la = LayerAssignment::assign(&gir);
    assert_eq!(la.layer_count, 1);
}

#[test]
fn test_layer_assignment_cycle_handled() {
    // A -> B -> A (cycle): should not panic
    let gir = make_gir(vec![("A", "B"), ("B", "A")]);
    let la = LayerAssignment::assign(&gir);
    assert!(la.layer_count >= 1);
    // One edge is reversed, so one of A or B is in a different layer
    assert!(la.layers.contains_key("A"));
    assert!(la.layers.contains_key("B"));
}

// ── Dummy Node Insertion ─────────────────────────────────────────────────

#[test]
fn test_insert_dummy_no_span() {
    // A->B (adjacent layers) — no dummy nodes
    let gir = make_gir(vec![("A", "B")]);
    let la = LayerAssignment::assign(&gir);
    let (ag, nd) = petgraph_to_adj(&gir.digraph);
    let (dag, _) = remove_cycles(&ag, &nd);
    let dag_nd = dag
        .nodes
        .iter()
        .map(|n| {
            (
                n.clone(),
                nd.get(n).cloned().unwrap_or_else(|| NodeData {
                    id: n.clone(),
                    label: n.clone(),
                    shape: NodeShape::Rectangle,
                    attrs: Vec::new(),
                    subgraph: None,
                }),
            )
        })
        .collect();
    let aug = insert_dummy_nodes(dag, dag_nd, &la);
    assert!(aug.dummy_edges.is_empty());
}

#[test]
fn test_insert_dummy_span_two() {
    // A->B->C, plus A->C which spans 2 layers -> 1 dummy
    let gir = make_gir(vec![("A", "B"), ("B", "C"), ("A", "C")]);
    let la = LayerAssignment::assign(&gir);
    let (ag, nd) = petgraph_to_adj(&gir.digraph);
    let (dag, _) = remove_cycles(&ag, &nd);
    let dag_nd: HashMap<String, NodeData> = dag
        .nodes
        .iter()
        .map(|n| {
            (
                n.clone(),
                nd.get(n).cloned().unwrap_or_else(|| NodeData {
                    id: n.clone(),
                    label: n.clone(),
                    shape: NodeShape::Rectangle,
                    attrs: Vec::new(),
                    subgraph: None,
                }),
            )
        })
        .collect();
    let aug = insert_dummy_nodes(dag, dag_nd, &la);
    // The A->C edge spans 2 layers (A at 0, C at 2), so 1 dummy
    assert_eq!(aug.dummy_edges.len(), 1);
    assert_eq!(aug.dummy_edges[0].dummy_ids.len(), 1);
    let dummy_id = &aug.dummy_edges[0].dummy_ids[0];
    assert!(dummy_id.starts_with(DUMMY_PREFIX));
    assert_eq!(aug.layers[dummy_id], 1);
}

// ── Minimise crossings ────────────────────────────────────────────────────

#[test]
fn test_minimise_crossings_chain() {
    let gir = make_gir(vec![("A", "B"), ("B", "C")]);
    let la = LayerAssignment::assign(&gir);
    let (ag, nd) = petgraph_to_adj(&gir.digraph);
    let (dag, _) = remove_cycles(&ag, &nd);
    let dag_nd: HashMap<String, NodeData> = dag
        .nodes
        .iter()
        .map(|n| {
            (
                n.clone(),
                nd.get(n).cloned().unwrap_or_else(|| NodeData {
                    id: n.clone(),
                    label: n.clone(),
                    shape: NodeShape::Rectangle,
                    attrs: Vec::new(),
                    subgraph: None,
                }),
            )
        })
        .collect();
    let aug = insert_dummy_nodes(dag, dag_nd, &la);
    let ordering = minimise_crossings(&aug);
    assert_eq!(ordering.len(), 3);
    // Each layer should have exactly one node
    for layer in &ordering {
        assert_eq!(layer.len(), 1);
    }
}

// ── Coordinate assignment ─────────────────────────────────────────────────

#[test]
fn test_assign_coordinates_single_node() {
    let gir = make_gir_nodes(vec!["A"], vec![]);
    let la = LayerAssignment::assign(&gir);
    let (ag, nd) = petgraph_to_adj(&gir.digraph);
    let (dag, _) = remove_cycles(&ag, &nd);
    let dag_nd: HashMap<String, NodeData> = dag
        .nodes
        .iter()
        .map(|n| {
            (
                n.clone(),
                nd.get(n).cloned().unwrap_or_else(|| NodeData {
                    id: n.clone(),
                    label: n.clone(),
                    shape: NodeShape::Rectangle,
                    attrs: Vec::new(),
                    subgraph: None,
                }),
            )
        })
        .collect();
    let aug = insert_dummy_nodes(dag, dag_nd, &la);
    let ordering = minimise_crossings(&aug);
    let layout = assign_coordinates_padded(&ordering, &aug, 1, &HashMap::new(), &Direction::TD);
    assert_eq!(layout.len(), 1);
    assert_eq!(layout[0].id, "A");
    assert_eq!(layout[0].layer, 0);
}

#[test]
fn test_assign_coordinates_chain_y_increases() {
    let gir = make_gir(vec![("A", "B"), ("B", "C")]);
    let la = LayerAssignment::assign(&gir);
    let (ag, nd) = petgraph_to_adj(&gir.digraph);
    let (dag, _) = remove_cycles(&ag, &nd);
    let dag_nd: HashMap<String, NodeData> = dag
        .nodes
        .iter()
        .map(|n| {
            (
                n.clone(),
                nd.get(n).cloned().unwrap_or_else(|| NodeData {
                    id: n.clone(),
                    label: n.clone(),
                    shape: NodeShape::Rectangle,
                    attrs: Vec::new(),
                    subgraph: None,
                }),
            )
        })
        .collect();
    let aug = insert_dummy_nodes(dag, dag_nd, &la);
    let ordering = minimise_crossings(&aug);
    let layout = assign_coordinates_padded(&ordering, &aug, 1, &HashMap::new(), &Direction::TD);

    let node_map: HashMap<&str, &LayoutNode> = layout.iter().map(|n| (n.id.as_str(), n)).collect();
    assert!(node_map["A"].y < node_map["B"].y);
    assert!(node_map["B"].y < node_map["C"].y);
}

// ── Full layout ───────────────────────────────────────────────────────────

#[test]
fn test_full_layout_empty() {
    let gir = make_gir(vec![]);
    let result = SugiyamaLayout::layout(&gir, 1);
    assert!(result.nodes.is_empty());
    assert!(result.edges.is_empty());
}

#[test]
fn test_full_layout_single_node() {
    let gir = make_gir_nodes(vec!["A"], vec![]);
    let result = SugiyamaLayout::layout(&gir, 1);
    assert_eq!(result.nodes.len(), 1);
    assert_eq!(result.nodes[0].id, "A");
}

#[test]
fn test_full_layout_chain() {
    let gir = make_gir(vec![("A", "B"), ("B", "C")]);
    let result = SugiyamaLayout::layout(&gir, 1);
    // Should have 3 nodes (no dummies since chain is adjacent layers)
    assert_eq!(result.nodes.len(), 3);
    // Should have 2 routed edges
    assert_eq!(result.edges.len(), 2);
}

#[test]
fn test_full_layout_direction_preserved() {
    let g = Graph {
        direction: Direction::LR,
        nodes: Vec::new(),
        edges: vec![Edge::new("A", "B", EdgeType::Arrow)],
        subgraphs: Vec::new(),
    };
    let gir = GraphIR::from_ast(&g);
    let result = SugiyamaLayout::layout(&gir, 1);
    assert_eq!(result.direction, Direction::LR);
}

#[test]
fn test_full_layout_cyclic() {
    // Should not panic on cyclic input; cycle removal may insert dummy nodes
    // Python returns 4 nodes: A, B, C, plus 1 dummy for the long span.
    let gir = make_gir(vec![("A", "B"), ("B", "C"), ("C", "A")]);
    let result = SugiyamaLayout::layout(&gir, 1);
    // At least 3 real nodes (A, B, C) plus possible dummy nodes
    assert!(result.nodes.len() >= 3);
    // Real nodes must be present
    let ids: Vec<&str> = result.nodes.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"A"));
    assert!(ids.contains(&"B"));
    assert!(ids.contains(&"C"));
}

#[test]
fn test_route_edges_self_loop_skipped() {
    // Self loops should be skipped in routing
    let gir = make_gir(vec![("A", "A")]);
    let result = SugiyamaLayout::layout(&gir, 1);
    // Self loop should produce no routed edges
    assert!(result.edges.is_empty());
}

// ── route_edges tests ─────────────────────────────────────────────────────

#[test]
fn test_route_edges_one_route_per_edge() {
    // A->B->C: should produce exactly 2 routed edges
    let gir = make_gir(vec![("A", "B"), ("B", "C")]);
    let result = SugiyamaLayout::layout(&gir, 1);
    assert_eq!(result.edges.len(), 2);
}

#[test]
fn test_route_edges_from_to_ids_match() {
    // A->B: routed edge should have from_id=A, to_id=B
    let gir = make_gir(vec![("A", "B")]);
    let result = SugiyamaLayout::layout(&gir, 1);
    assert_eq!(result.edges.len(), 1);
    let edge = &result.edges[0];
    let ids: std::collections::HashSet<&str> = [edge.from_id.as_str(), edge.to_id.as_str()].into();
    assert!(ids.contains("A"));
    assert!(ids.contains("B"));
}

#[test]
fn test_route_edges_each_has_waypoints() {
    // Each routed edge must have at least 2 waypoints
    let gir = make_gir(vec![("A", "B"), ("B", "C")]);
    let result = SugiyamaLayout::layout(&gir, 1);
    for edge in &result.edges {
        assert!(
            edge.waypoints.len() >= 2,
            "Edge {}→{} has {} waypoints",
            edge.from_id,
            edge.to_id,
            edge.waypoints.len()
        );
    }
}

#[test]
fn test_route_edges_label_preserved() {
    // Edge label should be preserved in the routed edge
    use crate::syntax::types::{Edge, Graph};
    let mut e = Edge::new("A", "B", EdgeType::Arrow);
    e.label = Some("hello".to_string());
    let g = Graph {
        direction: Direction::TD,
        nodes: Vec::new(),
        edges: vec![e],
        subgraphs: Vec::new(),
    };
    let gir = GraphIR::from_ast(&g);
    let result = SugiyamaLayout::layout(&gir, 1);
    assert_eq!(result.edges.len(), 1);
    assert_eq!(result.edges[0].label, Some("hello".to_string()));
}

#[test]
fn test_route_edges_edge_type_preserved() {
    // Edge type (e.g. DottedArrow) should be preserved in the routed edge
    let gir = make_gir_with_edge_type(vec![("A", "B", EdgeType::DottedArrow)]);
    let result = SugiyamaLayout::layout(&gir, 1);
    assert_eq!(result.edges.len(), 1);
    assert_eq!(result.edges[0].edge_type, EdgeType::DottedArrow);
}

#[test]
fn test_route_edges_no_self_loops() {
    // Self-loops must be excluded from routes; other edges still routed
    let gir = make_gir(vec![("A", "B"), ("A", "A")]);
    let result = SugiyamaLayout::layout(&gir, 1);
    for edge in &result.edges {
        assert_ne!(
            edge.from_id, edge.to_id,
            "Self-loop should not appear in routes"
        );
    }
}

#[test]
fn test_route_edges_all_waypoints_non_negative() {
    // All waypoints must have non-negative coordinates
    let gir = make_gir(vec![("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]);
    let result = SugiyamaLayout::layout(&gir, 1);
    for edge in &result.edges {
        for wp in &edge.waypoints {
            assert!(
                wp.x >= 0,
                "Negative x in edge {}→{}",
                edge.from_id,
                edge.to_id
            );
            assert!(
                wp.y >= 0,
                "Negative y in edge {}→{}",
                edge.from_id,
                edge.to_id
            );
        }
    }
}

// ── full_layout() integration tests ──────────────────────────────────────

#[test]
fn test_full_layout_subgraph_includes_members() {
    use crate::syntax::types::{Node, Subgraph};
    let g = Graph {
        direction: Direction::TD,
        nodes: Vec::new(),
        edges: Vec::new(),
        subgraphs: vec![Subgraph {
            name: "sg1".to_string(),
            description: None,
            direction: None,
            nodes: vec![Node::bare("A"), Node::bare("B")],
            edges: Vec::new(),
            subgraphs: Vec::new(),
        }],
    };
    let gir = GraphIR::from_ast(&g);
    let result = SugiyamaLayout::layout(&gir, 1);
    // Result should include both member nodes A and B
    let ids: Vec<&str> = result.nodes.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&"A"), "A missing from layout: {:?}", ids);
    assert!(ids.contains(&"B"), "B missing from layout: {:?}", ids);
}

#[test]
fn test_full_layout_all_coords_non_negative() {
    // All node and waypoint coordinates must be non-negative
    let gir = make_gir(vec![("A", "B"), ("B", "C"), ("A", "C")]);
    let result = SugiyamaLayout::layout(&gir, 1);
    for n in &result.nodes {
        assert!(n.x >= 0, "Negative x for node {}", n.id);
        assert!(n.y >= 0, "Negative y for node {}", n.id);
    }
    for edge in &result.edges {
        for wp in &edge.waypoints {
            assert!(
                wp.x >= 0,
                "Negative wp.x in edge {}→{}",
                edge.from_id,
                edge.to_id
            );
            assert!(
                wp.y >= 0,
                "Negative wp.y in edge {}→{}",
                edge.from_id,
                edge.to_id
            );
        }
    }
}

#[test]
fn test_full_layout_custom_padding_wider_nodes() {
    // Larger padding should result in wider nodes than smaller padding
    let gir1 = make_gir_nodes(vec!["A"], vec![]);
    let gir2 = make_gir_nodes(vec!["A"], vec![]);
    let result1 = SugiyamaLayout::layout(&gir1, 0);
    let result2 = SugiyamaLayout::layout(&gir2, 4);
    assert!(
        result2.nodes[0].width >= result1.nodes[0].width,
        "Larger padding should produce wider nodes"
    );
}
