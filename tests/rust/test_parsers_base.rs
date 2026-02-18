use super::*;

#[test]
fn test_cursor_eof() {
    let c = Cursor::new("");
    assert!(c.eof());
    let c = Cursor::new("a");
    assert!(!c.eof());
}

#[test]
fn test_cursor_peek_consume() {
    let mut c = Cursor::new("-->rest");
    assert!(c.peek("-->"));
    assert!(!c.peek("-.-"));
    assert!(c.consume("-->"));
    assert_eq!(c.pos, 3);
}

#[test]
fn test_skip_ws() {
    let mut c = Cursor::new("  \t  foo");
    c.skip_ws();
    assert_eq!(c.pos, 5);
}

#[test]
fn test_skip_ws_comment() {
    let mut c = Cursor::new("%% comment\nfoo");
    c.skip_ws();
    // should consume %% comment but stop before \n
    assert_eq!(&c.src[c.pos..].iter().collect::<String>(), "\nfoo");
}

#[test]
fn test_match_node_id() {
    let mut c = Cursor::new("my-node rest");
    let id = c.match_node_id().unwrap();
    assert_eq!(id, "my-node");
}

#[test]
fn test_match_direction() {
    let mut c = Cursor::new("TD");
    assert_eq!(c.match_direction(), Some(Direction::TD));
    let mut c = Cursor::new("LR");
    assert_eq!(c.match_direction(), Some(Direction::LR));
}

#[test]
fn test_parse_quoted_string() {
    let mut c = Cursor::new("\"Hello\\nWorld\"");
    let s = c.parse_quoted_string();
    assert_eq!(s, "Hello\nWorld");
}

#[test]
fn test_parse_node_shape_rect() {
    let mut c = Cursor::new("[Start]");
    let (shape, label) = c.parse_node_shape().unwrap();
    assert_eq!(shape, NodeShape::Rectangle);
    assert_eq!(label, "Start");
}

#[test]
fn test_parse_node_shape_circle() {
    let mut c = Cursor::new("((DB))");
    let (shape, label) = c.parse_node_shape().unwrap();
    assert_eq!(shape, NodeShape::Circle);
    assert_eq!(label, "DB");
}

#[test]
fn test_parse_node_ref_bare() {
    let mut c = Cursor::new("A");
    let node = c.parse_node_ref().unwrap();
    assert_eq!(node.id, "A");
    assert_eq!(node.label, "A");
    assert_eq!(node.shape, NodeShape::Rectangle);
}

#[test]
fn test_parse_node_ref_with_label() {
    let mut c = Cursor::new("A[Hello]");
    let node = c.parse_node_ref().unwrap();
    assert_eq!(node.id, "A");
    assert_eq!(node.label, "Hello");
    assert_eq!(node.shape, NodeShape::Rectangle);
}

#[test]
fn test_parse_edge_connector() {
    let mut c = Cursor::new("-->");
    assert_eq!(c.parse_edge_connector(), Some(EdgeType::Arrow));
    let mut c = Cursor::new("-.-");
    assert_eq!(c.parse_edge_connector(), Some(EdgeType::DottedLine));
}

#[test]
fn test_try_parse_edge_label() {
    let mut c = Cursor::new("|yes|");
    assert_eq!(c.try_parse_edge_label(), Some("yes".to_string()));
}

#[test]
fn test_upsert_node_first_wins() {
    let mut nodes = vec![Node::new("A", "First", NodeShape::Rectangle)];
    upsert_node(&mut nodes, Node::new("A", "Second", NodeShape::Rounded));
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].label, "First");
}

#[test]
fn test_at_end_keyword() {
    let c = Cursor::new("end");
    assert!(c.at_end_keyword());
    let c = Cursor::new("endgame");
    assert!(!c.at_end_keyword());
}

#[test]
fn test_try_parse_header() {
    let mut c = Cursor::new("graph TD\n");
    let dir = c.try_parse_header().unwrap();
    assert_eq!(dir, Direction::TD);

    let mut c = Cursor::new("flowchart LR\n");
    let dir = c.try_parse_header().unwrap();
    assert_eq!(dir, Direction::LR);
}

#[test]
fn test_parse_graph_simple() {
    let mut c = Cursor::new("graph TD\n    A --> B\n");
    let g = c.parse_graph();
    assert_eq!(g.direction, Direction::TD);
    assert_eq!(g.nodes.len(), 2);
    assert_eq!(g.edges.len(), 1);
    assert_eq!(g.edges[0].from_id, "A");
    assert_eq!(g.edges[0].to_id, "B");
}

#[test]
fn test_parse_graph_no_header() {
    let mut c = Cursor::new("A --> B\n");
    let g = c.parse_graph();
    assert_eq!(g.direction, Direction::TD); // default
    assert_eq!(g.nodes.len(), 2);
}

#[test]
fn test_parse_graph_edge_label() {
    let mut c = Cursor::new("graph TD\n    A -->|yes| B\n");
    let g = c.parse_graph();
    assert_eq!(g.edges[0].label, Some("yes".to_string()));
}

#[test]
fn test_parse_graph_subgraph() {
    let mut c = Cursor::new("graph TD\n    subgraph Group\n        A --> B\n    end\n");
    let g = c.parse_graph();
    assert_eq!(g.subgraphs.len(), 1);
    assert_eq!(g.subgraphs[0].name, "Group");
    assert_eq!(g.subgraphs[0].nodes.len(), 2);
}

#[test]
fn test_parse_graph_all_shapes() {
    let mut c = Cursor::new("graph TD\n    A[Rect] --> B(Round) --> C{Diamond} --> D((Circle))\n");
    let g = c.parse_graph();
    assert_eq!(g.nodes[0].shape, NodeShape::Rectangle);
    assert_eq!(g.nodes[1].shape, NodeShape::Rounded);
    assert_eq!(g.nodes[2].shape, NodeShape::Diamond);
    assert_eq!(g.nodes[3].shape, NodeShape::Circle);
}

#[test]
fn test_parse_graph_comment() {
    let mut c = Cursor::new("graph TD\n    %% This is a comment\n    A --> B\n");
    let g = c.parse_graph();
    assert_eq!(g.nodes.len(), 2);
}

#[test]
fn test_parse_edge_types() {
    let mut c = Cursor::new("graph TD\n    A --> B\n    C --- D\n    E -.-> F\n    G ==> H\n");
    let g = c.parse_graph();
    assert_eq!(g.edges[0].edge_type, EdgeType::Arrow);
    assert_eq!(g.edges[1].edge_type, EdgeType::Line);
    assert_eq!(g.edges[2].edge_type, EdgeType::DottedArrow);
    assert_eq!(g.edges[3].edge_type, EdgeType::ThickArrow);
}
