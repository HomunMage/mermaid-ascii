use pest::iterators::Pair;
use pest::Parser as PestParser;
use pest_derive::Parser;

use crate::ast::{Direction, Edge, EdgeType, Graph, Node, NodeShape, Subgraph};

// Derive the pest parser from the grammar file.
#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct GraphParser;

// ─── Public entry point ──────────────────────────────────────────────────────

/// Parse Mermaid flowchart text and return a `Graph` AST, or a pest error string.
pub fn parse(input: &str) -> Result<Graph, String> {
    let pairs = GraphParser::parse(Rule::file, input)
        .map_err(|e| e.to_string())?;

    let mut graph = Graph::new();

    for pair in pairs {
        if pair.as_rule() == Rule::file {
            for inner in pair.into_inner() {
                match inner.as_rule() {
                    Rule::header => {
                        graph.direction = parse_header(inner);
                    }
                    Rule::statement => {
                        process_statement(inner, &mut graph);
                    }
                    _ => {} // EOI
                }
            }
        }
    }

    Ok(graph)
}

// ─── Header ──────────────────────────────────────────────────────────────────

fn parse_header(pair: Pair<Rule>) -> Direction {
    let dir_pair = pair.into_inner().next().unwrap(); // direction_value
    parse_direction_value(dir_pair)
}

fn parse_direction_value(pair: Pair<Rule>) -> Direction {
    match pair.as_str() {
        "LR" => Direction::LR,
        "RL" => Direction::RL,
        "TD" | "TB" => Direction::TD,
        "BT" => Direction::BT,
        _ => Direction::TD,
    }
}

// ─── Statement dispatch ───────────────────────────────────────────────────────

fn process_statement(pair: Pair<Rule>, graph: &mut Graph) {
    let inner = match pair.into_inner().next() {
        Some(p) => p,
        None => return,
    };
    match inner.as_rule() {
        Rule::node_stmt => {
            let node = parse_node_stmt(inner);
            upsert_node(&mut graph.nodes, node);
        }
        Rule::edge_stmt => {
            let (nodes, edges) = parse_edge_stmt(inner);
            for n in nodes {
                upsert_node(&mut graph.nodes, n);
            }
            graph.edges.extend(edges);
        }
        Rule::subgraph_block => {
            let sg = parse_subgraph_block(inner);
            graph.subgraphs.push(sg);
        }
        _ => {} // blank_line, etc.
    }
}

/// Insert a node only if its id hasn't been seen before (first-definition-wins).
fn upsert_node(nodes: &mut Vec<Node>, node: Node) {
    if !nodes.iter().any(|n| n.id == node.id) {
        nodes.push(node);
    }
}

// ─── Node statement ───────────────────────────────────────────────────────────

fn parse_node_stmt(pair: Pair<Rule>) -> Node {
    let node_ref = pair.into_inner().next().unwrap();
    parse_node_ref(node_ref)
}

// ─── Node ref ─────────────────────────────────────────────────────────────────

fn parse_node_ref(pair: Pair<Rule>) -> Node {
    let mut inner = pair.into_inner();
    let id_pair = inner.next().unwrap(); // node_id
    let id = id_pair.as_str().to_string();

    // Check for optional shape bracket
    match inner.next() {
        Some(shape_pair) if shape_pair.as_rule() == Rule::node_shape => {
            let (shape, label) = parse_node_shape(shape_pair);
            Node::new(id, label, shape)
        }
        _ => {
            // Bare node: id = label, rectangle shape
            Node::bare(id)
        }
    }
}

fn parse_node_shape(pair: Pair<Rule>) -> (NodeShape, String) {
    let shape_pair = pair.into_inner().next().unwrap();
    let shape = match shape_pair.as_rule() {
        Rule::rect_shape => NodeShape::Rectangle,
        Rule::rounded_shape => NodeShape::Rounded,
        Rule::diamond_shape => NodeShape::Diamond,
        Rule::circle_shape => NodeShape::Circle,
        _ => NodeShape::Rectangle,
    };
    let label_pair = shape_pair.into_inner().next().unwrap(); // node_label
    let label = parse_node_label(label_pair);
    (shape, label)
}

fn parse_node_label(pair: Pair<Rule>) -> String {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::quoted_string => parse_quoted_string(inner),
        Rule::unquoted_label => inner.as_str().trim().to_string(),
        _ => inner.as_str().to_string(),
    }
}

fn parse_quoted_string(pair: Pair<Rule>) -> String {
    let raw = pair.as_str();
    let inner = &raw[1..raw.len() - 1]; // strip outer quotes
    inner.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\")
}

// ─── Edge statement ───────────────────────────────────────────────────────────

/// Returns (all referenced nodes in order, all edges in the chain).
fn parse_edge_stmt(pair: Pair<Rule>) -> (Vec<Node>, Vec<Edge>) {
    let mut inner = pair.into_inner();

    let source_ref = inner.next().unwrap(); // node_ref
    let source_node = parse_node_ref(source_ref);

    let chain_pair = inner.next().unwrap(); // edge_chain
    let segments = parse_edge_chain(chain_pair);

    let mut nodes: Vec<Node> = vec![source_node.clone()];
    let mut edges: Vec<Edge> = Vec::new();

    let mut prev_id = source_node.id.clone();
    for (etype, label, target_node) in segments {
        let mut edge = Edge::new(prev_id.clone(), target_node.id.clone(), etype);
        edge.label = label;
        prev_id = target_node.id.clone();
        nodes.push(target_node);
        edges.push(edge);
    }

    (nodes, edges)
}

/// Returns Vec<(EdgeType, Option<label>, Node)>
fn parse_edge_chain(pair: Pair<Rule>) -> Vec<(EdgeType, Option<String>, Node)> {
    let mut segments: Vec<(EdgeType, Option<String>, Node)> = Vec::new();

    let mut inner = pair.into_inner().peekable();
    while let Some(p) = inner.next() {
        match p.as_rule() {
            Rule::edge_connector => {
                let etype = parse_edge_connector(p);
                // Check for optional edge label
                let label = if let Some(next) = inner.peek() {
                    if next.as_rule() == Rule::edge_label {
                        let label_pair = inner.next().unwrap();
                        Some(parse_edge_label(label_pair))
                    } else {
                        None
                    }
                } else {
                    None
                };
                // Next must be node_ref
                if let Some(node_p) = inner.next() {
                    let node = parse_node_ref(node_p);
                    segments.push((etype, label, node));
                }
            }
            _ => {}
        }
    }

    segments
}

fn parse_edge_connector(pair: Pair<Rule>) -> EdgeType {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::arrow => EdgeType::Arrow,
        Rule::line => EdgeType::Line,
        Rule::dotted_arrow => EdgeType::DottedArrow,
        Rule::dotted_line => EdgeType::DottedLine,
        Rule::thick_arrow => EdgeType::ThickArrow,
        Rule::thick_line => EdgeType::ThickLine,
        Rule::bidir_arrow => EdgeType::BidirArrow,
        Rule::bidir_dotted => EdgeType::BidirDotted,
        Rule::bidir_thick => EdgeType::BidirThick,
        _ => EdgeType::Arrow,
    }
}

fn parse_edge_label(pair: Pair<Rule>) -> String {
    let inner = pair.into_inner().next().unwrap(); // label_text
    inner.as_str().trim().to_string()
}

// ─── Subgraph ─────────────────────────────────────────────────────────────────

fn parse_subgraph_block(pair: Pair<Rule>) -> Subgraph {
    let mut inner = pair.into_inner();
    let label_pair = inner.next().unwrap(); // subgraph_label
    let name = parse_subgraph_label(label_pair);
    let mut sg = Subgraph::new(name);

    for child in inner {
        match child.as_rule() {
            Rule::subgraph_direction => {
                let dir_pair = child.into_inner().next().unwrap();
                sg.direction = Some(parse_direction_value(dir_pair));
            }
            Rule::statement => {
                process_statement_into_subgraph(child, &mut sg);
            }
            _ => {}
        }
    }

    sg
}

fn parse_subgraph_label(pair: Pair<Rule>) -> String {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::quoted_string => parse_quoted_string(inner),
        Rule::bare_subgraph_label => inner.as_str().trim().to_string(),
        _ => inner.as_str().to_string(),
    }
}

fn process_statement_into_subgraph(pair: Pair<Rule>, sg: &mut Subgraph) {
    let inner = match pair.into_inner().next() {
        Some(p) => p,
        None => return,
    };
    match inner.as_rule() {
        Rule::node_stmt => {
            let node = parse_node_stmt(inner);
            upsert_node(&mut sg.nodes, node);
        }
        Rule::edge_stmt => {
            let (nodes, edges) = parse_edge_stmt(inner);
            for n in nodes {
                upsert_node(&mut sg.nodes, n);
            }
            sg.edges.extend(edges);
        }
        Rule::subgraph_block => {
            let nested = parse_subgraph_block(inner);
            sg.subgraphs.push(nested);
        }
        _ => {}
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_chain() {
        let input = "graph TD\n    A --> B --> C\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.direction, Direction::TD);
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.nodes[0].id, "A");
        assert_eq!(graph.nodes[0].label, "A");
        assert_eq!(graph.edges[0].from, "A");
        assert_eq!(graph.edges[0].to, "B");
    }

    #[test]
    fn test_parse_node_with_label() {
        let input = "graph TD\n    A[Start] --> B[End]\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.nodes[0].id, "A");
        assert_eq!(graph.nodes[0].label, "Start");
        assert_eq!(graph.nodes[0].shape, NodeShape::Rectangle);
        assert_eq!(graph.nodes[1].id, "B");
        assert_eq!(graph.nodes[1].label, "End");
    }

    #[test]
    fn test_parse_shapes() {
        let input = "graph TD\n    A[Rect] --> B(Round) --> C{Diamond} --> D((Circle))\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.nodes[0].shape, NodeShape::Rectangle);
        assert_eq!(graph.nodes[1].shape, NodeShape::Rounded);
        assert_eq!(graph.nodes[2].shape, NodeShape::Diamond);
        assert_eq!(graph.nodes[3].shape, NodeShape::Circle);
    }

    #[test]
    fn test_parse_edge_label() {
        let input = "graph TD\n    A -->|yes| B\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.edges[0].label, Some("yes".to_string()));
    }

    #[test]
    fn test_parse_flowchart_keyword() {
        let input = "flowchart LR\n    A --> B\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.direction, Direction::LR);
    }

    #[test]
    fn test_parse_subgraph() {
        let input = "graph TD\n    subgraph Group\n        A --> B\n    end\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.subgraphs.len(), 1);
        assert_eq!(graph.subgraphs[0].name, "Group");
        assert_eq!(graph.subgraphs[0].nodes.len(), 2);
        assert_eq!(graph.subgraphs[0].edges.len(), 1);
    }

    #[test]
    fn test_first_definition_wins() {
        let input = "graph TD\n    A[Hello] --> B\n    A[World] --> C\n";
        let graph = parse(input).unwrap();
        // First definition wins: A keeps label "Hello"
        assert_eq!(graph.nodes.iter().find(|n| n.id == "A").unwrap().label, "Hello");
    }

    #[test]
    fn test_parse_no_header() {
        let input = "A --> B\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.direction, Direction::TD); // default
        assert_eq!(graph.nodes.len(), 2);
    }

    #[test]
    fn test_parse_comments() {
        let input = "graph TD\n    %% This is a comment\n    A --> B\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.nodes.len(), 2);
    }

    #[test]
    fn test_parse_edge_types() {
        let input = "graph TD\n    A --> B\n    C --- D\n    E -.-> F\n    G ==> H\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.edges[0].edge_type, EdgeType::Arrow);
        assert_eq!(graph.edges[1].edge_type, EdgeType::Line);
        assert_eq!(graph.edges[2].edge_type, EdgeType::DottedArrow);
        assert_eq!(graph.edges[3].edge_type, EdgeType::ThickArrow);
    }

    #[test]
    fn test_parse_quoted_label() {
        let input = "graph TD\n    A[\"Hello World\"] --> B\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.nodes[0].label, "Hello World");
    }

    #[test]
    fn test_parse_multiline_label() {
        let input = "graph TD\n    A[\"Line1\\nLine2\"] --> B\n";
        let graph = parse(input).unwrap();
        assert_eq!(graph.nodes[0].label, "Line1\nLine2");
    }
}
