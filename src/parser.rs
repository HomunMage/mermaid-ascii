use pest::iterators::Pair;
use pest::Parser as PestParser;
use pest_derive::Parser;

use crate::ast::{Attr, Direction, Edge, EdgeType, Graph, Node, NodeShape, Subgraph};

// Derive the pest parser from the grammar file.
// The `grammar` attribute path is relative to the crate root (src/).
#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct GraphParser;

// ─── Public entry point ──────────────────────────────────────────────────────

/// Parse DSL text and return a `Graph` AST, or a pest error string.
pub fn parse(input: &str) -> Result<Graph, String> {
    let pairs = GraphParser::parse(Rule::file, input)
        .map_err(|e| e.to_string())?;

    let mut graph = Graph::new();

    // The `file` rule wraps everything; iterate its inner pairs.
    for pair in pairs {
        if pair.as_rule() == Rule::file {
            for stmt_pair in pair.into_inner() {
                if stmt_pair.as_rule() == Rule::statement {
                    process_statement(stmt_pair, &mut graph);
                }
            }
        }
    }

    Ok(graph)
}

// ─── Statement dispatch ───────────────────────────────────────────────────────

fn process_statement(pair: Pair<Rule>, graph: &mut Graph) {
    // blank_line is a silent rule (_), so statement may have no inner pairs.
    let inner = match pair.into_inner().next() {
        Some(p) => p,
        None => return,
    };
    match inner.as_rule() {
        Rule::direction_decl => {
            graph.direction = parse_direction_decl(inner);
        }
        Rule::node_stmt => {
            let (node, _attrs) = parse_node_stmt(inner);
            upsert_node(graph, node);
        }
        Rule::edge_stmt => {
            let (nodes, edges) = parse_edge_stmt(inner);
            for n in nodes {
                upsert_node(graph, n);
            }
            graph.edges.extend(edges);
        }
        Rule::subgraph_block => {
            let sg = parse_subgraph_block(inner);
            graph.subgraphs.push(sg);
        }
        _ => {} // blank_line, EOI, etc.
    }
}

// Insert a node only if its id hasn't been seen before.
fn upsert_node(graph: &mut Graph, node: Node) {
    if !graph.nodes.iter().any(|n| n.id == node.id) {
        graph.nodes.push(node);
    }
}

// ─── Direction ────────────────────────────────────────────────────────────────

fn parse_direction_decl(pair: Pair<Rule>) -> Direction {
    let value = pair.into_inner().next().unwrap(); // direction_value
    match value.as_str() {
        "LR" => Direction::LR,
        "RL" => Direction::RL,
        "TD" => Direction::TD,
        "BT" => Direction::BT,
        _ => Direction::TD,
    }
}

// ─── Node statement ───────────────────────────────────────────────────────────

fn parse_node_stmt(pair: Pair<Rule>) -> (Node, Vec<Attr>) {
    let mut inner = pair.into_inner();
    let node_ref = inner.next().unwrap();
    let node = parse_node_ref(node_ref);
    let attrs = match inner.next() {
        Some(ab) if ab.as_rule() == Rule::attr_block => parse_attr_block(ab),
        _ => Vec::new(),
    };
    (node, attrs)
}

// ─── Node ref ─────────────────────────────────────────────────────────────────

fn parse_node_ref(pair: Pair<Rule>) -> Node {
    // pair.as_rule() == Rule::node_ref
    let shape_pair = pair.into_inner().next().unwrap();
    let shape = match shape_pair.as_rule() {
        Rule::rect_node => NodeShape::Rectangle,
        Rule::rounded_node => NodeShape::Rounded,
        Rule::diamond_node => NodeShape::Diamond,
        Rule::circle_node => NodeShape::Circle,
        _ => NodeShape::Rectangle,
    };
    let label_pair = shape_pair.into_inner().next().unwrap(); // node_label
    let label = parse_node_label(label_pair);
    Node::new(label, shape)
}

fn parse_node_label(pair: Pair<Rule>) -> String {
    // pair.as_rule() == Rule::node_label
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::quoted_string => parse_quoted_string(inner),
        Rule::unquoted_label => inner.as_str().trim().to_string(),
        _ => inner.as_str().to_string(),
    }
}

fn parse_quoted_string(pair: Pair<Rule>) -> String {
    // pair.as_rule() == Rule::quoted_string
    // The raw text includes surrounding quotes; strip them and unescape.
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
    let attrs: Vec<Attr>;
    let (segments, chain_attrs) = parse_edge_chain(chain_pair);
    attrs = chain_attrs;

    // Build node + edge lists from the chain.
    let mut nodes: Vec<Node> = vec![source_node.clone()];
    let mut edges: Vec<Edge> = Vec::new();

    let mut prev_id = source_node.id.clone();
    for (etype, target_node) in segments {
        let edge = Edge {
            from: prev_id.clone(),
            to: target_node.id.clone(),
            edge_type: etype,
            label: None,
            attrs: attrs.clone(),
        };
        prev_id = target_node.id.clone();
        nodes.push(target_node);
        edges.push(edge);
    }

    (nodes, edges)
}

/// Returns (Vec<(EdgeType, Node)>, attrs on the whole chain)
fn parse_edge_chain(pair: Pair<Rule>) -> (Vec<(EdgeType, Node)>, Vec<Attr>) {
    let mut segments: Vec<(EdgeType, Node)> = Vec::new();
    let mut attrs = Vec::new();

    let mut inner = pair.into_inner().peekable();
    while let Some(p) = inner.next() {
        match p.as_rule() {
            Rule::edge_connector => {
                let etype = parse_edge_connector(p);
                // next must be node_ref
                if let Some(node_p) = inner.next() {
                    let node = parse_node_ref(node_p);
                    segments.push((etype, node));
                }
            }
            Rule::attr_block => {
                attrs = parse_attr_block(p);
            }
            _ => {}
        }
    }

    (segments, attrs)
}

fn parse_edge_connector(pair: Pair<Rule>) -> EdgeType {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::arrow => EdgeType::Arrow,
        Rule::line => EdgeType::Line,
        Rule::back_arrow => EdgeType::BackArrow,
        Rule::bidir_arrow => EdgeType::BidirArrow,
        Rule::thick_arrow => EdgeType::ThickArrow,
        Rule::double_line => EdgeType::DoubleLine,
        Rule::dotted_arrow => EdgeType::DottedArrow,
        _ => EdgeType::Arrow,
    }
}

// ─── Attributes ───────────────────────────────────────────────────────────────

fn parse_attr_block(pair: Pair<Rule>) -> Vec<Attr> {
    // pair.as_rule() == Rule::attr_block
    let mut attrs = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::attr_list {
            for attr_pair in inner.into_inner() {
                if attr_pair.as_rule() == Rule::attr {
                    attrs.push(parse_attr(attr_pair));
                }
            }
        }
    }
    attrs
}

fn parse_attr(pair: Pair<Rule>) -> Attr {
    let mut inner = pair.into_inner();
    let key = inner.next().unwrap().as_str().to_string();
    let val_pair = inner.next().unwrap(); // attr_value
    let value = val_pair.into_inner().next().map(|p| match p.as_rule() {
        Rule::quoted_string => parse_quoted_string(p),
        _ => p.as_str().trim().to_string(),
    }).unwrap_or_default();
    Attr { key, value }
}

// ─── Subgraph ─────────────────────────────────────────────────────────────────

fn parse_subgraph_block(pair: Pair<Rule>) -> Subgraph {
    let mut inner = pair.into_inner();
    let name_pair = inner.next().unwrap(); // subgraph_name
    let name = parse_subgraph_name(name_pair);
    let mut sg = Subgraph::new(name);

    for stmt_pair in inner {
        if stmt_pair.as_rule() == Rule::statement {
            process_statement_into_subgraph(stmt_pair, &mut sg);
        }
    }

    sg
}

fn parse_subgraph_name(pair: Pair<Rule>) -> String {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::quoted_string => parse_quoted_string(inner),
        Rule::bare_name => inner.as_str().to_string(),
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
            let (node, _) = parse_node_stmt(inner);
            if !sg.nodes.iter().any(|n| n.id == node.id) {
                sg.nodes.push(node);
            }
        }
        Rule::edge_stmt => {
            let (nodes, edges) = parse_edge_stmt(inner);
            for n in nodes {
                if !sg.nodes.iter().any(|x| x.id == n.id) {
                    sg.nodes.push(n);
                }
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
