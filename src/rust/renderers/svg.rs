//! SVG renderer — renders Layout IR to SVG string.
//!
//! Mirrors Python's renderers/svg.py.

use crate::layout::types::{
    COMPOUND_PREFIX, DUMMY_PREFIX, LayoutNode, LayoutResult, Point, RoutedEdge,
};
use crate::renderers::Renderer;
use crate::syntax::types::{Direction, EdgeType, NodeShape};

// ─── Constants ──────────────────────────────────────────────────────────────

const CELL_W: i64 = 10;
const CELL_H: i64 = 20;
const FONT_SIZE: i64 = 14;
const FONT_FAMILY: &str = "monospace";
const PADDING: i64 = 20;

// ─── Stroke Styles ──────────────────────────────────────────────────────────

fn stroke_style(et: &EdgeType) -> &'static str {
    match et {
        EdgeType::DottedArrow | EdgeType::DottedLine | EdgeType::BidirDotted => {
            r#"stroke-dasharray="6 4""#
        }
        EdgeType::ThickArrow | EdgeType::ThickLine | EdgeType::BidirThick => r#"stroke-width="3""#,
        _ => "",
    }
}

fn is_arrow(et: &EdgeType) -> bool {
    matches!(
        et,
        EdgeType::Arrow
            | EdgeType::DottedArrow
            | EdgeType::ThickArrow
            | EdgeType::BidirArrow
            | EdgeType::BidirDotted
            | EdgeType::BidirThick
    )
}

fn is_bidir(et: &EdgeType) -> bool {
    matches!(
        et,
        EdgeType::BidirArrow | EdgeType::BidirDotted | EdgeType::BidirThick
    )
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn font(size: i64) -> String {
    format!(r#"font-family="{FONT_FAMILY}" font-size="{size}""#)
}

fn px(col: i64) -> i64 {
    PADDING + col * CELL_W
}

fn py(row: i64) -> i64 {
    PADDING + row * CELL_H
}

fn node_rect(ln: &LayoutNode) -> (i64, i64, i64, i64) {
    (px(ln.x), py(ln.y), ln.width * CELL_W, ln.height * CELL_H)
}

const FILL_STROKE: &str = r#"fill="white" stroke="black" stroke-width="1.5""#;
const SG_STROKE: &str = r##"fill="none" stroke="#888" stroke-width="1" stroke-dasharray="4 2""##;

// ─── Shape Rendering ────────────────────────────────────────────────────────

fn render_node(ln: &LayoutNode) -> String {
    let (x, y, w, h) = node_rect(ln);
    let cx = x + w / 2;
    let cy = y + h / 2;
    let label = escape(&ln.label);
    let lines: Vec<&str> = label.split('\n').collect();
    let f = font(FONT_SIZE);

    let label_svg = if lines.len() == 1 {
        format!(
            r#"<text x="{cx}" y="{cy}" dominant-baseline="central" text-anchor="middle" {f}>{}</text>"#,
            lines[0]
        )
    } else {
        let total_h = lines.len() as i64 * (FONT_SIZE + 2);
        let start_y = cy - total_h / 2 + FONT_SIZE / 2;
        let tspans: String = lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let ty = start_y + i as i64 * (FONT_SIZE + 2);
                format!(r#"<tspan x="{cx}" y="{ty}">{line}</tspan>"#)
            })
            .collect();
        format!(r#"<text text-anchor="middle" {f}>{tspans}</text>"#)
    };

    let shape_svg = match ln.shape {
        NodeShape::Rectangle => {
            format!(r#"<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="0" {FILL_STROKE}/>"#)
        }
        NodeShape::Rounded => {
            let r = w.min(h) / 4;
            format!(r#"<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" {FILL_STROKE}/>"#)
        }
        NodeShape::Diamond => {
            let pts = format!("{cx},{y} {},{cy} {cx},{} {x},{cy}", x + w, y + h);
            format!(r#"<polygon points="{pts}" {FILL_STROKE}/>"#)
        }
        NodeShape::Circle => {
            let rx = w / 2;
            let ry = h / 2;
            format!(r#"<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" {FILL_STROKE}/>"#)
        }
    };

    format!("{shape_svg}\n{label_svg}")
}

fn render_compound(ln: &LayoutNode, sg_name: &str, description: Option<&str>) -> String {
    let (x, y, w, h) = node_rect(ln);
    let f = font(FONT_SIZE - 2);
    let ty = y + FONT_SIZE + 4;
    let mut parts = vec![
        format!(r#"<rect x="{x}" y="{y}" width="{w}" height="{h}" {SG_STROKE}/>"#),
        format!(
            r##"<text x="{}" y="{ty}" {f} fill="#666">{sg_name}</text>"##,
            x + 8
        ),
    ];
    if let Some(desc) = description {
        let desc = escape(desc);
        let dy = y + h - 6;
        parts.push(format!(
            r##"<text x="{}" y="{dy}" {f} fill="#666">{desc}</text>"##,
            x + 8
        ));
    }
    parts.join("\n")
}

// ─── Edge Rendering ─────────────────────────────────────────────────────────

fn render_edge(re: &RoutedEdge) -> String {
    if re.waypoints.len() < 2 {
        return String::new();
    }

    let style = stroke_style(&re.edge_type);
    let mut markers = String::new();
    if is_arrow(&re.edge_type) {
        markers.push_str(r#" marker-end="url(#arrowhead)""#);
    }
    if is_bidir(&re.edge_type) {
        markers.push_str(r#" marker-start="url(#arrowhead-rev)""#);
    }

    let pts: String = re
        .waypoints
        .iter()
        .map(|p| format!("{},{}", px(p.x), py(p.y)))
        .collect::<Vec<_>>()
        .join(" ");

    let mut parts = vec![format!(
        r#"<polyline points="{pts}" fill="none" stroke="black" stroke-width="1.5" {style}{markers}/>"#
    )];

    if let Some(ref label) = re.label {
        let mid = re.waypoints.len() / 2;
        let lp = &re.waypoints[mid];
        let lx = px(lp.x);
        let ly = py(lp.y) - 8;
        let f = font(FONT_SIZE - 2);
        parts.push(format!(
            r##"<text x="{lx}" y="{ly}" text-anchor="middle" {f} fill="#333">{}</text>"##,
            escape(label)
        ));
    }

    parts.join("\n")
}

// ─── Subgraph Borders ───────────────────────────────────────────────────────

fn render_subgraph_borders(
    subgraph_members: &[(String, Vec<String>)],
    nodes: &[LayoutNode],
) -> String {
    let node_pos: std::collections::HashMap<&str, &LayoutNode> =
        nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut parts = Vec::new();

    for (sg_name, members) in subgraph_members {
        if members.is_empty() {
            continue;
        }

        let mut min_x = i64::MAX;
        let mut min_y = i64::MAX;
        let mut max_x = i64::MIN;
        let mut max_y = i64::MIN;

        for member_id in members {
            if let Some(ln) = node_pos.get(member_id.as_str()) {
                let (npx, npy, npw, nph) = node_rect(ln);
                min_x = min_x.min(npx);
                min_y = min_y.min(npy);
                max_x = max_x.max(npx + npw);
                max_y = max_y.max(npy + nph);
            }
        }

        if min_x == i64::MAX {
            continue;
        }

        let margin = 15i64;
        let bx = min_x - margin;
        let by = min_y - margin;
        let bw = max_x - min_x + 2 * margin;
        let bh = max_y - min_y + 2 * margin;
        let f = font(FONT_SIZE - 2);
        let ty = by + FONT_SIZE + 2;

        parts.push(format!(
            r#"<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" {SG_STROKE}/>"#
        ));
        parts.push(format!(
            r##"<text x="{}" y="{ty}" {f} fill="#666">{sg_name}</text>"##,
            bx + 8
        ));
    }

    parts.join("\n")
}

// ─── Direction Transforms ───────────────────────────────────────────────────

fn transpose_node(ln: &LayoutNode) -> LayoutNode {
    LayoutNode {
        id: ln.id.clone(),
        layer: ln.layer,
        order: ln.order,
        x: ln.y,
        y: ln.x,
        width: ln.height,
        height: ln.width,
        label: ln.label.clone(),
        shape: ln.shape.clone(),
    }
}

fn transpose_edge(re: &RoutedEdge) -> RoutedEdge {
    RoutedEdge {
        from_id: re.from_id.clone(),
        to_id: re.to_id.clone(),
        label: re.label.clone(),
        edge_type: re.edge_type.clone(),
        waypoints: re.waypoints.iter().map(|p| Point::new(p.y, p.x)).collect(),
    }
}

// ─── Public Renderer ────────────────────────────────────────────────────────

pub struct SvgRenderer;

impl Renderer for SvgRenderer {
    fn render(&self, result: &LayoutResult) -> String {
        let (nodes, edges): (Vec<LayoutNode>, Vec<RoutedEdge>) =
            if result.direction == Direction::LR || result.direction == Direction::RL {
                (
                    result.nodes.iter().map(transpose_node).collect(),
                    result.edges.iter().map(transpose_edge).collect(),
                )
            } else {
                (result.nodes.clone(), result.edges.clone())
            };

        let has_compounds = nodes.iter().any(|n| n.id.starts_with(COMPOUND_PREFIX));
        let real_nodes: Vec<&LayoutNode> = nodes
            .iter()
            .filter(|n| !n.id.starts_with(DUMMY_PREFIX) && !n.id.starts_with(COMPOUND_PREFIX))
            .collect();
        let compound_nodes: Vec<&LayoutNode> = nodes
            .iter()
            .filter(|n| n.id.starts_with(COMPOUND_PREFIX))
            .collect();

        if real_nodes.is_empty() && compound_nodes.is_empty() {
            return String::new();
        }

        // Compute canvas size
        let mut max_col: i64 = 0;
        let mut max_row: i64 = 0;
        for n in &nodes {
            if n.id.starts_with(DUMMY_PREFIX) {
                continue;
            }
            max_col = max_col.max(n.x + n.width + 2);
            max_row = max_row.max(n.y + n.height + 2);
        }
        for re in &edges {
            for p in &re.waypoints {
                max_col = max_col.max(p.x + 2);
                max_row = max_row.max(p.y + 2);
            }
        }

        let svg_w = PADDING * 2 + max_col * CELL_W;
        let svg_h = PADDING * 2 + max_row * CELL_H;

        let transform = match result.direction {
            Direction::BT => format!(r#"<g transform="translate(0,{svg_h}) scale(1,-1)">"#),
            Direction::RL => format!(r#"<g transform="translate({svg_w},0) scale(-1,1)">"#),
            _ => String::new(),
        };

        let mut parts = vec![
            format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">"#),
            "<defs>".to_string(),
            r#"  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">"#.to_string(),
            r#"    <polygon points="0 0, 10 3.5, 0 7" fill="black"/>"#.to_string(),
            "  </marker>".to_string(),
            r#"  <marker id="arrowhead-rev" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">"#.to_string(),
            r#"    <polygon points="10 0, 0 3.5, 10 7" fill="black"/>"#.to_string(),
            "  </marker>".to_string(),
            "</defs>".to_string(),
            format!(r#"<rect width="{svg_w}" height="{svg_h}" fill="white"/>"#),
        ];

        if !transform.is_empty() {
            parts.push(transform);
        }

        // Subgraph borders
        if has_compounds {
            for ln in &compound_nodes {
                let sg_name = &ln.id[COMPOUND_PREFIX.len()..];
                let desc = result
                    .subgraph_descriptions
                    .get(sg_name)
                    .map(|s| s.as_str());
                parts.push(render_compound(ln, sg_name, desc));
            }
        } else {
            let borders = render_subgraph_borders(&result.subgraph_members, &nodes);
            if !borders.is_empty() {
                parts.push(borders);
            }
        }

        // Edges (behind nodes) — sort for deterministic output across implementations
        let mut sorted_edges = edges.clone();
        sorted_edges.sort_by(|a, b| (&a.from_id, &a.to_id).cmp(&(&b.from_id, &b.to_id)));
        for re in &sorted_edges {
            parts.push(render_edge(re));
        }

        // Nodes (on top)
        for ln in &real_nodes {
            parts.push(render_node(ln));
        }

        if result.direction == Direction::BT || result.direction == Direction::RL {
            parts.push("</g>".to_string());
        }

        parts.push("</svg>".to_string());
        parts.join("\n")
    }
}
