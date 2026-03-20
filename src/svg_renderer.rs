//! SVG renderer — 1:1 conversion from LayoutIR primitives to SVG.
//!
//! Each LayoutRect becomes an SVG shape, each LayoutEdge becomes a polyline.
//! No layout logic here — just drawing.

use crate::{LayoutIR, LayoutRect, LayoutEdge};

// ── Constants ────────────────────────────────────────────────────────────────

const CELL_W: i32 = 10;
const CELL_H: i32 = 20;
const FONT_SIZE: i32 = 14;
const FONT_FAMILY: &str = "monospace";
const PADDING: i32 = 20;

const FILL_STROKE: &str = r#"fill="white" stroke="black" stroke-width="1.5""#;
const SG_STROKE: &str = r##"fill="none" stroke="#888" stroke-width="1" stroke-dasharray="4 2""##;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn font(size: i32) -> String {
    format!(r#"font-family="{FONT_FAMILY}" font-size="{size}""#)
}

fn px(col: i32) -> i32 {
    PADDING + col * CELL_W
}

fn py(row: i32) -> i32 {
    PADDING + row * CELL_H
}

// ── Edge helpers ─────────────────────────────────────────────────────────────

fn stroke_style(et: &str) -> &'static str {
    match et {
        "DottedArrow" | "DottedLine" | "BidirDotted" => r#"stroke-dasharray="6 4""#,
        "ThickArrow" | "ThickLine" | "BidirThick" => r#"stroke-width="3""#,
        _ => "",
    }
}

fn is_arrow(et: &str) -> bool {
    matches!(
        et,
        "Arrow" | "DottedArrow" | "ThickArrow" | "BidirArrow" | "BidirDotted" | "BidirThick"
    )
}

fn is_bidir(et: &str) -> bool {
    matches!(et, "BidirArrow" | "BidirDotted" | "BidirThick")
}

// ── Rect → SVG ───────────────────────────────────────────────────────────────

fn render_label_svg(cx: i32, cy: i32, label: &str) -> String {
    let label_esc = escape(label);
    let lines: Vec<&str> = label_esc.split('\n').collect();
    let f = font(FONT_SIZE);

    if lines.len() == 1 {
        format!(
            r#"<text x="{cx}" y="{cy}" dominant-baseline="central" text-anchor="middle" {f}>{}</text>"#,
            lines[0]
        )
    } else {
        let total_h = lines.len() as i32 * (FONT_SIZE + 2);
        let start_y = cy - total_h / 2 + FONT_SIZE / 2;
        let tspans: String = lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let ty = start_y + i as i32 * (FONT_SIZE + 2);
                format!(r#"<tspan x="{cx}" y="{ty}">{line}</tspan>"#)
            })
            .collect();
        format!(r#"<text text-anchor="middle" {f}>{tspans}</text>"#)
    }
}

fn render_rect(r: &LayoutRect) -> String {
    let sx = px(r.x);
    let sy = py(r.y);
    let sw = r.w * CELL_W;
    let sh = r.h * CELL_H;
    let cx = sx + sw / 2;
    let cy = sy + sh / 2;

    match r.shape.as_str() {
        "Container" => {
            // Dashed border + title at top-left
            let f = font(FONT_SIZE - 2);
            let ty = sy + FONT_SIZE + 2;
            format!(
                r##"<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" {SG_STROKE}/>
<text x="{}" y="{ty}" {f} fill="#666">{}</text>"##,
                sx + 8,
                escape(&r.label)
            )
        }
        "Rounded" => {
            let rv = sw.min(sh) / 4;
            let shape_svg = format!(
                r#"<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="{rv}" {FILL_STROKE}/>"#
            );
            let label_svg = render_label_svg(cx, cy, &r.label);
            format!("{shape_svg}\n{label_svg}")
        }
        "Diamond" => {
            let pts = format!("{cx},{sy} {},{cy} {cx},{} {sx},{cy}", sx + sw, sy + sh);
            let shape_svg = format!(r#"<polygon points="{pts}" {FILL_STROKE}/>"#);
            let label_svg = render_label_svg(cx, cy, &r.label);
            format!("{shape_svg}\n{label_svg}")
        }
        "Circle" => {
            let rx = sw / 2;
            let ry = sh / 2;
            let shape_svg =
                format!(r#"<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" {FILL_STROKE}/>"#);
            let label_svg = render_label_svg(cx, cy, &r.label);
            format!("{shape_svg}\n{label_svg}")
        }
        _ => {
            // Rectangle (default)
            let shape_svg = format!(
                r#"<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="0" {FILL_STROKE}/>"#
            );
            let label_svg = render_label_svg(cx, cy, &r.label);
            format!("{shape_svg}\n{label_svg}")
        }
    }
}

// ── Edge → SVG ───────────────────────────────────────────────────────────────

fn render_edge(e: &LayoutEdge) -> String {
    if e.waypoints.len() < 2 {
        return String::new();
    }

    let style = stroke_style(&e.edge_type);
    let mut markers = String::new();
    if is_arrow(&e.edge_type) {
        markers.push_str(r#" marker-end="url(#arrowhead)""#);
    }
    if is_bidir(&e.edge_type) {
        markers.push_str(r#" marker-start="url(#arrowhead-rev)""#);
    }

    let pts: String = e
        .waypoints
        .iter()
        .map(|(x, y)| format!("{},{}", px(*x), py(*y)))
        .collect::<Vec<_>>()
        .join(" ");

    let mut parts = vec![format!(
        r#"<polyline points="{pts}" fill="none" stroke="black" stroke-width="1.5" {style}{markers}/>"#
    )];

    if !e.label.is_empty() {
        let mid = e.waypoints.len() / 2;
        let (lx, ly) = e.waypoints[mid];
        let lsx = px(lx);
        let lsy = py(ly) - 8;
        let f = font(FONT_SIZE - 2);
        parts.push(format!(
            r##"<text x="{lsx}" y="{lsy}" text-anchor="middle" {f} fill="#333">{}</text>"##,
            escape(&e.label)
        ));
    }

    parts.join("\n")
}

// ── Public API ───────────────────────────────────────────────────────────────

/// 1:1 render LayoutIR → SVG string. No layout logic, just drawing.
pub fn render_ir(ir: &LayoutIR, direction: &str) -> String {
    if ir.rects.is_empty() {
        return String::new();
    }

    // Compute canvas size from IR primitives
    let mut max_col: i32 = 0;
    let mut max_row: i32 = 0;
    for r in &ir.rects {
        max_col = max_col.max(r.x + r.w + 2);
        max_row = max_row.max(r.y + r.h + 2);
    }
    for e in &ir.edges {
        for &(wx, wy) in &e.waypoints {
            max_col = max_col.max(wx + 2);
            max_row = max_row.max(wy + 2);
        }
    }

    let svg_w = PADDING * 2 + max_col * CELL_W;
    let svg_h = PADDING * 2 + max_row * CELL_H;

    let transform = match direction {
        "BT" => format!(r#"<g transform="translate(0,{svg_h}) scale(1,-1)">"#),
        "RL" => format!(r#"<g transform="translate({svg_w},0) scale(-1,1)">"#),
        _ => String::new(),
    };

    let mut parts = vec![
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">"#
        ),
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

    // Draw containers first (behind everything)
    for r in &ir.rects {
        if r.shape == "Container" {
            parts.push(render_rect(r));
        }
    }

    // Draw edges (behind nodes)
    for e in &ir.edges {
        let svg = render_edge(e);
        if !svg.is_empty() {
            parts.push(svg);
        }
    }

    // Draw nodes on top
    for r in &ir.rects {
        if r.shape != "Container" {
            parts.push(render_rect(r));
        }
    }

    if direction == "BT" || direction == "RL" {
        parts.push("</g>".to_string());
    }

    parts.push("</svg>".to_string());
    parts.join("\n")
}
