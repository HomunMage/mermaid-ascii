/// Render module — Phase 5 of the pipeline.
///
/// Paints layout data (positioned nodes + routed edges) onto a 2D character
/// grid and converts it to a printable string.
///
/// ## Layer order (paint last wins):
///   1. Subgraph borders
///   2. Node boxes
///   3. Edge lines (horizontal and vertical segments)
///   4. Edge corners / junctions (merged using Unicode box-drawing rules)
///   5. Arrowheads
///   6. Edge labels
///
/// ## Character sets
///
/// The renderer supports two character sets:
/// - Unicode box-drawing (default): `┌ ┐ └ ┘ ─ │ ├ ┤ ┬ ┴ ┼ ► ▼ ◄ ▲`
/// - ASCII fallback:                `+ + + + - | + + + + + > v < ^`

use crate::ast::{EdgeType, NodeShape};
use crate::graph::GraphIR;
use crate::layout::{LayoutNode, RoutedEdge, DUMMY_PREFIX};

// ─── Geometry Types ───────────────────────────────────────────────────────────

/// A rectangle in character coordinates (top-left origin).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rect {
    /// Column of the left edge.
    pub x: usize,
    /// Row of the top edge.
    pub y: usize,
    /// Width in characters.
    pub width: usize,
    /// Height in characters.
    pub height: usize,
}

impl Rect {
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Rect { x, y, width, height }
    }

    /// Right edge column (exclusive).
    pub fn right(&self) -> usize {
        self.x + self.width
    }

    /// Bottom row (exclusive).
    pub fn bottom(&self) -> usize {
        self.y + self.height
    }
}

// ─── Character Set ────────────────────────────────────────────────────────────

/// Which character set to use for box-drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharSet {
    /// Unicode box-drawing characters (default).
    Unicode,
    /// ASCII-safe fallback (`+`, `-`, `|`).
    Ascii,
}

/// All the characters needed to draw boxes and edges.
pub struct BoxChars {
    pub top_left:     char, // ┌  or  +
    pub top_right:    char, // ┐  or  +
    pub bottom_left:  char, // └  or  +
    pub bottom_right: char, // ┘  or  +
    pub horizontal:   char, // ─  or  -
    pub vertical:     char, // │  or  |
    pub tee_right:    char, // ├  or  +
    pub tee_left:     char, // ┤  or  +
    pub tee_down:     char, // ┬  or  +
    pub tee_up:       char, // ┴  or  +
    pub cross:        char, // ┼  or  +
    pub arrow_right:  char, // ►  or  >
    pub arrow_left:   char, // ◄  or  <
    pub arrow_down:   char, // ▼  or  v
    pub arrow_up:     char, // ▲  or  ^
}

impl BoxChars {
    pub fn unicode() -> Self {
        BoxChars {
            top_left:     '┌',
            top_right:    '┐',
            bottom_left:  '└',
            bottom_right: '┘',
            horizontal:   '─',
            vertical:     '│',
            tee_right:    '├',
            tee_left:     '┤',
            tee_down:     '┬',
            tee_up:       '┴',
            cross:        '┼',
            arrow_right:  '►',
            arrow_left:   '◄',
            arrow_down:   '▼',
            arrow_up:     '▲',
        }
    }

    pub fn ascii() -> Self {
        BoxChars {
            top_left:     '+',
            top_right:    '+',
            bottom_left:  '+',
            bottom_right: '+',
            horizontal:   '-',
            vertical:     '|',
            tee_right:    '+',
            tee_left:     '+',
            tee_down:     '+',
            tee_up:       '+',
            cross:        '+',
            arrow_right:  '>',
            arrow_left:   '<',
            arrow_down:   'v',
            arrow_up:     '^',
        }
    }

    pub fn for_charset(cs: CharSet) -> Self {
        match cs {
            CharSet::Unicode => Self::unicode(),
            CharSet::Ascii   => Self::ascii(),
        }
    }
}

// ─── Junction Merging ─────────────────────────────────────────────────────────

/// Describes which of the four arms of a cell are "active" (connected).
///
/// Used when deciding what junction character to place at a cell where two
/// edges meet.  For example, if a vertical edge crosses a horizontal edge we
/// get `{up, down, left, right}` → `┼`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Arms {
    pub up:    bool,
    pub down:  bool,
    pub left:  bool,
    pub right: bool,
}

impl Arms {
    /// Compute the `Arms` implied by an existing box-drawing character.
    ///
    /// Returns `None` if the character is not a recognised box-drawing char
    /// (e.g. a letter inside a node label).
    pub fn from_char(c: char) -> Option<Arms> {
        let (u, d, l, r) = match c {
            '─' => (false, false, true,  true),
            '│' => (true,  true,  false, false),
            '┌' => (false, true,  false, true),
            '┐' => (false, true,  true,  false),
            '└' => (true,  false, false, true),
            '┘' => (true,  false, true,  false),
            '├' => (true,  true,  false, true),
            '┤' => (true,  true,  true,  false),
            '┬' => (false, true,  true,  true),
            '┴' => (true,  false, true,  true),
            '┼' => (true,  true,  true,  true),
            // ASCII equivalents — map to same topology.
            '-' => (false, false, true,  true),
            '|' => (true,  true,  false, false),
            '+' => (true,  true,  true,  true),
            _ => return None,
        };
        Some(Arms { up: u, down: d, left: l, right: r })
    }

    /// Merge two `Arms` by OR-ing their bits.
    pub fn merge(self, other: Arms) -> Arms {
        Arms {
            up:    self.up    || other.up,
            down:  self.down  || other.down,
            left:  self.left  || other.left,
            right: self.right || other.right,
        }
    }

    /// Convert the combined arms back to a Unicode box-drawing character.
    ///
    /// Falls back to `+` for ASCII mode when no exact match exists (the caller
    /// passes `ascii_mode`).  Returns `' '` if no arms are active.
    pub fn to_char(self, cs: CharSet) -> char {
        let bc = BoxChars::for_charset(cs);
        match (self.up, self.down, self.left, self.right) {
            (false, false, false, false) => ' ',
            // Straight lines.
            (false, false, true,  true)  => bc.horizontal,
            (true,  true,  false, false) => bc.vertical,
            // Corners.
            (false, true,  false, true)  => bc.top_left,
            (false, true,  true,  false) => bc.top_right,
            (true,  false, false, true)  => bc.bottom_left,
            (true,  false, true,  false) => bc.bottom_right,
            // Tees.
            (true,  true,  false, true)  => bc.tee_right,
            (true,  true,  true,  false) => bc.tee_left,
            (false, true,  true,  true)  => bc.tee_down,
            (true,  false, true,  true)  => bc.tee_up,
            // Full cross.
            (true,  true,  true,  true)  => bc.cross,
            // Partial / single arm — treat as the nearest line or a corner.
            (true,  false, false, false) => bc.vertical,
            (false, true,  false, false) => bc.vertical,
            (false, false, true,  false) => bc.horizontal,
            (false, false, false, true)  => bc.horizontal,
        }
    }
}

// ─── Canvas ───────────────────────────────────────────────────────────────────

/// A 2D character grid onto which graph elements are painted.
///
/// The canvas uses a column-major layout:  `cells[row][col]`.
/// All coordinates are in character units (column = x, row = y).
pub struct Canvas {
    /// Width in characters.
    pub width: usize,
    /// Height in characters.
    pub height: usize,
    /// The grid: `cells[row][col]`.
    cells: Vec<Vec<char>>,
    /// Which character set to use when merging junction characters.
    pub charset: CharSet,
}

impl Canvas {
    /// Create a new blank canvas filled with spaces.
    pub fn new(width: usize, height: usize, charset: CharSet) -> Self {
        Canvas {
            width,
            height,
            cells: vec![vec![' '; width]; height],
            charset,
        }
    }

    /// Read the character at `(col, row)`.  Returns `' '` if out of bounds.
    pub fn get(&self, col: usize, row: usize) -> char {
        self.cells.get(row).and_then(|r| r.get(col)).copied().unwrap_or(' ')
    }

    /// Write a character at `(col, row)`, ignoring out-of-bounds writes.
    pub fn set(&mut self, col: usize, row: usize, c: char) {
        if row < self.height && col < self.width {
            self.cells[row][col] = c;
        }
    }

    /// Write a character at `(col, row)` using junction merging:
    ///
    /// If the current cell already contains a recognised box-drawing character,
    /// the new character is merged with it so that all active arms are preserved.
    /// For example, painting `─` over `│` yields `┼`.
    ///
    /// Falls back to simple overwrite when either character is not a
    /// box-drawing character (e.g. writing a letter over a space).
    pub fn set_merge(&mut self, col: usize, row: usize, c: char) {
        if row >= self.height || col >= self.width {
            return;
        }
        let existing = self.cells[row][col];
        // Try to merge arms.
        if let (Some(ea), Some(na)) = (Arms::from_char(existing), Arms::from_char(c)) {
            let merged = ea.merge(na);
            self.cells[row][col] = merged.to_char(self.charset);
        } else {
            // Non-box-drawing character (e.g. label letter) — just overwrite.
            self.cells[row][col] = c;
        }
    }

    // ─── Primitive drawing operations ────────────────────────────────────────

    /// Draw a horizontal line of `c` from column `x1` to `x2` (inclusive)
    /// at row `y`.  Uses merging so existing junctions are preserved.
    pub fn hline(&mut self, y: usize, x1: usize, x2: usize, c: char) {
        let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
        for col in lo..=hi {
            self.set_merge(col, y, c);
        }
    }

    /// Draw a vertical line of `c` from row `y1` to `y2` (inclusive)
    /// at column `x`.  Uses merging so existing junctions are preserved.
    pub fn vline(&mut self, x: usize, y1: usize, y2: usize, c: char) {
        let (lo, hi) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
        for row in lo..=hi {
            self.set_merge(x, row, c);
        }
    }

    /// Draw a box outline described by `rect`, using `bc` box characters.
    ///
    /// The box consists of:
    ///   - Top-left / top-right / bottom-left / bottom-right corner characters
    ///   - Horizontal lines on the top and bottom rows
    ///   - Vertical lines on the left and right columns
    pub fn draw_box(&mut self, rect: &Rect, bc: &BoxChars) {
        if rect.width < 2 || rect.height < 2 {
            return; // Too small to draw a box.
        }
        let x0 = rect.x;
        let y0 = rect.y;
        let x1 = rect.x + rect.width - 1;  // right column
        let y1 = rect.y + rect.height - 1; // bottom row

        // Corners.
        self.set(x0, y0, bc.top_left);
        self.set(x1, y0, bc.top_right);
        self.set(x0, y1, bc.bottom_left);
        self.set(x1, y1, bc.bottom_right);

        // Top and bottom horizontal edges (inside the corners).
        for col in (x0 + 1)..x1 {
            self.set(col, y0, bc.horizontal);
            self.set(col, y1, bc.horizontal);
        }

        // Left and right vertical edges (inside the corners).
        for row in (y0 + 1)..y1 {
            self.set(x0, row, bc.vertical);
            self.set(x1, row, bc.vertical);
        }
    }

    /// Write a string starting at `(col, row)`.  Clips at canvas boundary.
    pub fn write_str(&mut self, col: usize, row: usize, s: &str) {
        for (i, ch) in s.chars().enumerate() {
            let c = col + i;
            if c >= self.width || row >= self.height {
                break;
            }
            self.cells[row][c] = ch;
        }
    }

    // ─── Render to string ─────────────────────────────────────────────────────

    /// Convert the canvas to a printable string.
    ///
    /// Each row becomes one line.  Trailing spaces on each line are stripped.
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        for row in &self.cells {
            let line: String = row.iter().collect();
            out.push_str(line.trim_end());
            out.push('\n');
        }
        // Strip trailing blank lines.
        let trimmed = out.trim_end_matches('\n');
        format!("{}\n", trimmed)
    }
}

// ─── Node Rendering ───────────────────────────────────────────────────────────

/// Box characters for each node shape.
fn box_chars_for_shape(shape: &NodeShape, cs: CharSet) -> BoxChars {
    match shape {
        NodeShape::Rectangle => BoxChars::for_charset(cs),
        NodeShape::Rounded => {
            // Rounded corners: ╭ ╮ ╰ ╯
            let mut bc = BoxChars::unicode();
            if cs == CharSet::Ascii {
                return BoxChars::ascii();
            }
            bc.top_left = '╭';
            bc.top_right = '╮';
            bc.bottom_left = '╰';
            bc.bottom_right = '╯';
            bc
        }
        NodeShape::Diamond => {
            // Diamond uses / \ characters for corners
            let mut bc = BoxChars::for_charset(cs);
            bc.top_left = '/';
            bc.top_right = '\\';
            bc.bottom_left = '\\';
            bc.bottom_right = '/';
            bc
        }
        NodeShape::Circle => {
            // Circle uses ( ) for left/right borders
            let mut bc = BoxChars::for_charset(cs);
            bc.top_left = '(';
            bc.top_right = ')';
            bc.bottom_left = '(';
            bc.bottom_right = ')';
            bc.vertical = ' '; // no side bars — parens serve as borders
            bc
        }
    }
}

/// Paint a single node box with its label onto the canvas.
///
/// Layout (for Rectangle, height=3):
/// ```text
///   ┌─────────┐
///   │  Label  │
///   └─────────┘
/// ```
fn paint_node(canvas: &mut Canvas, ln: &LayoutNode, shape: &NodeShape, label: &str) {
    let x = ln.x;
    let y = ln.y;
    let w = ln.width;
    let h = ln.height;

    let bc = box_chars_for_shape(shape, canvas.charset);

    // Draw the outer box border.
    let rect = Rect::new(x, y, w, h);
    canvas.draw_box(&rect, &bc);

    let inner_w = w.saturating_sub(2);
    let lines: Vec<&str> = label.split('\n').collect();

    for (i, line) in lines.iter().enumerate() {
        let label_row = y + 1 + i; // row after top border
        let line_len = line.len();
        let pad = inner_w.saturating_sub(line_len) / 2;
        let col_start = x + 1 + pad;
        canvas.write_str(col_start, label_row, line);
    }
}

// ─── Subgraph Border Rendering ────────────────────────────────────────────────

/// Paint a dashed bounding box around the nodes in each subgraph.
///
/// The border is placed 2 columns / 1 row outside the member nodes' bounding box.
/// The subgraph name is embedded in the top border row.
fn paint_subgraph_borders(
    gir: &GraphIR,
    layout_nodes: &[LayoutNode],
    canvas: &mut Canvas,
) {
    use std::collections::HashMap;
    let node_pos: HashMap<&str, &LayoutNode> =
        layout_nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    let bc = BoxChars::for_charset(canvas.charset);

    for (sg_name, members) in &gir.subgraph_members {
        if members.is_empty() {
            continue;
        }

        // Compute bounding box of member nodes.
        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0usize;
        let mut max_y = 0usize;

        for member_id in members {
            if let Some(ln) = node_pos.get(member_id.as_str()) {
                if ln.x < min_x { min_x = ln.x; }
                if ln.y < min_y { min_y = ln.y; }
                let right = ln.x + ln.width;
                let bottom = ln.y + ln.height;
                if right > max_x { max_x = right; }
                if bottom > max_y { max_y = bottom; }
            }
        }

        if min_x == usize::MAX {
            continue; // no positioned members
        }

        // Expand bounding box by margin.
        let margin_x = 2;
        let margin_y = 1;
        let bx = min_x.saturating_sub(margin_x);
        let by = min_y.saturating_sub(margin_y);
        let bw = (max_x + margin_x).saturating_sub(bx);
        let bh = (max_y + margin_y).saturating_sub(by);

        // Draw the subgraph border box.
        let rect = Rect::new(bx, by, bw, bh);
        canvas.draw_box(&rect, &bc);

        // Embed subgraph name in top border (overwrite some '─' chars).
        let label = format!(" {} ", sg_name);
        let label_col = bx + 2;
        if label.len() + 4 <= bw {
            canvas.write_str(label_col, by, &label);
        }
    }
}

// ─── Edge Rendering ───────────────────────────────────────────────────────────

/// Determine the EdgeType for a visual edge from `from_id` to `to_id`.
///
/// Searches the original GraphIR for a matching edge (in either direction,
/// since back-edges are flipped for display). Falls back to `Arrow`.
fn edge_type_for(gir: &GraphIR, from_id: &str, to_id: &str) -> EdgeType {
    use petgraph::visit::EdgeRef;

    // Check forward direction.
    if let Some(&from_idx) = gir.node_index.get(from_id) {
        for e in gir.digraph.edges(from_idx) {
            if gir.digraph[e.target()].id == to_id {
                return e.weight().edge_type.clone();
            }
        }
    }
    // Check reverse direction (for reversed back-edges displayed as forward).
    if let Some(&to_idx) = gir.node_index.get(to_id) {
        for e in gir.digraph.edges(to_idx) {
            if gir.digraph[e.target()].id == from_id {
                return e.weight().edge_type.clone();
            }
        }
    }
    EdgeType::Arrow
}

/// Select horizontal and vertical line characters for an edge type.
fn line_chars_for(edge_type: &EdgeType, cs: CharSet) -> (char, char) {
    let bc = BoxChars::for_charset(cs);
    match edge_type {
        EdgeType::ThickArrow | EdgeType::DoubleLine => ('═', '║'),
        EdgeType::DottedArrow => ('╌', '╎'),
        _ => (bc.horizontal, bc.vertical),
    }
}

/// Paint a single routed edge: line segments + arrowhead + optional label.
fn paint_edge(canvas: &mut Canvas, re: &RoutedEdge, edge_type: &EdgeType) {
    if re.waypoints.len() < 2 {
        return;
    }

    let cs = canvas.charset;
    let (h_ch, v_ch) = line_chars_for(edge_type, cs);
    let bc = BoxChars::for_charset(cs);

    // Draw each segment between consecutive waypoints.
    for i in 0..re.waypoints.len() - 1 {
        let p0 = &re.waypoints[i];
        let p1 = &re.waypoints[i + 1];

        if p0.y == p1.y {
            // Horizontal segment.
            canvas.hline(p0.y, p0.x, p1.x, h_ch);
        } else if p0.x == p1.x {
            // Vertical segment.
            canvas.vline(p0.x, p0.y, p1.y, v_ch);
        }
        // Diagonal segments not supported (orthogonal routing only).
    }

    // Arrowhead at the last waypoint (entry point into target node).
    let last = re.waypoints.last().unwrap();
    let prev = &re.waypoints[re.waypoints.len() - 2];

    // Undirected edges (Line) get no arrowhead.
    if edge_type != &EdgeType::Line {
        let arrow = if last.y < prev.y {
            bc.arrow_up
        } else if last.y > prev.y {
            bc.arrow_down
        } else if last.x > prev.x {
            bc.arrow_right
        } else {
            bc.arrow_left
        };
        canvas.set(last.x, last.y, arrow);
    }

    // Bidirectional edges get an arrowhead at the start too.
    if edge_type == &EdgeType::BidirArrow {
        let first = &re.waypoints[0];
        let second = &re.waypoints[1];
        let start_arrow = if first.y < second.y {
            bc.arrow_up
        } else if first.y > second.y {
            bc.arrow_down
        } else if first.x > second.x {
            bc.arrow_right
        } else {
            bc.arrow_left
        };
        canvas.set(first.x, first.y, start_arrow);
    }

    // Edge label: placed at the midpoint waypoint, one row above the line.
    if let Some(label) = &re.label {
        let mid = re.waypoints.len() / 2;
        let lp = &re.waypoints[mid];
        let label_y = lp.y.saturating_sub(1);
        canvas.write_str(lp.x, label_y, label);
    }
}

// ─── Public Render Entry Point ────────────────────────────────────────────────

/// Compute the canvas dimensions needed to fit all nodes and edge waypoints.
pub fn canvas_dimensions(layout_nodes: &[LayoutNode], routed_edges: &[RoutedEdge]) -> (usize, usize) {
    let mut max_col = 40usize;
    let mut max_row = 10usize;

    for n in layout_nodes {
        if n.id.starts_with(DUMMY_PREFIX) {
            continue;
        }
        max_col = max_col.max(n.x + n.width + 2);
        max_row = max_row.max(n.y + n.height + 4);
    }
    for re in routed_edges {
        for p in &re.waypoints {
            max_col = max_col.max(p.x + 4);
            max_row = max_row.max(p.y + 4);
        }
    }

    (max_col, max_row)
}

/// Render a fully-laid-out graph to a multi-line String.
///
/// # Arguments
/// * `gir`          — The graph IR (provides node shapes, subgraph membership, edge types).
/// * `layout_nodes` — Positioned nodes from the layout phase (may include dummy nodes).
/// * `routed_edges` — Routed edges with waypoints from the edge routing phase.
/// * `unicode`      — `true` for Unicode box-drawing; `false` for ASCII fallback.
pub fn render(
    gir: &GraphIR,
    layout_nodes: &[LayoutNode],
    routed_edges: &[RoutedEdge],
    unicode: bool,
) -> String {
    use std::collections::HashMap;

    let cs = if unicode { CharSet::Unicode } else { CharSet::Ascii };

    // Filter out dummy nodes — they are layout artefacts, not rendered.
    let real_nodes: Vec<&LayoutNode> = layout_nodes
        .iter()
        .filter(|n| !n.id.starts_with(DUMMY_PREFIX))
        .collect();

    if real_nodes.is_empty() {
        return String::new();
    }

    let (width, height) = canvas_dimensions(layout_nodes, routed_edges);
    let mut canvas = Canvas::new(width, height, cs);

    // Build id → NodeData for shape / label lookup.
    let node_data_map: HashMap<&str, _> = gir
        .digraph
        .node_indices()
        .map(|ni| (gir.digraph[ni].id.as_str(), &gir.digraph[ni]))
        .collect();

    // 1. Subgraph borders (drawn first so node boxes paint over them).
    paint_subgraph_borders(gir, layout_nodes, &mut canvas);

    // 2. Node boxes + labels.
    for ln in &real_nodes {
        let shape = node_data_map
            .get(ln.id.as_str())
            .map(|d| &d.shape)
            .unwrap_or(&NodeShape::Rectangle);
        let label = node_data_map
            .get(ln.id.as_str())
            .map(|d| d.label.as_str())
            .unwrap_or(ln.id.as_str());
        paint_node(&mut canvas, ln, shape, label);
    }

    // 3–5. Edges: line segments, arrowheads, labels.
    for re in routed_edges {
        let edge_type = edge_type_for(gir, &re.from_id, &re.to_id);
        paint_edge(&mut canvas, re, &edge_type);
    }

    canvas.to_string()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arms_from_char_horizontal() {
        let a = Arms::from_char('─').unwrap();
        assert!(!a.up && !a.down && a.left && a.right);
    }

    #[test]
    fn test_arms_from_char_vertical() {
        let a = Arms::from_char('│').unwrap();
        assert!(a.up && a.down && !a.left && !a.right);
    }

    #[test]
    fn test_arms_merge_cross() {
        let horiz = Arms::from_char('─').unwrap();
        let vert  = Arms::from_char('│').unwrap();
        let merged = horiz.merge(vert);
        assert_eq!(merged.to_char(CharSet::Unicode), '┼');
    }

    #[test]
    fn test_arms_merge_tee_right() {
        // Vertical + right arm → tee pointing right (├)
        let vert  = Arms { up: true, down: true, left: false, right: false };
        let right = Arms { up: false, down: false, left: false, right: true };
        let merged = vert.merge(right);
        assert_eq!(merged.to_char(CharSet::Unicode), '├');
    }

    #[test]
    fn test_arms_to_char_ascii_cross() {
        let cross = Arms { up: true, down: true, left: true, right: true };
        assert_eq!(cross.to_char(CharSet::Ascii), '+');
    }

    #[test]
    fn test_canvas_set_get() {
        let mut canvas = Canvas::new(10, 5, CharSet::Unicode);
        canvas.set(3, 2, 'X');
        assert_eq!(canvas.get(3, 2), 'X');
        assert_eq!(canvas.get(0, 0), ' ');
    }

    #[test]
    fn test_canvas_set_out_of_bounds() {
        // Should not panic.
        let mut canvas = Canvas::new(5, 5, CharSet::Unicode);
        canvas.set(10, 10, 'X'); // out of bounds — silently ignored
        assert_eq!(canvas.get(10, 10), ' '); // returns ' ' for OOB
    }

    #[test]
    fn test_canvas_set_merge_junction() {
        let mut canvas = Canvas::new(10, 10, CharSet::Unicode);
        canvas.set(5, 5, '─');
        canvas.set_merge(5, 5, '│');
        assert_eq!(canvas.get(5, 5), '┼');
    }

    #[test]
    fn test_canvas_hline() {
        let mut canvas = Canvas::new(20, 5, CharSet::Unicode);
        canvas.hline(2, 3, 7, '─');
        for col in 3..=7 {
            assert_eq!(canvas.get(col, 2), '─', "col={}", col);
        }
        assert_eq!(canvas.get(2, 2), ' ');
        assert_eq!(canvas.get(8, 2), ' ');
    }

    #[test]
    fn test_canvas_vline() {
        let mut canvas = Canvas::new(10, 20, CharSet::Unicode);
        canvas.vline(4, 2, 8, '│');
        for row in 2..=8 {
            assert_eq!(canvas.get(4, row), '│', "row={}", row);
        }
    }

    #[test]
    fn test_canvas_draw_box() {
        let mut canvas = Canvas::new(20, 10, CharSet::Unicode);
        let bc = BoxChars::unicode();
        let rect = Rect::new(2, 1, 6, 3);
        canvas.draw_box(&rect, &bc);

        // Corners.
        assert_eq!(canvas.get(2, 1), '┌');
        assert_eq!(canvas.get(7, 1), '┐');
        assert_eq!(canvas.get(2, 3), '└');
        assert_eq!(canvas.get(7, 3), '┘');

        // Top edge.
        for col in 3..7 {
            assert_eq!(canvas.get(col, 1), '─', "top col={}", col);
        }

        // Left edge.
        assert_eq!(canvas.get(2, 2), '│');
        // Right edge.
        assert_eq!(canvas.get(7, 2), '│');
    }

    #[test]
    fn test_canvas_write_str() {
        let mut canvas = Canvas::new(20, 5, CharSet::Unicode);
        canvas.write_str(3, 2, "hello");
        assert_eq!(canvas.get(3, 2), 'h');
        assert_eq!(canvas.get(4, 2), 'e');
        assert_eq!(canvas.get(7, 2), 'o');
    }

    #[test]
    fn test_canvas_to_string_trims_trailing_spaces() {
        let mut canvas = Canvas::new(10, 3, CharSet::Unicode);
        canvas.set(0, 0, 'A');
        let s = canvas.to_string();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0], "A"); // trailing spaces stripped
    }

    #[test]
    fn test_hline_vline_junction_merge() {
        // Drawing a horizontal line then a vertical line crossing it should
        // produce a ┼ at the intersection.
        let mut canvas = Canvas::new(20, 20, CharSet::Unicode);
        canvas.hline(5, 2, 10, '─');
        canvas.vline(6, 2, 10, '│');
        // At (6, 5) we have both h and v — should be ┼.
        assert_eq!(canvas.get(6, 5), '┼');
        // At (6, 2) — only v so far before the h crosses (top of v).
        assert_eq!(canvas.get(6, 3), '│');
    }

    #[test]
    fn test_rect_right_bottom() {
        let r = Rect::new(3, 4, 10, 5);
        assert_eq!(r.right(), 13);
        assert_eq!(r.bottom(), 9);
    }
}
