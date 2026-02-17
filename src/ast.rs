/// AST data structures for Mermaid flowchart syntax.
///
/// These types represent the parsed form of the input DSL.
/// Aligned with mermaid-ascii / beautiful-mermaid conventions.

// ─── Direction ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Direction {
    LR,
    RL,
    #[default]
    TD,
    BT,
}

// ─── Node Shapes ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum NodeShape {
    #[default]
    Rectangle,  // id[Label]
    Rounded,    // id(Label)
    Diamond,    // id{Label}
    Circle,     // id((Label))
}

// ─── Edge Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeType {
    Arrow,       // -->
    Line,        // ---
    DottedArrow, // -.->
    DottedLine,  // -.-
    ThickArrow,  // ==>
    ThickLine,   // ===
    BidirArrow,  // <-->
    BidirDotted, // <-.->
    BidirThick,  // <==>
}

// ─── Attribute ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attr {
    pub key: String,
    pub value: String,
}

// ─── Node ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// Mermaid identifier (e.g. "A", "Start", "my-node").
    pub id: String,
    /// Display label (e.g. "Hello World"). Defaults to id if no shape bracket.
    pub label: String,
    pub shape: NodeShape,
    pub attrs: Vec<Attr>,
}

impl Node {
    pub fn new(id: impl Into<String>, label: impl Into<String>, shape: NodeShape) -> Self {
        Self { id: id.into(), label: label.into(), shape, attrs: Vec::new() }
    }

    /// Create a bare node (id = label, default rectangle shape).
    pub fn bare(id: impl Into<String>) -> Self {
        let id = id.into();
        let label = id.clone();
        Self { id, label, shape: NodeShape::Rectangle, attrs: Vec::new() }
    }
}

// ─── Edge ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    /// ID of the source node.
    pub from: String,
    /// ID of the target node.
    pub to: String,
    pub edge_type: EdgeType,
    /// Optional inline label on the edge (from |text| syntax).
    pub label: Option<String>,
    pub attrs: Vec<Attr>,
}

impl Edge {
    pub fn new(from: impl Into<String>, to: impl Into<String>, edge_type: EdgeType) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            edge_type,
            label: None,
            attrs: Vec::new(),
        }
    }
}

// ─── Subgraph ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subgraph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// Nested subgraphs.
    pub subgraphs: Vec<Subgraph>,
    /// Optional description text shown inside the subgraph box.
    pub description: Option<String>,
    /// Optional direction override within this subgraph.
    pub direction: Option<Direction>,
}

impl Subgraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            subgraphs: Vec::new(),
            description: None,
            direction: None,
        }
    }
}

// ─── Graph (top-level AST) ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Graph {
    pub direction: Direction,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub subgraphs: Vec<Subgraph>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }
}
