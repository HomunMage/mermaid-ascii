/// AST data structures for the text-graph DSL.
///
/// These types represent the parsed form of the input DSL.
/// They map closely to the grammar rules in out/phase0/grammar.pest.

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
    Rectangle,  // [Label]
    Rounded,    // (Label)
    Diamond,    // {Label}
    Circle,     // ((Label))
}

// ─── Edge Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeType {
    Arrow,       // --> or ->
    Line,        // --
    BackArrow,   // <-- or <-
    BidirArrow,  // <--> or <->
    ThickArrow,  // ==> or =>
    DoubleLine,  // ===
    DottedArrow, // .--> or .->
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
    /// Unique identifier derived from the label (or assigned label).
    pub id: String,
    /// Display label (may be quoted or bare text from the DSL).
    pub label: String,
    pub shape: NodeShape,
    pub attrs: Vec<Attr>,
}

impl Node {
    pub fn new(label: impl Into<String>, shape: NodeShape) -> Self {
        let label = label.into();
        let id = label.clone();
        Self { id, label, shape, attrs: Vec::new() }
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
    /// Optional inline label on the edge.
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
    /// Nested subgraphs (the grammar allows nesting via `statement*`).
    pub subgraphs: Vec<Subgraph>,
    /// Optional description text shown inside the subgraph box.
    pub description: Option<String>,
}

impl Subgraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            subgraphs: Vec::new(),
            description: None,
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
