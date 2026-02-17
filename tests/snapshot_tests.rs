/// Integration snapshot tests for text-graph phase 5 and 6 examples.
///
/// Run `cargo test` to execute.  On first run, snapshots are written to
/// `tests/snapshots/`.  Use `cargo insta review` or `INSTA_UPDATE=always
/// cargo test` to update snapshots when intentional output changes occur.
use text_graph::render_dsl;
use text_graph::render_dsl_padded;

// ─── Phase 5 examples ───────────────────────────────────────────────────────

#[test]
fn snapshot_simple_chain() {
    let src = "[A] --> [B] --> [C]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("simple_chain", out);
}

#[test]
fn snapshot_diamond() {
    let src = "[A] --> [B]\n[A] --> [C]\n[B] --> [D]\n[C] --> [D]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("diamond", out);
}

#[test]
fn snapshot_labeled_edges() {
    let src = "[A] --> [B] { label: \"step 1\" }\n[B] --> [C] { label: \"step 2\" }\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("labeled_edges", out);
}

#[test]
fn snapshot_subgraph() {
    let src = "subgraph \"Group\" {\n  [A] --> [B]\n}\n[B] --> [C]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("subgraph", out);
}

#[test]
fn snapshot_td_layout() {
    let src = "direction: TD\n[Start] --> [Middle] --> [End]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("td_layout", out);
}

#[test]
fn snapshot_lr_layout() {
    let src = "direction: LR\n[Start] --> [Middle] --> [End]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("lr_layout", out);
}

#[test]
fn snapshot_ascii_mode() {
    let src = "[A] --> [B] --> [C]\n[A] --> [C]\n";
    let out = render_dsl(src, false).expect("render failed");
    insta::assert_snapshot!("ascii_mode", out);
}

// ─── Phase 6 examples (CLI feature parity) ───────────────────────────────────

#[test]
fn snapshot_padding3() {
    let src = "[A] --> [B] --> [C]\n";
    let out = render_dsl_padded(src, true, 3).expect("render failed");
    insta::assert_snapshot!("padding3", out);
}

#[test]
fn snapshot_empty_graph() {
    let src = "";
    let out = render_dsl(src, true).expect("render failed");
    // Empty graph renders as empty string.
    insta::assert_snapshot!("empty_graph", out);
}

#[test]
fn snapshot_single_node() {
    let src = "[Solo]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("single_node", out);
}

#[test]
fn snapshot_complex_branch() {
    // 6-node graph with multiple branches — tests crossing minimisation.
    let src = "[A] --> [B]\n[A] --> [C]\n[B] --> [D]\n[C] --> [E]\n[D] --> [F]\n[E] --> [F]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("complex_branch", out);
}

#[test]
fn snapshot_long_labels() {
    let src = "[Very Long Label A] --> [Very Long Label B]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("long_labels", out);
}

// ─── Phase 7 edge cases ──────────────────────────────────────────────────────

#[test]
fn snapshot_self_loop() {
    // Self-loop: a node with an edge back to itself.
    // Cycle removal removes the self-loop edge; the node still renders.
    let src = "[Loop] --> [Loop]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("self_loop", out);
}

#[test]
fn snapshot_self_loop_ascii() {
    let src = "[Loop] --> [Loop]\n";
    let out = render_dsl(src, false).expect("render failed");
    insta::assert_snapshot!("self_loop_ascii", out);
}

#[test]
fn snapshot_very_long_label() {
    // A single very long label — tests that node width is computed correctly.
    let src = "[This is a very long node label that spans many characters] --> [Short]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("very_long_label", out);
}

#[test]
fn snapshot_very_long_label_chain() {
    // Two very long labels chained — tests alignment across different widths.
    let src = concat!(
        "[This is a very long node label that spans many characters]",
        " --> [Another quite lengthy label here]\n",
        "[Another quite lengthy label here] --> [Short]\n",
    );
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("very_long_label_chain", out);
}

#[test]
fn snapshot_self_loop_node_topology() {
    // Validate graph topology for self-loop: exactly 1 node, 1 edge.
    use text_graph::{graph, parser};
    let src = "[Loop] --> [Loop]\n";
    let ast = parser::parse(src).expect("parse failed");
    let gir = graph::GraphIR::from_ast(&ast);
    assert_eq!(gir.node_count(), 1, "self-loop: exactly 1 node");
    assert_eq!(gir.edge_count(), 1, "self-loop: exactly 1 edge");
}

#[test]
fn snapshot_disconnected_nodes() {
    // Two nodes with no edge — disconnected components.
    let src = "[Alpha]\n[Beta]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("disconnected_nodes", out);
}

// ─── Architecture diagram / compound node tests ────────────────────────────

#[test]
fn snapshot_subgraph_with_members() {
    let src = "subgraph \"Frontend\" {\n  [Grid View]\n  [Timeline]\n  [Board View]\n}\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("subgraph_with_members", out);
}

#[test]
fn snapshot_empty_subgraph_with_desc() {
    let src = "subgraph \"Worker\" {\n  desc: \"runs background tasks\"\n}\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("empty_subgraph_with_desc", out);
}

#[test]
fn snapshot_multiline_labels() {
    let src = "[\"Line1\\nLine2\"] --> [B]\n";
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("multiline_labels", out);
}

#[test]
fn snapshot_architecture_diagram() {
    let src = include_str!("../examples/architecture.txt");
    let out = render_dsl(src, true).expect("render failed");
    insta::assert_snapshot!("architecture_diagram", out);
}

#[test]
fn snapshot_two_node_cycle_renders() {
    // A → B → A: a two-node cycle; layout must not panic and must produce output.
    // The exact node ordering after cycle removal is not deterministic, so we
    // only assert that the output is non-empty and contains node boxes.
    let src = "[A] --> [B]\n[B] --> [A]\n";
    let out = render_dsl(src, true).expect("render failed");
    assert!(!out.is_empty(), "cycle graph should produce non-empty output");
    // Edge lines may overwrite single-char labels, so check for box corners instead.
    assert!(out.contains('┌') && out.contains('┘'), "output should contain node boxes");
}
