mod ast;
mod graph;
mod layout;
mod parser;
mod render;

use std::fs;
use std::io::{self, Read};

use clap::Parser as ClapParser;

/// text-graph: DSL text input → ASCII/Unicode graph output
#[derive(ClapParser, Debug)]
#[command(name = "text-graph", version, about)]
struct Cli {
    /// Input file (reads from stdin if omitted)
    input: Option<String>,

    /// Use plain ASCII characters instead of Unicode box-drawing
    #[arg(long, short = 'a')]
    ascii: bool,

    /// Override graph direction (LR, RL, TD, BT)
    #[arg(long, short = 'd', value_name = "DIR")]
    direction: Option<String>,

    /// Node padding (spaces inside node border on each side)
    #[arg(long, short = 'p', value_name = "N", default_value = "1")]
    padding: usize,

    /// Write output to this file instead of stdout
    #[arg(long, short = 'o', value_name = "FILE")]
    output: Option<String>,

    /// Generate phase1 output files (dev/debug)
    #[arg(long, hide = true)]
    gen_phase1: bool,

    /// Generate phase2 output files (dev/debug)
    #[arg(long, hide = true)]
    gen_phase2: bool,

    /// Generate phase3 output files (dev/debug)
    #[arg(long, hide = true)]
    gen_phase3: bool,

    /// Generate phase4 output files (dev/debug)
    #[arg(long, hide = true)]
    gen_phase4: bool,

    /// Generate phase5 output files (dev/debug)
    #[arg(long, hide = true)]
    gen_phase5: bool,
}

fn main() {
    let cli = Cli::parse();

    // Dev/debug generation flags — these ignore all other options.
    if cli.gen_phase1 { gen_phase1_output(); return; }
    if cli.gen_phase2 { gen_phase2_output(); return; }
    if cli.gen_phase3 { gen_phase3_output(); return; }
    if cli.gen_phase4 { gen_phase4_output(); return; }
    if cli.gen_phase5 { gen_phase5_output(); return; }

    // Read input from file or stdin.
    let input = match &cli.input {
        Some(path) => fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("error: cannot read '{}': {}", path, e);
            std::process::exit(1);
        }),
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf).unwrap_or_else(|e| {
                eprintln!("error: cannot read stdin: {}", e);
                std::process::exit(1);
            });
            buf
        }
    };

    // Parse.
    let mut ast_graph = parser::parse(&input).unwrap_or_else(|e| {
        eprintln!("parse error:\n{}", e);
        std::process::exit(1);
    });

    // Direction override.
    if let Some(ref dir_str) = cli.direction {
        ast_graph.direction = match dir_str.to_uppercase().as_str() {
            "LR" => ast::Direction::LR,
            "RL" => ast::Direction::RL,
            "TD" | "TB" => ast::Direction::TD,
            "BT" => ast::Direction::BT,
            other => {
                eprintln!("error: unknown direction '{}'; use LR, RL, TD, or BT", other);
                std::process::exit(1);
            }
        };
    }

    // Build graph IR.
    let gir = graph::GraphIR::from_ast(&ast_graph);

    if gir.node_count() == 0 {
        // Empty graph — output nothing gracefully.
        if let Some(ref out_path) = cli.output {
            fs::write(out_path, "").unwrap_or_else(|e| {
                eprintln!("error: cannot write '{}': {}", out_path, e);
                std::process::exit(1);
            });
        }
        return;
    }

    // Layout + render.  padding option passed through to layout constants.
    let (layout_nodes, routed_edges) = layout::full_layout_with_padding(&gir, cli.padding);
    let use_unicode = !cli.ascii;
    let rendered = render::render(&gir, &layout_nodes, &routed_edges, use_unicode);

    // Write output.
    match &cli.output {
        Some(path) => {
            fs::write(path, &rendered).unwrap_or_else(|e| {
                eprintln!("error: cannot write '{}': {}", path, e);
                std::process::exit(1);
            });
        }
        None => print!("{}", rendered),
    }
}

// ─── Dev/debug generation helpers ───────────────────────────────────────────

fn render_example(src: &str, unicode: bool) -> String {
    match parser::parse(src) {
        Err(e) => format!("PARSE ERROR: {}\n", e),
        Ok(ast_graph) => {
            let gir = graph::GraphIR::from_ast(&ast_graph);
            if gir.node_count() == 0 {
                return "(empty graph)\n".to_string();
            }
            let (layout_nodes, routed_edges) = layout::full_layout(&gir);
            render::render(&gir, &layout_nodes, &routed_edges, unicode)
        }
    }
}

fn extract_examples(text: &str) -> Vec<(String, String)> {
    let mut results = Vec::new();
    let separator = "=".repeat(80);
    let mut lines = text.lines().peekable();
    let mut current_title: Option<String> = None;
    let mut current_body = String::new();
    let mut in_body = false;

    while let Some(line) = lines.next() {
        if line == separator {
            if let Some(title_line) = lines.next() {
                let _sep2 = lines.next();
                if let Some(title) = current_title.take() {
                    let trimmed = current_body.trim().to_string();
                    if !trimmed.is_empty() {
                        results.push((title, format!("{}\n", trimmed)));
                    }
                }
                current_title = Some(title_line.trim().to_string());
                current_body.clear();
                in_body = true;
            }
        } else if in_body {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    if let Some(title) = current_title {
        let trimmed = current_body.trim().to_string();
        if !trimmed.is_empty() {
            results.push((title, format!("{}\n", trimmed)));
        }
    }

    results
}

fn gen_phase1_output() {
    fs::create_dir_all("out/phase1").unwrap();

    let examples_txt = fs::read_to_string("out/phase0/syntax_examples.txt")
        .expect("Cannot read out/phase0/syntax_examples.txt");
    let examples = extract_examples(&examples_txt);

    let bad_inputs: &[(&str, &str)] = &[
        ("missing_closing_bracket", "[A --> [B]\n"),
        ("empty_edge_target", "[A] -->\n"),
        ("bad_direction", "direction: DIAG\n[A] --> [B]\n"),
        ("unclosed_subgraph", "subgraph \"Foo\" {\n  [A]\n"),
        ("unknown_token", "@ invalid @\n"),
    ];

    let mut ast_out = String::new();
    let mut err_out = String::new();

    ast_out.push_str("# Phase 1: AST Dump\n");
    ast_out.push_str("# Generated by: cargo run -- --gen-phase1\n");
    ast_out.push_str("# Each example parsed from out/phase0/syntax_examples.txt\n\n");

    err_out.push_str("# Phase 1: Parse Error Examples\n");
    err_out.push_str("# Generated by: cargo run -- --gen-phase1\n\n");

    for (title, src) in &examples {
        ast_out.push_str(&format!("{}\n", "=".repeat(80)));
        ast_out.push_str(&format!("{}\n", title));
        ast_out.push_str(&format!("{}\n", "=".repeat(80)));
        ast_out.push_str("\nInput:\n");
        ast_out.push_str(src.trim());
        ast_out.push_str("\n\nParsed AST:\n");
        match parser::parse(src) {
            Ok(graph) => ast_out.push_str(&format!("{:#?}", graph)),
            Err(e) => ast_out.push_str(&format!("PARSE ERROR:\n{}", e)),
        }
        ast_out.push_str("\n\n");
    }

    for (label, src) in bad_inputs {
        err_out.push_str(&format!("{}\n", "=".repeat(80)));
        err_out.push_str(&format!("Bad input: {}\n", label));
        err_out.push_str(&format!("{}\n", "=".repeat(80)));
        err_out.push_str("\nInput:\n");
        err_out.push_str(src.trim());
        err_out.push_str("\n\nError:\n");
        match parser::parse(src) {
            Ok(graph) => {
                err_out.push_str(&format!("(parsed successfully — expected error)\n{:#?}", graph));
            }
            Err(e) => err_out.push_str(&e),
        }
        err_out.push_str("\n\n");
    }

    fs::write("out/phase1/ast_dump.txt", &ast_out).expect("Cannot write ast_dump.txt");
    fs::write("out/phase1/parse_errors.txt", &err_out).expect("Cannot write parse_errors.txt");
    println!("Generated out/phase1/ast_dump.txt");
    println!("Generated out/phase1/parse_errors.txt");
}

fn gen_phase2_output() {
    fs::create_dir_all("out/phase2").unwrap();

    let examples_txt = fs::read_to_string("out/phase0/syntax_examples.txt")
        .expect("Cannot read out/phase0/syntax_examples.txt");
    let examples = extract_examples(&examples_txt);

    let mut info_out = String::new();
    let mut adj_out = String::new();

    info_out.push_str("# Phase 2: Graph Info\n");
    info_out.push_str("# Generated by: cargo run -- --gen-phase2\n");
    info_out.push_str("# Shows topology metrics for each example\n\n");

    adj_out.push_str("# Phase 2: Adjacency Lists\n");
    adj_out.push_str("# Generated by: cargo run -- --gen-phase2\n");
    adj_out.push_str("# Shows outgoing neighbors for each node\n\n");

    for (title, src) in &examples {
        let sep = "=".repeat(80);
        info_out.push_str(&format!("{}\n{}\n{}\n\n", sep, title, sep));
        adj_out.push_str(&format!("{}\n{}\n{}\n\n", sep, title, sep));

        match parser::parse(src) {
            Err(e) => {
                info_out.push_str(&format!("PARSE ERROR: {}\n\n", e));
                adj_out.push_str(&format!("PARSE ERROR: {}\n\n", e));
            }
            Ok(ast_graph) => {
                let gir = graph::GraphIR::from_ast(&ast_graph);

                let topo_str = match gir.topological_order() {
                    Some(order) => order.join(", "),
                    None => "(cyclic — no topological order)".to_string(),
                };

                info_out.push_str(&format!("Direction:        {:?}\n", gir.direction));
                info_out.push_str(&format!("Node count:       {}\n", gir.node_count()));
                info_out.push_str(&format!("Edge count:       {}\n", gir.edge_count()));
                info_out.push_str(&format!("Is DAG:           {}\n", gir.is_dag()));
                info_out.push_str(&format!("Topological order: {}\n", topo_str));
                info_out.push_str("\nPer-node degrees:\n");
                let mut node_ids: Vec<String> = gir.node_index.keys().cloned().collect();
                node_ids.sort();
                for id in &node_ids {
                    info_out.push_str(&format!(
                        "  {:30} in={} out={}\n",
                        id,
                        gir.in_degree(id),
                        gir.out_degree(id)
                    ));
                }
                if !gir.subgraph_members.is_empty() {
                    info_out.push_str("\nSubgraph members:\n");
                    for (sg_name, members) in &gir.subgraph_members {
                        info_out.push_str(&format!("  [{}]: {}\n", sg_name, members.join(", ")));
                    }
                }
                info_out.push_str("\n");

                for (id, neighbors) in gir.adjacency_list() {
                    if neighbors.is_empty() {
                        adj_out.push_str(&format!("  {} → (none)\n", id));
                    } else {
                        adj_out.push_str(&format!("  {} → {}\n", id, neighbors.join(", ")));
                    }
                }
                adj_out.push_str("\n");
            }
        }
    }

    fs::write("out/phase2/graph_info.txt", &info_out).expect("Cannot write graph_info.txt");
    fs::write("out/phase2/adjacency.txt", &adj_out).expect("Cannot write adjacency.txt");
    println!("Generated out/phase2/graph_info.txt");
    println!("Generated out/phase2/adjacency.txt");
}

fn gen_phase3_output() {
    fs::create_dir_all("out/phase3").unwrap();

    let examples_txt = fs::read_to_string("out/phase0/syntax_examples.txt")
        .expect("Cannot read out/phase0/syntax_examples.txt");
    let examples = extract_examples(&examples_txt);

    let mut layer_out = String::new();
    let mut dummy_out = String::new();

    layer_out.push_str("# Phase 3: Layer Assignment\n");
    layer_out.push_str("# Generated by: cargo run -- --gen-phase3\n");
    layer_out.push_str("# Shows which layer (rank) each node is assigned to\n\n");

    dummy_out.push_str("# Phase 3: Dummy Node Insertion\n");
    dummy_out.push_str("# Generated by: cargo run -- --gen-phase3\n");
    dummy_out.push_str("# Shows dummy nodes inserted for edges spanning multiple layers\n\n");

    for (title, src) in &examples {
        let sep = "=".repeat(80);
        layer_out.push_str(&format!("{}\n{}\n{}\n\n", sep, title, sep));
        dummy_out.push_str(&format!("{}\n{}\n{}\n\n", sep, title, sep));

        match parser::parse(src) {
            Err(e) => {
                layer_out.push_str(&format!("PARSE ERROR: {}\n\n", e));
                dummy_out.push_str(&format!("PARSE ERROR: {}\n\n", e));
            }
            Ok(ast_graph) => {
                let gir = graph::GraphIR::from_ast(&ast_graph);
                let la = layout::LayerAssignment::assign(&gir);
                layer_out.push_str(&la.format_report(&gir));
                layer_out.push('\n');

                let (dag, _) = layout::remove_cycles(&gir.digraph);
                let aug = layout::insert_dummy_nodes(&dag, &la);

                dummy_out.push_str(&format!(
                    "Original nodes: {}  |  After insertion: {}  |  Dummy nodes: {}\n",
                    gir.node_count(),
                    aug.graph.node_count(),
                    aug.graph.node_count() - gir.node_count()
                ));
                dummy_out.push_str(&format!(
                    "Long edges broken up: {}\n",
                    aug.dummy_edges.len()
                ));

                if aug.dummy_edges.is_empty() {
                    dummy_out.push_str("  (no long edges — all edges are already adjacent-layer)\n");
                } else {
                    for de in &aug.dummy_edges {
                        dummy_out.push_str(&format!(
                            "  {} → {} : via [{}]\n",
                            de.original_src,
                            de.original_tgt,
                            de.dummy_ids.join(", ")
                        ));
                        for dummy_id in &de.dummy_ids {
                            dummy_out.push_str(&format!(
                                "    {} (layer {})\n",
                                dummy_id,
                                aug.layers[dummy_id]
                            ));
                        }
                    }
                }
                dummy_out.push('\n');
            }
        }
    }

    fs::write("out/phase3/layer_assignment.txt", &layer_out).expect("Cannot write layer_assignment.txt");
    fs::write("out/phase3/dummy_nodes.txt", &dummy_out).expect("Cannot write dummy_nodes.txt");
    println!("Generated out/phase3/layer_assignment.txt");
    println!("Generated out/phase3/dummy_nodes.txt");
}

fn gen_phase4_output() {
    fs::create_dir_all("out/phase4").unwrap();

    let examples_txt = fs::read_to_string("out/phase0/syntax_examples.txt")
        .expect("Cannot read out/phase0/syntax_examples.txt");
    let examples = extract_examples(&examples_txt);

    let mut paths_out = String::new();
    let mut visual_out = String::new();

    paths_out.push_str("# Phase 4: Edge Routing Waypoints\n");
    paths_out.push_str("# Generated by: cargo run -- --gen-phase4\n");
    paths_out.push_str("# Shows orthogonal waypoints for each edge\n\n");

    visual_out.push_str("# Phase 4: Routing Visualisation\n");
    visual_out.push_str("# Generated by: cargo run -- --gen-phase4\n");
    visual_out.push_str("# Shows node positions and edge waypoints as a table\n\n");

    for (title, src) in &examples {
        let sep = "=".repeat(80);
        paths_out.push_str(&format!("{}\n{}\n{}\n\n", sep, title, sep));
        visual_out.push_str(&format!("{}\n{}\n{}\n\n", sep, title, sep));

        match parser::parse(src) {
            Err(e) => {
                paths_out.push_str(&format!("PARSE ERROR: {}\n\n", e));
                visual_out.push_str(&format!("PARSE ERROR: {}\n\n", e));
            }
            Ok(ast_graph) => {
                let gir = graph::GraphIR::from_ast(&ast_graph);
                let (layout_nodes, routed_edges) = layout::full_layout(&gir);

                if layout_nodes.is_empty() {
                    paths_out.push_str("  (empty graph)\n");
                    visual_out.push_str("  (empty graph)\n");
                } else {
                    paths_out.push_str("Nodes:\n");
                    let mut sorted_nodes = layout_nodes.clone();
                    sorted_nodes.sort_by(|a, b| a.layer.cmp(&b.layer).then(a.order.cmp(&b.order)));
                    for n in &sorted_nodes {
                        if !n.id.starts_with(layout::DUMMY_PREFIX) {
                            paths_out.push_str(&format!(
                                "  {:30} layer={} order={} x={} y={} w={} h={}\n",
                                n.id, n.layer, n.order, n.x, n.y, n.width, n.height
                            ));
                        }
                    }

                    paths_out.push_str("\nEdge paths:\n");
                    if routed_edges.is_empty() {
                        paths_out.push_str("  (no edges)\n");
                    }
                    let mut sorted_edges = routed_edges.clone();
                    sorted_edges.sort_by(|a, b| a.from_id.cmp(&b.from_id).then(a.to_id.cmp(&b.to_id)));
                    for re in &sorted_edges {
                        let pts: Vec<String> = re.waypoints.iter()
                            .map(|p| format!("({},{})", p.x, p.y))
                            .collect();
                        let label_str = re.label.as_deref()
                            .map(|l| format!(" [label: \"{}\"]", l))
                            .unwrap_or_default();
                        paths_out.push_str(&format!(
                            "  {} → {}{}: {}\n",
                            re.from_id, re.to_id, label_str, pts.join(" → ")
                        ));
                    }
                    paths_out.push('\n');

                    visual_out.push_str("Node grid (layer, order → x, y, w, h):\n");
                    for n in &sorted_nodes {
                        if !n.id.starts_with(layout::DUMMY_PREFIX) {
                            visual_out.push_str(&format!(
                                "  [{:2},{:2}]  {:30}  x={:4} y={:3}  {}x{}\n",
                                n.layer, n.order, n.id, n.x, n.y, n.width, n.height
                            ));
                        }
                    }
                    visual_out.push_str("\nEdge routing summary:\n");
                    for re in &sorted_edges {
                        let from_n = layout_nodes.iter().find(|n| n.id == re.from_id);
                        let to_n = layout_nodes.iter().find(|n| n.id == re.to_id);
                        let (fl, tl) = match (from_n, to_n) {
                            (Some(f), Some(t)) => (f.layer, t.layer),
                            _ => (0, 0),
                        };
                        visual_out.push_str(&format!(
                            "  {} (L{}) → {} (L{})  : {} waypoints\n",
                            re.from_id, fl, re.to_id, tl, re.waypoints.len()
                        ));
                    }
                    visual_out.push('\n');
                }
            }
        }
    }

    fs::write("out/phase4/edge_paths.txt", &paths_out).expect("Cannot write edge_paths.txt");
    fs::write("out/phase4/routing_visual.txt", &visual_out).expect("Cannot write routing_visual.txt");
    println!("Generated out/phase4/edge_paths.txt");
    println!("Generated out/phase4/routing_visual.txt");
}

fn gen_phase5_output() {
    fs::create_dir_all("out/phase5").unwrap();

    let examples_txt = fs::read_to_string("out/phase0/syntax_examples.txt")
        .expect("Cannot read out/phase0/syntax_examples.txt");
    let examples = extract_examples(&examples_txt);

    let named_inputs: &[(&str, &str, &str)] = &[
        ("simple_chain",  "simple_chain.txt",  "[A] --> [B] --> [C]\n"),
        ("diamond",       "diamond.txt",        "[A] --> [B]\n[A] --> [C]\n[B] --> [D]\n[C] --> [D]\n"),
        ("labeled_edges", "labeled_edges.txt",  "[A] --> [B] { label: \"step 1\" }\n[B] --> [C] { label: \"step 2\" }\n"),
        ("subgraph",      "subgraph.txt",       "subgraph \"Group\" {\n  [A] --> [B]\n}\n[B] --> [C]\n"),
        ("td_layout",     "td_layout.txt",      "direction: TD\n[Start] --> [Middle] --> [End]\n"),
        ("lr_layout",     "lr_layout.txt",      "direction: LR\n[Start] --> [Middle] --> [End]\n"),
        ("ascii_mode",    "ascii_mode.txt",     "[A] --> [B] --> [C]\n[A] --> [C]\n"),
    ];

    for (name, filename, src) in named_inputs {
        let use_unicode = *name != "ascii_mode";
        let out_text = render_example(src, use_unicode);
        let path = format!("out/phase5/{}", filename);
        fs::write(&path, &out_text).unwrap_or_else(|e| eprintln!("Cannot write {}: {}", path, e));
        println!("Generated {}", path);
    }

    let mut complex_out = String::new();
    complex_out.push_str("# Phase 5: All phase0 examples rendered\n");
    complex_out.push_str("# Generated by: cargo run -- --gen-phase5\n\n");

    for (title, src) in &examples {
        let sep = "=".repeat(80);
        complex_out.push_str(&format!("{}\n{}\n{}\n\n", sep, title, sep));
        complex_out.push_str(&render_example(src, true));
        complex_out.push('\n');
    }

    fs::write("out/phase5/complex.txt", &complex_out).expect("Cannot write complex.txt");
    println!("Generated out/phase5/complex.txt");
}
