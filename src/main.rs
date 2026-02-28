//! mermaid-ascii CLI entry point.

use std::fs;
use std::io::{self, Read, Write};
use std::process;

use clap::Parser;

use mermaid_ascii::{render_dsl, render_svg_dsl};

/// Mermaid flowchart to ASCII/Unicode graph output.
#[derive(Parser, Debug)]
#[command(
    name = "mermaid-ascii",
    version = env!("MERMAID_ASCII_VERSION"),
    about = "Mermaid flowchart to ASCII/Unicode graph output"
)]
struct Cli {
    /// Input file (reads from stdin if not provided)
    input: Option<String>,

    /// Use plain ASCII instead of Unicode box-drawing characters
    #[arg(short = 'a', long = "ascii")]
    use_ascii: bool,

    /// Override direction (LR, RL, TD, BT)
    #[arg(short = 'd', long = "direction")]
    direction: Option<String>,

    /// Node padding (spaces inside border)
    #[arg(short = 'p', long = "padding", default_value = "1")]
    padding: usize,

    /// Output SVG instead of ASCII/Unicode
    #[arg(short = 's', long = "svg")]
    use_svg: bool,

    /// Write output to this file instead of stdout
    #[arg(short = 'o', long = "output")]
    output: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let text = if let Some(ref path) = cli.input {
        match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot read '{}': {}", path, e);
                process::exit(1);
            }
        }
    } else {
        let mut buf = String::new();
        if let Err(e) = io::stdin().read_to_string(&mut buf) {
            eprintln!("error: cannot read stdin: {}", e);
            process::exit(1);
        }
        buf
    };

    let direction = cli.direction.as_deref();
    let rendered = if cli.use_svg {
        match render_svg_dsl(&text, cli.padding, direction) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {}", e);
                process::exit(1);
            }
        }
    } else {
        let unicode = !cli.use_ascii;
        match render_dsl(&text, unicode, cli.padding, direction) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {}", e);
                process::exit(1);
            }
        }
    };

    if let Some(ref path) = cli.output {
        match fs::write(path, rendered) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: cannot write '{}': {}", path, e);
                process::exit(1);
            }
        }
    } else {
        print!("{}", rendered);
        if let Err(e) = io::stdout().flush() {
            eprintln!("error: cannot flush stdout: {}", e);
            process::exit(1);
        }
    }
}
