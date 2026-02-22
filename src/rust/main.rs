//! mermaid-ascii CLI entry point.
//!
//! Mirrors Python's `__main__.py` (using clap instead of click).

use std::fs;
use std::io::{self, Read, Write};
use std::process;

use clap::Parser;

use mermaid_ascii::{render_dsl, render_svg};

/// Mermaid flowchart to ASCII/Unicode graph output.
#[derive(Parser, Debug)]
#[command(
    name = "mermaid-ascii",
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

    /// Output as SVG instead of text
    #[arg(long = "svg")]
    svg: bool,

    /// Write output to this file instead of stdout
    #[arg(short = 'o', long = "output")]
    output: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    // Read input from file or stdin
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

    // Render
    let direction = cli.direction.as_deref();
    let rendered = if cli.svg {
        match render_svg(&text, cli.padding, direction) {
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

    // Write output to file or stdout
    if let Some(ref path) = cli.output {
        match fs::write(path, rendered) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("error: cannot write '{}': {}", path, e);
                process::exit(1);
            }
        }
    } else {
        // Print without trailing newline (like Python's click.echo(nl=False))
        print!("{}", rendered);
        if let Err(e) = io::stdout().flush() {
            eprintln!("error: cannot flush stdout: {}", e);
            process::exit(1);
        }
    }
}
