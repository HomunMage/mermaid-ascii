//! Integration tests for the mermaid-ascii binary.
//!
//! These tests run the compiled binary and verify output against golden .expect files.
//! Ported from the former Python tests/e2e/test_rust_binary.py.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Get the path to the compiled binary (debug build, built by `cargo test`).
fn binary_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push("debug");
    path.push("mermaid-ascii");
    path
}

/// Get the examples directory.
fn examples_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("examples");
    path
}

/// Run the binary with the given stdin input and extra CLI args. Returns stdout.
fn run_binary(input: &str, extra_args: &[&str]) -> String {
    let bin = binary_path();
    assert!(
        bin.exists(),
        "Binary not found at {:?}. Run `cargo build` first.",
        bin
    );

    let output = Command::new(&bin)
        .args(extra_args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                stdin.write_all(input.as_bytes()).ok();
            }
            child.wait_with_output()
        })
        .expect("Failed to run binary");

    assert!(
        output.status.success(),
        "Binary exited with {:?}:\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).expect("Non-UTF8 output")
}

/// Find all (name, mm_file, expect_file) triples in the examples directory.
fn find_example_pairs() -> Vec<(String, PathBuf, PathBuf)> {
    let dir = examples_dir();
    let mut pairs = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md")
                && path.to_string_lossy().ends_with(".mm.md")
            {
                let stem = path.file_name().unwrap().to_string_lossy();
                let name = stem.trim_end_matches(".mm.md").to_string();
                let expect_path = dir.join(format!("{}.expect", name));
                if expect_path.exists() {
                    pairs.push((name, path, expect_path));
                }
            }
        }
    }
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

// ─── Golden file tests ──────────────────────────────────────────────────────

#[test]
fn test_all_examples_match_expect() {
    let pairs = find_example_pairs();
    assert!(
        !pairs.is_empty(),
        "No example pairs found in {:?}",
        examples_dir()
    );

    let mut failures = Vec::new();
    for (name, mm_file, expect_file) in &pairs {
        let src = fs::read_to_string(mm_file)
            .unwrap_or_else(|e| panic!("Cannot read {:?}: {}", mm_file, e));
        let expected = fs::read_to_string(expect_file)
            .unwrap_or_else(|e| panic!("Cannot read {:?}: {}", expect_file, e));

        let mut actual = run_binary(&src, &[]);
        if expected.ends_with('\n') && !actual.ends_with('\n') {
            actual.push('\n');
        }

        if actual != expected {
            failures.push(format!(
                "FAIL: {} (expected {} bytes, got {} bytes)",
                name,
                expected.len(),
                actual.len()
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Golden file mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

// ─── SVG golden file tests ──────────────────────────────────────────────

/// Find all (name, mm_file, expect_svg) triples for SVG golden tests.
fn find_svg_pairs() -> Vec<(String, PathBuf, PathBuf)> {
    let dir = examples_dir();
    let mut pairs = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md")
                && path.to_string_lossy().ends_with(".mm.md")
            {
                let stem = path.file_name().unwrap().to_string_lossy();
                let name = stem.trim_end_matches(".mm.md").to_string();
                let expect_svg = dir.join(format!("{}.expect.svg", name));
                if expect_svg.exists() {
                    pairs.push((name, path, expect_svg));
                }
            }
        }
    }
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

#[test]
fn test_all_examples_match_expect_svg() {
    let pairs = find_svg_pairs();
    assert!(
        !pairs.is_empty(),
        "No SVG expect pairs found in {:?}",
        examples_dir()
    );

    let mut failures = Vec::new();
    for (name, mm_file, expect_svg) in &pairs {
        let src = fs::read_to_string(mm_file)
            .unwrap_or_else(|e| panic!("Cannot read {:?}: {}", mm_file, e));
        let expected = fs::read_to_string(expect_svg)
            .unwrap_or_else(|e| panic!("Cannot read {:?}: {}", expect_svg, e));

        let actual = run_binary(&src, &["--svg"]);

        let expected_trimmed = expected.trim_end_matches('\n');
        let actual_trimmed = actual.trim_end_matches('\n');

        if actual_trimmed != expected_trimmed {
            failures.push(format!(
                "FAIL: {} (expected {} bytes, got {} bytes)",
                name,
                expected_trimmed.len(),
                actual_trimmed.len()
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "SVG golden file mismatches ({}/{}):\n{}",
            failures.len(),
            pairs.len(),
            failures.join("\n")
        );
    }
}

// ─── Flag tests ─────────────────────────────────────────────────────────────

#[test]
fn test_ascii_flag() {
    let src = "graph TD\n    A --> B\n";
    let output = run_binary(src, &["--ascii"]);
    assert!(
        !output.contains('┌'),
        "Unicode char found in --ascii output"
    );
    assert!(
        !output.contains('│'),
        "Unicode char found in --ascii output"
    );
    assert!(
        !output.contains('─'),
        "Unicode char found in --ascii output"
    );
    assert!(
        output.contains('+') || output.contains('|') || output.contains('-'),
        "No ASCII box chars in output"
    );
}

#[test]
fn test_direction_override_lr() {
    let src = "graph TD\n    A --> B --> C\n";
    let output_td = run_binary(src, &[]);
    let output_lr = run_binary(src, &["--direction", "LR"]);

    let lines_td: Vec<&str> = output_td.lines().filter(|l| !l.trim().is_empty()).collect();
    let lines_lr: Vec<&str> = output_lr.lines().filter(|l| !l.trim().is_empty()).collect();
    let width_td = lines_td.iter().map(|l| l.len()).max().unwrap_or(0);
    let width_lr = lines_lr.iter().map(|l| l.len()).max().unwrap_or(0);

    assert!(
        width_lr > width_td,
        "LR output should be wider than TD output"
    );
    assert!(
        lines_lr.len() < lines_td.len(),
        "LR output should have fewer lines than TD output"
    );
}

#[test]
fn test_direction_override_bt() {
    let src = "graph TD\n    A --> B\n";
    let output = run_binary(src, &["--direction", "BT"]);
    let lines: Vec<&str> = output.lines().collect();
    let b_line = lines.iter().position(|l| l.contains('B'));
    let a_line = lines.iter().position(|l| l.contains('A'));
    assert!(
        b_line.is_some() && a_line.is_some(),
        "Both A and B must appear in output"
    );
    assert!(
        b_line.unwrap() < a_line.unwrap(),
        "In BT layout, B (target) should appear above A (source)"
    );
}

#[test]
fn test_reads_from_file() {
    let dir = std::env::temp_dir().join("mermaid_ascii_test_read");
    fs::create_dir_all(&dir).ok();
    let input_file = dir.join("test_input.mm.md");
    fs::write(&input_file, "graph LR\n    X --> Y\n").unwrap();

    let bin = binary_path();
    let output = Command::new(&bin)
        .arg(input_file.to_str().unwrap())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("Failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains('X'), "Output should contain X");
    assert!(stdout.contains('Y'), "Output should contain Y");

    fs::remove_file(&input_file).ok();
    fs::remove_dir(&dir).ok();
}

#[test]
fn test_output_to_file() {
    let dir = std::env::temp_dir().join("mermaid_ascii_test_write");
    fs::create_dir_all(&dir).ok();
    let out_file = dir.join("out.txt");

    let src = "graph TD\n    A --> B\n";
    let bin = binary_path();
    let output = Command::new(&bin)
        .args(["--output", out_file.to_str().unwrap()])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(ref mut stdin) = child.stdin {
                stdin.write_all(src.as_bytes()).ok();
            }
            child.wait_with_output()
        })
        .expect("Failed to run binary");

    assert!(output.status.success());
    assert!(out_file.exists(), "Output file should exist");
    let content = fs::read_to_string(&out_file).unwrap();
    assert!(content.contains('A'), "Output file should contain A");
    assert!(content.contains('B'), "Output file should contain B");

    fs::remove_file(&out_file).ok();
    fs::remove_dir(&dir).ok();
}
