//! Integration test: run ALL _site/examples/*.mm.md through the full pipeline
//! and compare output against .expect.txt and .expect.svg golden files.

use mermaid_ascii::{render_dsl, render_svg_dsl};
use std::fs;
use std::path::Path;

#[test]
fn test_all_examples_txt() {
    let examples_dir = Path::new("_site/examples");
    let mut tested = 0;
    let mut failures = Vec::new();

    for entry in fs::read_dir(examples_dir).expect("_site/examples/ dir must exist") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "md") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy();
        if !name.ends_with(".mm") {
            continue;
        }
        let base = name.strip_suffix(".mm").unwrap();

        let expect_path = examples_dir.join(format!("{}.expect.txt", base));
        if !expect_path.exists() {
            continue;
        }

        let input = fs::read_to_string(&path).unwrap();
        let expected = fs::read_to_string(&expect_path).unwrap();
        let result = render_dsl(&input, true, 1, None).unwrap();

        if result.trim() != expected.trim() {
            failures.push(base.to_string());
        }
        tested += 1;
    }

    assert!(tested > 0, "no examples found to test");
    assert!(
        failures.is_empty(),
        "txt output mismatch for: {}",
        failures.join(", ")
    );
}

#[test]
fn test_all_examples_svg() {
    let examples_dir = Path::new("_site/examples");
    let mut tested = 0;
    let mut failures = Vec::new();

    for entry in fs::read_dir(examples_dir).expect("_site/examples/ dir must exist") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "md") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy();
        if !name.ends_with(".mm") {
            continue;
        }
        let base = name.strip_suffix(".mm").unwrap();

        let expect_path = examples_dir.join(format!("{}.expect.svg", base));
        if !expect_path.exists() {
            continue;
        }

        let input = fs::read_to_string(&path).unwrap();
        let expected = fs::read_to_string(&expect_path).unwrap();
        let result = render_svg_dsl(&input, 1, None).unwrap();

        if result.trim() != expected.trim() {
            failures.push(base.to_string());
        }
        tested += 1;
    }

    assert!(tested > 0, "no examples found to test");
    assert!(
        failures.is_empty(),
        "svg output mismatch for: {}",
        failures.join(", ")
    );
}
