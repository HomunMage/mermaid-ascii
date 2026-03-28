use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Prefer MERMAID_ASCII_VERSION env (set by Docker/CI), fall back to git tag, then "dev".
    let version = env::var("MERMAID_ASCII_VERSION")
        .ok()
        .filter(|s| !s.is_empty() && s != "dev")
        .or_else(|| {
            Command::new("git")
                .args(["describe", "--tags", "--always"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "dev".to_string());

    println!("cargo:rustc-env=MERMAID_ASCII_VERSION={}", version);

    let homunc = find_homunc();

    // Generate runtime.rs via homunc --emit-runtime
    generate_runtime(&homunc);

    // Compile .hom files → .rs into OUT_DIR (inside target/).
    // Generated .rs never pollute src/. cargo clean removes everything.
    compile_hom_files(&homunc);
}

/// Generate runtime.rs in OUT_DIR using `homunc --emit-runtime`.
/// The runtime (builtin, std, re, heap) is embedded in the homunc binary.
fn generate_runtime(homunc: &PathBuf) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let runtime_path = out_dir.join("runtime.rs");

    let output = Command::new(homunc)
        .arg("--emit-runtime")
        .output()
        .expect("failed to run homunc --emit-runtime");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("homunc --emit-runtime failed: {}", stderr);
    }

    let raw = String::from_utf8(output.stdout).expect("homunc output is not valid UTF-8");

    // The emitted runtime is designed for standalone files.
    // When included via include!(), we must:
    // 1. Strip #![...] inner attributes (not valid inside a module)
    // 2. Deduplicate imports (use std::cell::RefCell, use std::collections::HashMap, etc.)
    // 3. Deduplicate function definitions (is_alpha, is_alnum, is_digit)
    let mut seen_imports: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut seen_fns: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut code = String::with_capacity(raw.len());
    let mut skip_fn_body = false;
    let mut brace_depth: i32 = 0;

    for line in raw.lines() {
        let trimmed = line.trim();

        // Skip inner attributes
        if trimmed.starts_with("#![") {
            continue;
        }

        // Skip duplicate fn bodies
        if skip_fn_body {
            brace_depth += line.chars().filter(|&c| c == '{').count() as i32;
            brace_depth -= line.chars().filter(|&c| c == '}').count() as i32;
            if brace_depth <= 0 {
                skip_fn_body = false;
            }
            continue;
        }

        // Deduplicate `use` imports — track individual items from grouped imports
        if trimmed.starts_with("use ") && trimmed.ends_with(';') {
            // Expand `use foo::{A, B};` into individual items for tracking
            let inner = trimmed.trim_start_matches("use ").trim_end_matches(';');
            if let Some(brace_start) = inner.find('{') {
                let prefix = &inner[..brace_start];
                let items_str = inner[brace_start + 1..].trim_end_matches('}');
                let mut all_dup = true;
                for item in items_str.split(',') {
                    let item = item.trim();
                    if !item.is_empty() {
                        let full = format!("use {}{};", prefix, item);
                        if seen_imports.insert(full) {
                            all_dup = false;
                        }
                    }
                }
                if all_dup {
                    continue;
                }
            } else if !seen_imports.insert(trimmed.to_string()) {
                continue;
            }
        }

        // Deduplicate `pub fn` definitions
        if trimmed.starts_with("pub fn ") {
            let sig = trimmed.split('{').next().unwrap_or(trimmed).trim();
            if !seen_fns.insert(sig.to_string()) {
                // Strip preceding doc comments that were buffered
                while code.ends_with('\n') {
                    let last_line_start = code[..code.len() - 1].rfind('\n').map_or(0, |i| i + 1);
                    let last_line = code[last_line_start..code.len() - 1].trim();
                    if last_line.starts_with("///") || last_line.is_empty() {
                        code.truncate(last_line_start);
                    } else {
                        break;
                    }
                }
                // Skip this duplicate function body
                brace_depth = line.chars().filter(|&c| c == '{').count() as i32
                    - line.chars().filter(|&c| c == '}').count() as i32;
                if brace_depth > 0 {
                    skip_fn_body = true;
                }
                continue;
            }
        }

        code.push_str(line);
        code.push('\n');
    }

    // Strip #[cfg(test)] mod tests { ... } blocks to avoid duplicates
    let code = strip_test_modules(&code);

    std::fs::write(&runtime_path, &code).unwrap();
    println!(
        "cargo:warning=Generated runtime.rs ({} bytes) via homunc --emit-runtime",
        code.len()
    );
}

fn find_homunc() -> PathBuf {
    let local = PathBuf::from(".tmp/homunc");
    if local.exists() {
        return local;
    }
    // Try PATH
    if Command::new("homunc").arg("--version").output().is_ok() {
        return PathBuf::from("homunc");
    }
    // Download from GitHub releases
    std::fs::create_dir_all(".tmp").unwrap();
    let url = "https://github.com/homun-lang/homun/releases/latest/download/homunc-linux-x86_64";
    let status = Command::new("wget")
        .args(["-q", url, "-O", local.to_str().unwrap()])
        .status();
    if let Ok(s) = status
        && s.success()
    {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&local, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        println!("cargo:warning=Downloaded homunc to .tmp/homunc");
        return local;
    }
    panic!("Cannot find or download homunc");
}

fn compile_hom_files(homunc: &PathBuf) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let Ok(entries) = std::fs::read_dir("src") else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "hom") {
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let rs_path = out_dir.join(format!("{}.rs", stem));

            // Only recompile if .hom is newer than .rs
            let needs_compile = !rs_path.exists()
                || std::fs::metadata(&path)
                    .and_then(|hom| {
                        std::fs::metadata(&rs_path)
                            .map(|rs| hom.modified().unwrap() > rs.modified().unwrap())
                    })
                    .unwrap_or(true);

            if needs_compile {
                let status = Command::new(homunc)
                    .args([
                        "--module",
                        &path.to_string_lossy(),
                        "-o",
                        &rs_path.to_string_lossy(),
                    ])
                    .status();
                match status {
                    Ok(s) if s.success() => {
                        println!(
                            "cargo:warning=Compiled {} -> {}",
                            path.display(),
                            rs_path.display()
                        );
                    }
                    Ok(s) => {
                        println!(
                            "cargo:warning=homunc failed for {} (exit code {:?})",
                            path.display(),
                            s.code()
                        );
                    }
                    Err(e) => {
                        println!("cargo:warning=homunc error for {}: {}", path.display(), e);
                    }
                }
            }
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Strip `#[cfg(test)] mod tests { ... }` blocks from Rust source.
/// Handles nested braces correctly by counting brace depth.
fn strip_test_modules(src: &str) -> String {
    let mut result = String::with_capacity(src.len());
    let mut lines = src.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed == "#[cfg(test)]"
            && let Some(&next) = lines.peek()
            && next.trim().starts_with("mod tests")
        {
            let mod_line = lines.next().unwrap();
            let mut depth: i32 = mod_line.chars().filter(|&c| c == '{').count() as i32
                - mod_line.chars().filter(|&c| c == '}').count() as i32;
            while depth > 0 {
                if let Some(inner) = lines.next() {
                    depth += inner.chars().filter(|&c| c == '{').count() as i32;
                    depth -= inner.chars().filter(|&c| c == '}').count() as i32;
                } else {
                    break;
                }
            }
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}
