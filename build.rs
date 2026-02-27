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

    // Compile .hom files → .rs into OUT_DIR (inside target/).
    // Generated .rs never pollute src/. cargo clean removes everything.
    compile_hom_files();
}

fn compile_hom_files() {
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
                let status = Command::new("homunc")
                    .args([
                        "--raw",
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
                    _ => {
                        // homunc not available or compilation failed — skip
                        println!(
                            "cargo:warning=homunc skipped for {} (not available or failed)",
                            path.display()
                        );
                    }
                }
            }
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
