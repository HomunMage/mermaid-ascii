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

    // Generate runtime.rs — Homun's builtin + std + re + heap runtime.
    generate_runtime();

    // Compile .hom files → .rs into OUT_DIR (inside target/).
    // Generated .rs never pollute src/. cargo clean removes everything.
    compile_hom_files();
}

/// Generate runtime.rs in OUT_DIR by concatenating .rs files from src/hom/
/// (homun-std submodule). No homunc needed — just cat the .rs files together.
fn generate_runtime() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let runtime_path = out_dir.join("runtime.rs");
    let hom = PathBuf::from("src/hom");

    // builtin.rs — macros (range!, len!, filter!, map!, dict!, set!, slice!, homun_in!),
    // traits (HomunIndex, HomunLen, HomunContains)
    let builtin = std::fs::read_to_string(hom.join("builtin.rs"))
        .expect("src/hom/builtin.rs not found — is the hom submodule initialized?");

    // std/ — standard library (str, math, collection, dict, stack, deque, io)
    // Read mod.rs but strip include!() lines (we inline the sub-files directly)
    let std_mod: String = std::fs::read_to_string(hom.join("std/mod.rs")).unwrap()
        .lines()
        .filter(|l| !l.trim().starts_with("include!("))
        .collect::<Vec<_>>()
        .join("\n");
    let std_str = std::fs::read_to_string(hom.join("std/str.rs")).unwrap();
    let std_math = std::fs::read_to_string(hom.join("std/math.rs")).unwrap();
    let std_collection = std::fs::read_to_string(hom.join("std/collection.rs")).unwrap();
    let std_dict = std::fs::read_to_string(hom.join("std/dict.rs")).unwrap();
    let std_stack = std::fs::read_to_string(hom.join("std/stack.rs")).unwrap();
    let std_deque = std::fs::read_to_string(hom.join("std/deque.rs")).unwrap();
    let std_io = std::fs::read_to_string(hom.join("std/io.rs")).unwrap();

    // re.rs — regex helpers
    let re = std::fs::read_to_string(hom.join("re.rs")).unwrap();
    // heap.rs — BinaryHeap helpers (strip only duplicate RefCell import)
    let heap: String = std::fs::read_to_string(hom.join("heap.rs")).unwrap()
        .lines()
        .filter(|l| l.trim() != "use std::cell::RefCell;")
        .collect::<Vec<_>>()
        .join("\n");

    let code = format!(
        "// ── builtin ────────────────────────────────────────────────\n\
         {builtin}\n\n\
         // ── std ────────────────────────────────────────────────────\n\
         {std_mod}\n{std_str}\n{std_math}\n{std_collection}\n{std_dict}\n{std_stack}\n{std_deque}\n{std_io}\n\n\
         // ── re ─────────────────────────────────────────────────────\n\
         {re}\n\n\
         // ── heap ───────────────────────────────────────────────────\n\
         {heap}\n"
    );

    std::fs::write(&runtime_path, &code).unwrap();
    println!("cargo:warning=Generated runtime.rs ({} bytes) from src/hom/", code.len());

    // Rerun if any hom runtime file changes
    println!("cargo:rerun-if-changed=src/hom/builtin.rs");
    println!("cargo:rerun-if-changed=src/hom/std/mod.rs");
    println!("cargo:rerun-if-changed=src/hom/re.rs");
    println!("cargo:rerun-if-changed=src/hom/heap.rs");
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
    let url = "https://github.com/HomunMage/Homun-Lang/releases/latest/download/homunc-linux-x86_64";
    let status = Command::new("wget")
        .args(["-q", url, "-O", local.to_str().unwrap()])
        .status();
    if let Ok(s) = status {
        if s.success() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&local, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            println!("cargo:warning=Downloaded homunc to .tmp/homunc");
            return local;
        }
    }
    panic!("Cannot find or download homunc");
}

fn compile_hom_files() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let homunc = find_homunc();

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
                let status = Command::new(&homunc)
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
                            path.display(), s.code()
                        );
                    }
                    Err(e) => {
                        println!(
                            "cargo:warning=homunc error for {}: {}",
                            path.display(), e
                        );
                    }
                }
            }
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
