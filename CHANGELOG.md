# Changelog

## v0.10 — Homun + Rust

- Restructure to Homun (.hom) + Rust architecture
- Full Sugiyama layout pipeline in hand-written Rust (`src/lib.rs`)
- Hand-written `graph/` module: petgraph wrapper, `Rc<RefCell<...>>` mutable state types
- Homun modules: types, config, canvas, charset, pathfinder, parser, layout
- `build.rs` compiles `.hom` → `.rs` via `homunc` at build time
- Add `#[wasm_bindgen]` exports (`render`, `renderWithOptions`, `renderSvg`)
- 35 tests passing

## v0.5 — SVG Renderer

- Add SVG output mode to playground (ASCII + SVG tabs)

## v0.4 — A* Edge Routing

- Port A* pathfinding edge routing from Python to Rust

## v0.3 — Full Rust Port + CI/CD

- Complete Python → Rust port (1:1 module map)
- Parser: recursive descent tokenizer + flowchart parser
- GraphIR: petgraph DiGraph wrapper with `from_ast()`
- Sugiyama layout engine: cycle removal, layering, crossing minimization, coordinates, routing
- ASCII renderer: shape-aware box drawing, edge painting, direction transforms
- API + CLI: `render_dsl()`, clap with `--ascii`, `--direction`, `--padding`, `--output`
- E2E tests: Python pytest against Rust binary (golden file comparison)
- CI/CD: cross-platform binaries (linux x86_64/aarch64, windows) + WASM tarball
- GitHub Pages playground with interactive WASM demo

## v0.2 — Python Package + PyPI

- CI/CD: GitHub Actions for test, build, release
- Dockerfile multi-stage build, PyPI publishing

## v0.1 — Python Implementation

- Recursive descent parser, GraphIR (networkx), Sugiyama layout, ASCII/Unicode renderer
- `render_dsl()` public API, 232 Python tests
