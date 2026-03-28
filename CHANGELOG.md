# Changelog

## v0.16 — Embedded Runtime + Examples

- Use `homunc --emit-runtime` instead of `src/hom` submodule — runtime is now embedded in the compiler
- Remove `hom-std` submodule dependency
- Simplify `build.rs`: no more manual file concatenation
- Move examples to `_site/examples/` for web playground
- Add example dropdown selector to playground (like Homun Playground)
- Update GitHub URL to `homun-lang/mermaid-ascii`
- Requires homunc v0.79.1+

## v0.15 — Kill `Rc<RefCell<>>`

- Eliminate all `Rc<RefCell<>>` wrapper types in `layout_state.rs`, replacing with plain structs and return-value mutation
- Convert 11 types: DegMap, NodeSet, StrList, EdgePairList, PosMap, FloatMap, IntList, EdgeInfoList, OrderingList, DummyEdgeList, MutableGraph
- Remove ~900 lines of boilerplate wrapper code
- Clean up dead code (`float_map_new`, `float_map_get_or_inf`)

## v0.14 — Layout IR Refactor

- Refactor layout pipeline into dedicated layout IR modules
- Introduce `::` mutable-reference param convention (following `pathfinder.hom` pattern)
- Add `.hom` source files for layout modules
- Remove duplicated code, clean up module structure

## v0.13 — Syntax Upgrade

- Upgrade all Homun source files to latest `homunc` syntax
- Adopt `::` namespace operator throughout codebase

## v0.12 — SVG Renderer

- Add real geometry-based SVG renderer (`render_svg_dsl`)
- Add `--svg` CLI flag to `main.rs`
- Update `gen.sh` to generate and verify SVG golden files

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
