"""E2E tests for the Rust binary — compiles binary and verifies output against .expect golden files."""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
RUST_DIR = REPO_ROOT / "src" / "rust"
BINARY_PATH = RUST_DIR / "target" / "release" / "mermaid-ascii"


def find_example_pairs() -> list[tuple[str, Path, Path]]:
    """Find all .mm.md files that have a matching .expect file."""
    pairs = []
    for mm_file in sorted(EXAMPLES_DIR.glob("*.mm.md")):
        expect_file = mm_file.with_suffix("").with_suffix(".expect")
        if expect_file.exists():
            name = mm_file.stem.replace(".mm", "")
            pairs.append((name, mm_file, expect_file))
    return pairs


EXAMPLE_PAIRS = find_example_pairs()


@pytest.fixture(scope="session")
def rust_binary() -> Path:
    """Build the Rust binary (release mode) once per session and return its path."""
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=RUST_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"cargo build --release failed:\n{result.stderr}")
    assert BINARY_PATH.exists(), f"Binary not found at {BINARY_PATH}"
    return BINARY_PATH


def run_binary(binary: Path, input_text: str, *extra_args: str) -> str:
    """Run the Rust binary with the given input and return stdout."""
    result = subprocess.run(
        [str(binary), *extra_args],
        input=input_text,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Binary exited {result.returncode}:\n{result.stderr}")
    return result.stdout


@pytest.mark.parametrize("name,mm_file,expect_file", EXAMPLE_PAIRS, ids=[p[0] for p in EXAMPLE_PAIRS])
def test_example_matches_expect(name: str, mm_file: Path, expect_file: Path, rust_binary: Path) -> None:
    """Run Rust binary on each .mm.md example and compare output against .expect golden file."""
    src = mm_file.read_text()
    expected = expect_file.read_text()
    actual = run_binary(rust_binary, src)
    # The binary doesn't append a trailing newline; add one to match .expect files if needed
    if expected.endswith("\n") and not actual.endswith("\n"):
        actual += "\n"
    assert actual == expected, f"Rust output for '{name}' differs from .expect"


def test_ascii_flag(rust_binary: Path) -> None:
    """--ascii flag produces plain ASCII box-drawing characters (no Unicode)."""
    src = "graph TD\n    A --> B\n"
    output = run_binary(rust_binary, src, "--ascii")
    # Unicode box chars must not appear
    assert "┌" not in output
    assert "│" not in output
    assert "─" not in output
    # Plain ASCII box chars must appear
    assert "+" in output or "|" in output or "-" in output


def test_direction_override_lr(rust_binary: Path) -> None:
    """--direction LR overrides the direction declared in the diagram."""
    src = "graph TD\n    A --> B --> C\n"
    output_td = run_binary(rust_binary, src)
    output_lr = run_binary(rust_binary, src, "--direction", "LR")
    # LR layout should be wider than TD layout (more columns, fewer rows)
    lines_td = [line for line in output_td.splitlines() if line.strip()]
    lines_lr = [line for line in output_lr.splitlines() if line.strip()]
    width_td = max(len(line) for line in lines_td)
    width_lr = max(len(line) for line in lines_lr)
    assert width_lr > width_td, "LR output should be wider than TD output"
    assert len(lines_lr) < len(lines_td), "LR output should have fewer lines than TD output"


def test_direction_override_bt(rust_binary: Path) -> None:
    """--direction BT produces bottom-to-top layout (first node at bottom)."""
    src = "graph TD\n    A --> B\n"
    output = run_binary(rust_binary, src, "--direction", "BT")
    # BT means B (target) is above A (source) — B should appear earlier in output
    lines = output.splitlines()
    b_line = next((i for i, line in enumerate(lines) if "B" in line), None)
    a_line = next((i for i, line in enumerate(lines) if "A" in line), None)
    assert b_line is not None and a_line is not None
    assert b_line < a_line, "In BT layout, B (target) should appear above A (source)"


def test_reads_from_file(rust_binary: Path, tmp_path: Path) -> None:
    """Binary reads from a file path argument instead of stdin."""
    src = "graph LR\n    X --> Y\n"
    input_file = tmp_path / "test.mm.md"
    input_file.write_text(src)
    result = subprocess.run(
        [str(rust_binary), str(input_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "X" in result.stdout
    assert "Y" in result.stdout


def test_output_to_file(rust_binary: Path, tmp_path: Path) -> None:
    """--output flag writes result to file instead of stdout."""
    src = "graph TD\n    A --> B\n"
    out_file = tmp_path / "out.txt"
    result = subprocess.run(
        [str(rust_binary), "--output", str(out_file)],
        input=src,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out_file.exists()
    content = out_file.read_text()
    assert "A" in content
    assert "B" in content
