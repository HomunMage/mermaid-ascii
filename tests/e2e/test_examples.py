"""Tests that verify example outputs match .expect golden files."""

from pathlib import Path

import pytest

from mermaid_ascii import render_dsl

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


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


@pytest.fixture(autouse=True, scope="session")
def clean_out_files():
    """Delete all *.out.txt before and after the test session."""
    for f in EXAMPLES_DIR.rglob("*.out.txt"):
        f.unlink()
    yield
    for f in EXAMPLES_DIR.rglob("*.out.txt"):
        f.unlink()


@pytest.mark.parametrize("name,mm_file,expect_file", EXAMPLE_PAIRS, ids=[p[0] for p in EXAMPLE_PAIRS])
def test_example_matches_expect(name: str, mm_file: Path, expect_file: Path) -> None:
    """Render a .mm.md file and compare output against .expect golden file."""
    src = mm_file.read_text()
    expected = expect_file.read_text()
    actual = render_dsl(src)
    assert actual == expected, f"Output for {name} differs from .expect"
