"""Tests that Python SVG output matches .expect.svg golden files."""

from pathlib import Path

import pytest

from mermaid_ascii.api import render_svg

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def find_svg_pairs() -> list[tuple[str, Path, Path]]:
    """Find all .mm.md files that have a matching .expect.svg file."""
    pairs = []
    for mm_file in sorted(EXAMPLES_DIR.glob("*.mm.md")):
        name = mm_file.stem.replace(".mm", "")
        expect_svg = EXAMPLES_DIR / f"{name}.expect.svg"
        if expect_svg.exists():
            pairs.append((name, mm_file, expect_svg))
    return pairs


SVG_PAIRS = find_svg_pairs()


@pytest.mark.parametrize("name,mm_file,expect_svg", SVG_PAIRS, ids=[p[0] for p in SVG_PAIRS])
def test_svg_matches_expect(name: str, mm_file: Path, expect_svg: Path) -> None:
    """Render a .mm.md file to SVG and compare against .expect.svg golden file."""
    src = mm_file.read_text()
    expected = expect_svg.read_text().rstrip("\n")
    actual = render_svg(src).rstrip("\n")
    assert actual == expected, f"SVG output for {name} differs from .expect.svg"
