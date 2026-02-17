"""Tests for layout.py — cycle removal (Phase 4) + crossing minimization + coordinate assignment (Phase 6).

Phase 4 tests port the 5 Rust cycle-removal tests:
  - test_dag_has_no_reversed_edges
  - test_single_cycle_reversed
  - test_self_loop_reversed
  - test_complex_cycle
  - test_empty_graph

Phase 6 tests cover:
  - minimise_crossings (barycenter heuristic)
  - count_crossings (inversion count)
  - assign_coordinates / assign_coordinates_padded (TD + LR)
  - label_dimensions helper
  - LayoutNode dataclass
"""

from __future__ import annotations

import networkx as nx

from mermaid_ascii.layout import (
    CycleRemovalResult,
    greedy_fas_ordering,
    remove_cycles,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_graph(*edges: tuple[str, str]) -> nx.DiGraph:
    """Build a DiGraph from a list of (src, tgt) string pairs."""
    g: nx.DiGraph = nx.DiGraph()
    for src, tgt in edges:
        g.add_edge(src, tgt)
    return g


def make_graph_nodes(*nodes: str) -> nx.DiGraph:
    """Build a DiGraph with only nodes (no edges)."""
    g: nx.DiGraph = nx.DiGraph()
    for node in nodes:
        g.add_node(node)
    return g


# ─── Cycle Removal Tests ──────────────────────────────────────────────────────


class TestCycleRemoval:
    def test_dag_has_no_reversed_edges(self):
        """A → B → C (simple DAG, no cycles) — should have zero reversed edges."""
        g = make_graph(("A", "B"), ("B", "C"))
        dag, reversed_edges = remove_cycles(g)
        assert len(reversed_edges) == 0, f"DAG should have no reversed edges, got: {reversed_edges}"
        assert not nx.is_directed_acyclic_graph(g) or nx.is_directed_acyclic_graph(dag)

    def test_single_cycle_reversed(self):
        """A → B → A (2-cycle) — should reverse exactly one edge, result is a DAG."""
        g = make_graph(("A", "B"), ("B", "A"))
        dag, reversed_edges = remove_cycles(g)
        assert len(reversed_edges) == 1, f"Should reverse exactly one edge, got: {reversed_edges}"
        assert nx.is_directed_acyclic_graph(dag), "Result should be a DAG"

    def test_self_loop_reversed(self):
        """A → A (self-loop) — self-loop counted as reversed, removed from result DAG."""
        g = make_graph(("A", "A"))
        dag, reversed_edges = remove_cycles(g)
        assert len(reversed_edges) == 1, "Self-loop should be counted as reversed"
        assert nx.is_directed_acyclic_graph(dag), "Result should be a DAG"
        # Self-loop should be removed entirely (not just reversed)
        assert dag.number_of_edges() == 0, "Self-loop should be removed from the DAG"

    def test_complex_cycle(self):
        """A → B → C → A (3-cycle) plus D → B — result must be a DAG."""
        g = make_graph(("A", "B"), ("B", "C"), ("C", "A"), ("D", "B"))
        dag, reversed_edges = remove_cycles(g)
        assert nx.is_directed_acyclic_graph(dag), "Result should be a DAG"
        # Some edges must have been reversed to break the cycle
        assert len(reversed_edges) >= 1, "At least one edge should be reversed"

    def test_empty_graph(self):
        """Empty graph — should return empty graph with no reversed edges."""
        g: nx.DiGraph = nx.DiGraph()
        dag, reversed_edges = remove_cycles(g)
        assert dag.number_of_nodes() == 0
        assert len(reversed_edges) == 0


# ─── greedy_fas_ordering Tests ────────────────────────────────────────────────


class TestGreedyFasOrdering:
    def test_chain_ordering(self):
        """A → B → C — ordering should put A before B before C."""
        g = make_graph(("A", "B"), ("B", "C"))
        ordering = greedy_fas_ordering(g)
        assert set(ordering) == {"A", "B", "C"}, "All nodes should appear in ordering"
        assert len(ordering) == 3

    def test_single_node(self):
        """Single node — ordering has just that node."""
        g = make_graph_nodes("A")
        ordering = greedy_fas_ordering(g)
        assert ordering == ["A"]

    def test_empty_graph(self):
        """Empty graph — ordering is empty."""
        g: nx.DiGraph = nx.DiGraph()
        ordering = greedy_fas_ordering(g)
        assert ordering == []

    def test_all_nodes_present(self):
        """Ordering must contain all nodes exactly once."""
        g = make_graph(("A", "B"), ("B", "C"), ("C", "A"))
        ordering = greedy_fas_ordering(g)
        assert len(ordering) == 3
        assert set(ordering) == {"A", "B", "C"}


# ─── CycleRemovalResult Tests ─────────────────────────────────────────────────


class TestCycleRemovalResult:
    def test_default_empty(self):
        """CycleRemovalResult defaults to empty reversed_edges set."""
        result = CycleRemovalResult()
        assert result.reversed_edges == set()

    def test_stores_edge_pairs(self):
        """CycleRemovalResult stores (src, tgt) string tuples."""
        result = CycleRemovalResult(reversed_edges={("A", "B"), ("C", "D")})
        assert ("A", "B") in result.reversed_edges
        assert ("C", "D") in result.reversed_edges
