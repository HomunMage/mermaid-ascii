"""Graph IR — converts AST into a networkx DiGraph for layout and analysis.

This module owns the canonical graph data structure used by all downstream
phases (layout, routing, rendering). It flattens subgraphs into the main
node/edge lists while preserving subgraph membership for later rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from mermaid_ascii.syntax import types as ast
from mermaid_ascii.types import EdgeType, NodeShape


@dataclass
class NodeData:
    id: str
    label: str
    shape: NodeShape
    attrs: list[ast.Attr]
    subgraph: str | None = None


@dataclass
class EdgeData:
    edge_type: EdgeType
    label: str | None
    attrs: list[ast.Attr]


class GraphIR:
    """The graph intermediate representation built from an AST Graph.

    Wraps a networkx DiGraph and exposes helpers for topology queries.
    """

    def __init__(
        self,
        digraph: nx.DiGraph,
        direction: object,
        subgraph_members: list[tuple[str, list[str]]],
        subgraph_descriptions: dict[str, str],
    ) -> None:
        self.digraph = digraph
        self.direction = direction
        self.subgraph_members = subgraph_members
        self.subgraph_descriptions = subgraph_descriptions

    @classmethod
    def from_ast(cls, ast_graph: ast.Graph) -> GraphIR:
        """Build a GraphIR from an AST Graph."""
        digraph: nx.DiGraph = nx.DiGraph()
        subgraph_members: list[tuple[str, list[str]]] = []

        sg_names: set[str] = {sg.name for sg in ast_graph.subgraphs}

        for node in ast_graph.nodes:
            if node.id not in sg_names:
                _add_node_if_absent(digraph, node, subgraph_name=None)

        for sg in ast_graph.subgraphs:
            _collect_subgraph(sg, digraph, subgraph_members)

        for edge in ast_graph.edges:
            _ensure_node(digraph, edge.from_id)
            _ensure_node(digraph, edge.to_id)
            _add_edge(digraph, edge)

        for sg in ast_graph.subgraphs:
            _collect_subgraph_edges(sg, digraph)

        subgraph_descriptions: dict[str, str] = {}
        for sg in ast_graph.subgraphs:
            if sg.description is not None:
                subgraph_descriptions[sg.name] = sg.description

        return cls(
            digraph=digraph,
            direction=ast_graph.direction,
            subgraph_members=subgraph_members,
            subgraph_descriptions=subgraph_descriptions,
        )

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.digraph)

    def topological_order(self) -> list[str] | None:
        try:
            order = list(nx.topological_sort(self.digraph))
            return [self.digraph.nodes[n]["data"].id for n in order]
        except nx.NetworkXUnfeasible:
            return None

    def node_count(self) -> int:
        return self.digraph.number_of_nodes()

    def edge_count(self) -> int:
        return self.digraph.number_of_edges()

    def in_degree(self, node_id: str) -> int:
        if node_id not in self.digraph:
            return 0
        return self.digraph.in_degree(node_id)

    def out_degree(self, node_id: str) -> int:
        if node_id not in self.digraph:
            return 0
        return self.digraph.out_degree(node_id)

    def adjacency_list(self) -> list[tuple[str, list[str]]]:
        result: list[tuple[str, list[str]]] = []
        for node_id in self.digraph.nodes:
            neighbors = sorted(self.digraph.successors(node_id))
            result.append((node_id, neighbors))
        result.sort(key=lambda x: x[0])
        return result


def _add_node_if_absent(
    digraph: nx.DiGraph,
    ast_node: ast.Node,
    subgraph_name: str | None,
) -> None:
    if ast_node.id not in digraph:
        data = NodeData(
            id=ast_node.id,
            label=ast_node.label,
            shape=ast_node.shape,
            attrs=list(ast_node.attrs),
            subgraph=subgraph_name,
        )
        digraph.add_node(ast_node.id, data=data)


def _ensure_node(digraph: nx.DiGraph, node_id: str) -> None:
    if node_id not in digraph:
        data = NodeData(
            id=node_id,
            label=node_id,
            shape=NodeShape.Rectangle,
            attrs=[],
            subgraph=None,
        )
        digraph.add_node(node_id, data=data)


def _add_edge(digraph: nx.DiGraph, edge: ast.Edge) -> None:
    data = EdgeData(
        edge_type=edge.edge_type,
        label=edge.label,
        attrs=list(edge.attrs),
    )
    digraph.add_edge(edge.from_id, edge.to_id, data=data)


def _collect_subgraph(
    sg: ast.Subgraph,
    digraph: nx.DiGraph,
    subgraph_members: list[tuple[str, list[str]]],
) -> None:
    member_ids: list[str] = []
    for node in sg.nodes:
        _add_node_if_absent(digraph, node, subgraph_name=sg.name)
        member_ids.append(node.id)
    subgraph_members.append((sg.name, member_ids))
    for nested in sg.subgraphs:
        _collect_subgraph(nested, digraph, subgraph_members)


def _collect_subgraph_edges(sg: ast.Subgraph, digraph: nx.DiGraph) -> None:
    for edge in sg.edges:
        _ensure_node(digraph, edge.from_id)
        _ensure_node(digraph, edge.to_id)
        _add_edge(digraph, edge)
    for nested in sg.subgraphs:
        _collect_subgraph_edges(nested, digraph)
