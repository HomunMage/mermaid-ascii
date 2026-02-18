"""CLI entry point for mermaid-ascii."""

import sys

import click

from mermaid_ascii.layout.engine import full_layout_with_padding
from mermaid_ascii.parsers.registry import parse
from mermaid_ascii.renderers.ascii import AsciiRenderer
from mermaid_ascii.syntax.graph import GraphIR
from mermaid_ascii.types import Direction

_DIRECTION_MAP: dict[str, Direction] = {
    "LR": Direction.LR,
    "RL": Direction.RL,
    "TD": Direction.TD,
    "TB": Direction.TD,
    "BT": Direction.BT,
}


@click.command()
@click.argument("input", required=False, type=click.Path(exists=True))
@click.option("--ascii", "-a", "use_ascii", is_flag=True, help="Use plain ASCII instead of Unicode")
@click.option("--direction", "-d", "direction", type=str, default=None, help="Override direction (LR, RL, TD, BT)")
@click.option("--padding", "-p", "padding", type=int, default=1, help="Node padding (spaces inside border)")
@click.option("--output", "-o", "output", type=str, default=None, help="Write output to this file instead of stdout")
def main(input: str | None, use_ascii: bool, direction: str | None, padding: int, output: str | None) -> None:
    """Mermaid flowchart to ASCII/Unicode graph output."""
    if input:
        try:
            with open(input) as f:
                text = f.read()
        except OSError as e:
            click.echo(f"error: cannot read '{input}': {e}", err=True)
            sys.exit(1)
    else:
        text = sys.stdin.read()

    try:
        ast_graph = parse(text)
    except ValueError as e:
        click.echo(f"parse error:\n{e}", err=True)
        sys.exit(1)

    if direction is not None:
        key = direction.upper()
        if key not in _DIRECTION_MAP:
            click.echo(f"error: unknown direction '{direction}'; use LR, RL, TD, or BT", err=True)
            sys.exit(1)
        ast_graph.direction = _DIRECTION_MAP[key]

    gir = GraphIR.from_ast(ast_graph)

    if gir.node_count() == 0 and not gir.subgraph_members:
        if output:
            try:
                with open(output, "w") as f:
                    f.write("")
            except OSError as e:
                click.echo(f"error: cannot write '{output}': {e}", err=True)
                sys.exit(1)
        return

    layout_nodes, routed_edges = full_layout_with_padding(gir, padding)
    renderer = AsciiRenderer(unicode=not use_ascii)
    rendered = renderer.render(gir, layout_nodes, routed_edges)

    if output:
        try:
            with open(output, "w") as f:
                f.write(rendered)
        except OSError as e:
            click.echo(f"error: cannot write '{output}': {e}", err=True)
            sys.exit(1)
    else:
        click.echo(rendered, nl=False)


if __name__ == "__main__":
    main()
