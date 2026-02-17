"""CLI entry point for mermaid-ascii."""

import sys

import click


@click.command()
@click.argument("input", required=False, type=click.Path(exists=True))
@click.option("--ascii", "-a", "use_ascii", is_flag=True, help="Use plain ASCII characters instead of Unicode box-drawing")
@click.option("--direction", "-d", "direction", type=str, default=None, help="Override graph direction (LR, RL, TD, BT)")
@click.option("--padding", "-p", "padding", type=int, default=1, help="Node padding (spaces inside node border on each side)")
@click.option("--output", "-o", "output", type=str, default=None, help="Write output to this file instead of stdout")
def main(input: str | None, use_ascii: bool, direction: str | None, padding: int, output: str | None) -> None:
    """Mermaid flowchart to ASCII/Unicode graph output."""
    # Read input from file or stdin
    if input:
        with open(input) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    click.echo("not implemented", err=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
