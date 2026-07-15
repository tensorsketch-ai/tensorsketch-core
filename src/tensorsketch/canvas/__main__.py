"""`python -m tensorsketch.canvas <file>` — open the TensorSketch Studio on a source file.

The Studio is a visual *projection* of the code: it renders the graph the file defines and
writes edits straight back. Requires the `canvas` extra (`pip install tensorsketch-core[canvas]`).

`python -m tensorsketch.canvas --holes [paths…]` instead lists every node that still needs code (a
`Hole`) across the given files/directories — the "what's left to implement?" view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .holes import find_holes
from .server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tensorsketch.canvas", description="Open the TensorSketch Studio."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="*",
        help="the source file to open (or files/dirs to scan with --holes)",
    )
    parser.add_argument(
        "--holes",
        action="store_true",
        help="list nodes that still need code (a Hole), across the given files/dirs",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.holes:
        return _report_holes(args.path)

    if len(args.path) != 1 or not args.path[0].is_file():
        print("error: provide exactly one source file to open", file=sys.stderr)
        return 2
    serve(args.path[0], host=args.host, port=args.port)
    return 0


def _report_holes(paths: list[Path]) -> int:
    holes = find_holes(*paths)
    if not holes:
        print("No holes — every node has code. ✓")
        return 0
    print(f"{len(holes)} node{'s' if len(holes) != 1 else ''} need code:")
    current: str | None = None
    for hole in holes:
        if hole.file != current:
            current = hole.file
            print(f"\n  {hole.file}")
        spec = f"  — {hole.spec}" if hole.spec else ""
        print(f"    {hole.line:>4}  {hole.node}{spec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
