"""The code⇄canvas engine (L3).

Code is the single source of truth; a visual canvas is a *projection* of it. This package turns
TensorSketch source into a graph structure a canvas can draw (`extract`), and — in later increments
—
applies canvas edits back to the source surgically, leaving node bodies untouched.

Only the wiring and typed interfaces round-trip; node bodies are opaque (a computability
necessity — Rice's theorem). Extraction is purely syntactic (a CST), so it works on *incomplete*
code: undefined names, missing imports, and unfilled holes are all fine.

Requires the `canvas` extra: `pip install tensorsketch-core[canvas]`.
"""

from .extract import extract
from .holes import HoleRef, find_holes
from .ir import EdgeIR, GraphIR, NodeIR, Port
from .reconstruct import reconstruct

__all__ = [
    "EdgeIR",
    "GraphIR",
    "HoleRef",
    "NodeIR",
    "Port",
    "extract",
    "find_holes",
    "reconstruct",
]
