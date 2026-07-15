"""A tiny local bridge that closes the code⇄canvas loop.

The Studio frontend needs two operations, and this exposes exactly those over HTTP:

* **load** — read the source file, `extract` its `GraphIR`, hand it to the canvas as JSON;
* **save** — take an edited `GraphIR` from the canvas, `reconstruct` it into the source file.

Because *code is the source of truth*, the server holds no graph state of its own: every load
re-reads the file, every save re-writes it and returns the freshly re-extracted graph. It binds
to localhost only — it's a developer tool, not a service. Stdlib only, so it adds no dependency.

    python -m tensorsketch.canvas examples/support_router.py         # then open the printed URL
"""

from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .extract import extract
from .holes import find_holes
from .ir import GraphIR
from .reconstruct import reconstruct

STUDIO = Path(__file__).parent / "studio"

_CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


class TraceBuffer:
    """An ephemeral in-memory tail of live spans for the overlay.

    Holds **no durable state** — it's a view buffer (like a dashboard's live tail), gone when the
    server stops. The trace's system of record is whatever exporter the run sent spans to; this is
    only what the canvas paints. Bounded so a long session can't grow without limit. Thread-safe:
    the run POSTs spans on one thread while the browser polls on another.
    """

    def __init__(self, limit: int = 5000) -> None:
        self._lock = threading.Lock()
        self._records: deque[dict[str, Any]] = deque(maxlen=limit)
        self._seq = 0

    def add(self, span: dict[str, Any]) -> int:
        with self._lock:
            self._seq += 1
            self._records.append({"seq": self._seq, "span": span})
            return self._seq

    def since(self, cursor: int) -> dict[str, Any]:
        with self._lock:
            fresh = [r for r in self._records if r["seq"] > cursor]
            return {"cursor": self._seq, "records": fresh}


def load_graph(path: Path) -> dict[str, Any]:
    """The current graph, extracted fresh from the file (the source of truth)."""
    ir = extract(path.read_text()).to_dict()
    return {"path": str(path), "ir": ir, "layout": load_layout(path)}


# -- layout sidecar ---------------------------------------------------------------------------
#
# Node *positions* aren't part of the graph — they're pure presentation, so they must never touch
# the code (code is the source of truth about the graph, not its picture). Instead a manually
# arranged node is remembered in a sidecar `<file>.layout.json` beside the source. It's optional:
# a node with no saved position falls back to the automatic layered layout, and a stale entry for
# a node that no longer exists is simply ignored. Losing the sidecar loses only the arrangement.


def _layout_path(path: Path) -> Path:
    return path.parent / (path.name + ".layout.json")


def load_layout(path: Path) -> dict[str, Any]:
    """Saved node positions for `path` (empty if there's no sidecar or it's unreadable)."""
    side = _layout_path(path)
    if not side.is_file():
        return {"nodes": {}}
    try:
        data = json.loads(side.read_text())
    except (OSError, ValueError):
        return {"nodes": {}}
    return {"nodes": _clean_positions(data.get("nodes") if isinstance(data, dict) else None)}


def save_layout(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Write the manually-arranged node positions to the sidecar; returns what was stored."""
    positions = _clean_positions(data.get("nodes"))
    _layout_path(path).write_text(json.dumps({"version": 1, "nodes": positions}, indent=2))
    return {"ok": True, "nodes": positions}


def _clean_positions(nodes: Any) -> dict[str, dict[str, float]]:
    """Keep only well-formed `{name: {x, y}}` numeric entries (the sidecar may be hand-edited)."""
    clean: dict[str, dict[str, float]] = {}
    if isinstance(nodes, dict):
        for name, pos in nodes.items():
            if isinstance(pos, dict) and _is_number(pos.get("x")) and _is_number(pos.get("y")):
                clean[str(name)] = {"x": float(pos["x"]), "y": float(pos["y"])}
    return clean


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_holes(path: Path) -> dict[str, Any]:
    """Every node still needing code across the project (the served file's directory tree)."""
    root = path.parent if path.parent != Path("") else Path(".")
    holes = find_holes(root)
    return {"root": str(root), "holes": [h.to_dict() for h in holes]}


def apply_edits(path: Path, ir_data: dict[str, Any]) -> dict[str, Any]:
    """Write an edited graph back to the file and return the re-extracted result.

    Only the wiring is applied; node bodies, imports, and comments are preserved by
    `reconstruct`. Returning the re-extracted IR lets the canvas resync to exactly what the code
    now says — the round-trip in action.
    """
    new_source = reconstruct(path.read_text(), GraphIR.from_dict(ir_data))
    path.write_text(new_source)
    return {"path": str(path), "ir": extract(new_source).to_dict(), "layout": load_layout(path)}


def make_handler(path: Path, traces: TraceBuffer) -> type[BaseHTTPRequestHandler]:
    class StudioHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlparse(self.path)
            if route.path in ("/", "/index.html"):
                self._send_file(STUDIO / "index.html")
            elif route.path == "/api/graph":
                self._send_json(load_graph(path))
            elif route.path == "/api/holes":
                self._send_json(load_holes(path))
            elif route.path == "/api/trace":
                # Live overlay poll: hand back spans newer than the browser's cursor.
                since = int((parse_qs(route.query).get("since", ["0"])[0]) or 0)
                self._send_json(traces.since(since))
            elif route.path.lstrip("/") in {"studio.js", "studio.css"}:
                self._send_file(STUDIO / route.path.lstrip("/"))
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            route = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) or b"{}"
            if route.path == "/api/trace":
                # A running agent pushes each finished span here (via `http_span_sink`).
                try:
                    traces.add(json.loads(body))
                except ValueError:
                    self._send_json({"error": "bad span"}, status=400)
                    return
                self._send_json({"ok": True})
                return
            if route.path == "/api/layout":
                # Persist a manual arrangement to the sidecar. Positions are cosmetic — this never
                # touches the code, so it's a separate route from graph write-back.
                try:
                    self._send_json(save_layout(path, json.loads(body)))
                except ValueError:
                    self._send_json({"error": "bad layout"}, status=400)
                return
            if route.path != "/api/graph":
                self.send_error(404)
                return
            try:
                payload = json.loads(body)
                result = apply_edits(path, payload["ir"])
            except (ValueError, KeyError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

        def _send_file(self, file: Path) -> None:
            if not file.is_file():
                self.send_error(404)
                return
            body = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", _CONTENT_TYPES.get(file.suffix, "text/plain"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # keep the console quiet; the studio is the interface

    return StudioHandler


def serve(path: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the Studio for `path` until interrupted."""
    server = ThreadingHTTPServer((host, port), make_handler(path, TraceBuffer()))
    print(f"TensorSketch Studio → http://{host}:{port}   (editing {path})")
    print("Code is the source of truth; canvas edits are written straight back. Ctrl-C to stop.")
    print(f"Live overlay: feed spans to http://{host}:{port}/api/trace via http_span_sink.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
