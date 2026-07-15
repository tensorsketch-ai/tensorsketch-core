"""The one `Schema` abstraction.

A single typed-value abstraction does four jobs across TensorSketch:

1. **Tool I/O** — a tool's arguments and result are Schemas.
2. **Structured output** — an LLM's typed return value is a Schema.
3. **Typed state channels** — a graph's state is a Schema; each field is a channel.
4. **Typed ports** — a node's `In`/`Out` are Schemas; their fields are the node's ports.

Keeping these unified means one mental model, one validator, and one place to generate JSON
Schema (for providers, canvas rendering, and design-time port checks). `Schema` is a thin
layer over Pydantic v2 — we get its Rust-backed validation for free and add only TensorSketch-facing
conveniences.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    """Base class for every typed value in TensorSketch.

    Subclass it and declare fields the Pydantic way::

        class Query(Schema):
            text: str
            top_k: int = 5

    Unknown fields are rejected (`extra="forbid"`) so a typo in a port name is an error, not a
    silently dropped value.
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for this type (for providers, tools, and the canvas)."""
        return cls.model_json_schema()

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """The declared field names, in declaration order — a Schema's ports."""
        return tuple(cls.model_fields.keys())

    @classmethod
    def field_type(cls, name: str) -> Any:
        """The resolved Python annotation for one field (used for port type-checking)."""
        return cls.model_fields[name].annotation
