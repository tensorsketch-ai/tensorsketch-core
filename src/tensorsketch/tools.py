"""Tools: plain functions the model can call, with schemas derived from their signatures.

Write an ordinary (sync or async) function, annotate its parameters, and decorate it with
`@tool`. TensorSketch builds the JSON schema the model needs straight from the signature and the
docstring — no hand-written schema, no boilerplate. Argument values coming back from the model
are validated against that schema before your function runs.

    @tool
    def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, overload

from pydantic import create_model

from .core.schema import Schema

if TYPE_CHECKING:
    from .core.context import Context


class Tool:
    """A callable exposed to a model: a `name`, a `description`, and an argument schema.

    Most tools come from `@tool`, which derives a typed `args_schema` (a `Schema`) from the
    function signature. A tool can also be built from a **raw JSON Schema** by passing
    `json_schema=` and leaving `args_schema` unset — that's how remote tools (e.g. MCP) plug in,
    since their interface arrives as JSON Schema rather than a Python signature. With no schema at
    all, the tool advertises an empty object of arguments.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str,
        description: str,
        args_schema: type[Schema] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self._json_schema = json_schema
        self._wants_ctx = _declares_ctx(fn)

    def json_schema(self) -> dict[str, Any]:
        """The JSON Schema for this tool's arguments (what a provider advertises to the model)."""
        if self._json_schema is not None:
            return self._json_schema
        if self.args_schema is not None:
            return self.args_schema.json_schema()
        return {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx: Context | None = None) -> Any:
        """Validate `args` against the schema (if typed), then invoke the function (async-aware).

        If the function declares a `ctx` parameter, the run `Context` is injected into it — it's
        never part of the model-facing schema. That lets a tool journal durable steps, emit
        events, or run a sub-agent under the same trace.
        """
        call_args = args
        if self.args_schema is not None:
            call_args = self.args_schema(**args).model_dump()
        if self._wants_ctx:
            call_args = {**call_args, "ctx": ctx}
        result = self.fn(**call_args)
        if inspect.isawaitable(result):
            return await result
        return result

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


def _declares_ctx(fn: Callable[..., Any]) -> bool:
    """Whether `fn` takes a `ctx` parameter (so the runtime should inject the `Context`)."""
    try:
        return "ctx" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _args_schema(fn: Callable[..., Any], model_name: str) -> type[Schema]:
    """Build a `Schema` describing a function's call arguments from its signature."""
    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    fields: dict[str, Any] = {}
    for pname, param in signature.parameters.items():
        if pname in ("self", "ctx"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (annotation, default)
    return cast(type[Schema], create_model(model_name, __base__=Schema, **fields))


@overload
def tool(fn: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a `Tool`. Use bare (`@tool`) or configured (`@tool(name=...)`).

    The tool's name defaults to the function name and its description to the docstring; the
    argument schema is inferred from the parameter annotations.
    """

    def make(target: Callable[..., Any]) -> Tool:
        return Tool(
            target,
            name=name or target.__name__,
            description=description or (inspect.getdoc(target) or "").strip(),
            args_schema=_args_schema(target, f"{target.__name__}_Args"),
        )

    return make(fn) if fn is not None else make
