"""A small, overridable cost model — tokens → dollars, for spend evaluation.

Prices are USD per **million** tokens, `(input, output)`. The table is deliberately tiny and
easy to override — pass your own to `estimate_cost` — because pricing changes and every
deployment negotiates its own. This is a convenience for the tracer, not a source of truth.
"""

from __future__ import annotations

from collections.abc import Mapping

#: USD per 1M tokens, (input, output). Update or replace freely — see `estimate_cost`.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}


def estimate_cost(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    *,
    prices: Mapping[str, tuple[float, float]] = DEFAULT_PRICES,
) -> float | None:
    """USD cost for a call, or None if the model's price is unknown.

    Matches by exact model id, then by longest known prefix (so dated ids like
    `gpt-4o-2024-08-06` resolve to `gpt-4o`).
    """
    if model is None:
        return None
    rate = prices.get(model)
    if rate is None:
        candidates = [name for name in prices if model.startswith(name)]
        if not candidates:
            return None
        rate = prices[max(candidates, key=len)]
    input_rate, output_rate = rate
    return round((input_tokens * input_rate + output_tokens * output_rate) / 1_000_000, 8)
