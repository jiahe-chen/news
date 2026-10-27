"""Per-token prices (USD per million tokens) for cost accounting."""

from __future__ import annotations

from newsren.llm.base import Usage

# Cache writes bill at 1.25x the input price.
PRICES: dict[str, tuple[float, float, float]] = {
    # model: (input $/MTok, output $/MTok, cache read $/MTok)
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}

DEFAULT_PRICE = PRICES["claude-opus-5-5"]


def cost_usd(model: str | None, usage: Usage) -> float:
    price_in, price_out, price_cache_read = PRICES.get(model or "", DEFAULT_PRICE)
    return (
        usage.input_tokens * price_in
        + usage.cache_write_tokens * price_in * 1.25
        + usage.cache_read_tokens * price_cache_read
        + usage.output_tokens * price_out
    ) / 1_000_000
