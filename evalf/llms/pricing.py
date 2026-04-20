"""Static pricing registry used to estimate judge-call cost in USD."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class PriceBand:
    """One pricing tier keyed by an optional maximum input-token bound."""

    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    max_input_tokens: int | None = None


@dataclass(frozen=True)
class ModelPricing:
    """Pricing table for a normalized model identifier."""

    bands: tuple[PriceBand, ...]


def _build_pricing_registry(
    entries: tuple[tuple[str, tuple[PriceBand, ...]], ...],
) -> dict[str, ModelPricing]:
    """Build the model-pricing lookup table from normalized model identifiers."""

    return {model: ModelPricing(bands=bands) for model, bands in entries}


PRICING_REGISTRY: dict[str, ModelPricing] = _build_pricing_registry(
    (
        # OpenAI
        (
            "gpt-5.4",
            (
                PriceBand(Decimal("2.50"), Decimal("15.00"), 272000),
                PriceBand(Decimal("5.00"), Decimal("22.50")),
            ),
        ),
        (
            "gpt-5.4-2026-03-05",
            (
                PriceBand(Decimal("2.50"), Decimal("15.00"), 272000),
                PriceBand(Decimal("5.00"), Decimal("22.50")),
            ),
        ),
        (
            "gpt-5.4-pro",
            (
                PriceBand(Decimal("30.00"), Decimal("180.00"), 272000),
                PriceBand(Decimal("60.00"), Decimal("270.00")),
            ),
        ),
        (
            "gpt-5.4-pro-2026-03-05",
            (
                PriceBand(Decimal("30.00"), Decimal("180.00"), 272000),
                PriceBand(Decimal("60.00"), Decimal("270.00")),
            ),
        ),
        ("gpt-5.4-mini", (PriceBand(Decimal("0.75"), Decimal("4.50")),)),
        (
            "gpt-5.4-mini-2026-03-17",
            (PriceBand(Decimal("0.75"), Decimal("4.50")),),
        ),
        ("gpt-5.4-nano", (PriceBand(Decimal("0.20"), Decimal("1.25")),)),
        (
            "gpt-5.4-nano-2026-03-17",
            (PriceBand(Decimal("0.20"), Decimal("1.25")),),
        ),
        ("gpt-4.1", (PriceBand(Decimal("2.00"), Decimal("8.00")),)),
        ("gpt-4.1-mini", (PriceBand(Decimal("0.40"), Decimal("1.60")),)),
        ("gpt-4.1-nano", (PriceBand(Decimal("0.10"), Decimal("0.40")),)),
        ("gpt-4o-mini", (PriceBand(Decimal("0.15"), Decimal("0.60")),)),
        # Anthropic / Claude
        ("claude-opus-4.6", (PriceBand(Decimal("5.00"), Decimal("25.00")),)),
        ("claude-opus-4-6", (PriceBand(Decimal("5.00"), Decimal("25.00")),)),
        ("claude-opus-4.1", (PriceBand(Decimal("15.00"), Decimal("75.00")),)),
        ("claude-opus-4-1-20250805", (PriceBand(Decimal("15.00"), Decimal("75.00")),)),
        ("claude-opus-4.5", (PriceBand(Decimal("5.00"), Decimal("25.00")),)),
        ("claude-opus-4-5-20251101", (PriceBand(Decimal("5.00"), Decimal("25.00")),)),
        ("claude-sonnet-4", (PriceBand(Decimal("3.00"), Decimal("15.00")),)),
        ("claude-sonnet-4-0", (PriceBand(Decimal("3.00"), Decimal("15.00")),)),
        ("claude-sonnet-4-20250514", (PriceBand(Decimal("3.00"), Decimal("15.00")),)),
        ("claude-sonnet-4.5", (PriceBand(Decimal("3.00"), Decimal("15.00")),)),
        ("claude-sonnet-4-5", (PriceBand(Decimal("3.00"), Decimal("15.00")),)),
        ("claude-sonnet-4-5-20250929", (PriceBand(Decimal("3.00"), Decimal("15.00")),)),
        ("claude-sonnet-4.6", (PriceBand(Decimal("3.00"), Decimal("15.00")),)),
        ("claude-sonnet-4-6", (PriceBand(Decimal("3.00"), Decimal("15.00")),)),
        ("claude-haiku-4.5", (PriceBand(Decimal("1.00"), Decimal("5.00")),)),
        ("claude-haiku-4-5", (PriceBand(Decimal("1.00"), Decimal("5.00")),)),
        ("claude-haiku-4-5-20251001", (PriceBand(Decimal("1.00"), Decimal("5.00")),)),
        ("claude-haiku-3.5", (PriceBand(Decimal("0.80"), Decimal("4.00")),)),
        ("claude-3-5-haiku-20241022", (PriceBand(Decimal("0.80"), Decimal("4.00")),)),
        # Google Gemini
        (
            "gemini-2.5-pro",
            (
                PriceBand(Decimal("1.25"), Decimal("10.00"), 200000),
                PriceBand(Decimal("2.50"), Decimal("15.00")),
            ),
        ),
        ("gemini-2.5-flash", (PriceBand(Decimal("0.30"), Decimal("2.50")),)),
        ("gemini-2.5-flash-lite", (PriceBand(Decimal("0.10"), Decimal("0.40")),)),
        ("gemini-3-flash-preview", (PriceBand(Decimal("0.50"), Decimal("3.00")),)),
        (
            "gemini-3-pro-preview",
            (
                PriceBand(Decimal("2.00"), Decimal("12.00"), 200000),
                PriceBand(Decimal("4.00"), Decimal("18.00")),
            ),
        ),
        (
            "gemini-3.1-pro-preview",
            (
                PriceBand(Decimal("2.00"), Decimal("12.00"), 200000),
                PriceBand(Decimal("4.00"), Decimal("18.00")),
            ),
        ),
        (
            "gemini-3.1-pro-preview-customtools",
            (
                PriceBand(Decimal("2.00"), Decimal("12.00"), 200000),
                PriceBand(Decimal("4.00"), Decimal("18.00")),
            ),
        ),
        (
            "gemini-3.1-flash-lite-preview",
            (PriceBand(Decimal("0.25"), Decimal("1.50")),),
        ),
    ),
)


def estimate_cost_usd(
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Estimate request cost in USD when both token counts are available."""

    if input_tokens is None or output_tokens is None:
        return None

    pricing = PRICING_REGISTRY.get(model.strip().lower())
    if pricing is None:
        return None

    band = pricing.bands[-1]
    for candidate in pricing.bands:
        if candidate.max_input_tokens is None or input_tokens <= candidate.max_input_tokens:
            band = candidate
            break

    total = Decimal(input_tokens) * band.input_usd_per_million / Decimal(1_000_000) + Decimal(
        output_tokens
    ) * band.output_usd_per_million / Decimal(1_000_000)
    return float(total.quantize(Decimal("0.00000001")))
