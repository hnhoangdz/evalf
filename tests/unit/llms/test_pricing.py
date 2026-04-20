import pytest

from evalf.llms.pricing import estimate_cost_usd

pytestmark = pytest.mark.unit


def test_estimate_cost_usd_returns_number_for_known_model() -> None:
    cost = estimate_cost_usd(model="gpt-4.1-mini", input_tokens=1000, output_tokens=500)
    assert cost is not None


def test_estimate_cost_usd_returns_none_without_usage() -> None:
    cost = estimate_cost_usd(model="gpt-4.1-mini", input_tokens=None, output_tokens=500)
    assert cost is None


def test_estimate_cost_usd_returns_none_for_unknown_models() -> None:
    cost = estimate_cost_usd(model="unknown-model", input_tokens=1000, output_tokens=500)
    assert cost is None


def test_estimate_cost_usd_uses_the_correct_pricing_band() -> None:
    low_band_cost = estimate_cost_usd(
        model="gemini-2.5-pro",
        input_tokens=200000,
        output_tokens=1000,
    )
    high_band_cost = estimate_cost_usd(
        model="gemini-2.5-pro",
        input_tokens=200001,
        output_tokens=1000,
    )

    assert low_band_cost == 0.26
    assert high_band_cost == 0.5150025


def test_estimate_cost_usd_supports_latest_openai_snapshot_ids() -> None:
    cost = estimate_cost_usd(model="gpt-5.4-pro-2026-03-05", input_tokens=1000, output_tokens=500)

    assert cost == 0.12


def test_estimate_cost_usd_uses_openai_long_context_pricing_band() -> None:
    low_band_cost = estimate_cost_usd(
        model="gpt-5.4",
        input_tokens=272000,
        output_tokens=1000,
    )
    high_band_cost = estimate_cost_usd(
        model="gpt-5.4",
        input_tokens=272001,
        output_tokens=1000,
    )

    assert low_band_cost == 0.695
    assert high_band_cost == 1.382505


def test_estimate_cost_usd_supports_latest_gemini_preview_models() -> None:
    cost = estimate_cost_usd(
        model="gemini-3.1-pro-preview-customtools",
        input_tokens=1000,
        output_tokens=500,
    )

    assert cost == 0.008


def test_estimate_cost_usd_supports_latest_claude_models() -> None:
    cost = estimate_cost_usd(
        model="claude-haiku-4-5-20251001",
        input_tokens=1000,
        output_tokens=500,
    )

    assert cost == 0.0035
