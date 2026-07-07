"""Unit tests for rrg_engine — offline, using synthetic price data."""

import numpy as np
import pandas as pd
import pytest

from rrg_engine import (
    BENCHMARK, TAIL_LENGTH, classify_quadrant, compute_rrg, get_tails,
)


@pytest.fixture
def synthetic_prices():
    """~3 years of weekly prices: benchmark plus assets as random walks."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-06", periods=160, freq="W-FRI")
    data = {BENCHMARK: 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(idx))))}
    for sym in ["AAA", "BBB", "CCC"]:
        data[sym] = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(idx))))
    return pd.DataFrame(data, index=idx)


def test_rs_ratio_mean_near_100(synthetic_prices):
    """RS-Ratio is a z-score offset around 100, so its mean must stay near 100."""
    rrg = compute_rrg(synthetic_prices)
    for sym in ["AAA", "BBB", "CCC"]:
        mean = rrg[(sym, "ratio")].dropna().mean()
        assert abs(mean - 100) < 1.0, f"{sym} RS-Ratio mean {mean:.2f} not near 100"
        mom_mean = rrg[(sym, "momentum")].dropna().mean()
        assert abs(mom_mean - 100) < 1.0, f"{sym} RS-Momentum mean {mom_mean:.2f}"


def test_benchmark_excluded(synthetic_prices):
    rrg = compute_rrg(synthetic_prices)
    assert BENCHMARK not in rrg.columns.get_level_values("symbol")


def test_flat_relative_strength_is_centered(synthetic_prices):
    """An asset that exactly tracks the benchmark has RS constant; ratio should
    be NaN or ~100 (zero std), never diverge."""
    prices = synthetic_prices.copy()
    prices["SAME"] = prices[BENCHMARK] * 2.0
    rrg = compute_rrg(prices)
    ratio = rrg[("SAME", "ratio")].dropna()
    # constant RS -> std 0 -> division yields inf/NaN; all finite values ≈ 100
    finite = ratio[np.isfinite(ratio)]
    assert (finite.sub(100).abs() < 1e-6).all()


def test_tails_shape_and_order(synthetic_prices):
    rrg = compute_rrg(synthetic_prices)
    tails = get_tails(rrg)
    for sym, grp in tails.groupby("symbol"):
        assert len(grp) == TAIL_LENGTH
        assert grp["date"].is_monotonic_increasing
    assert set(tails.columns) == {"symbol", "date", "ratio", "momentum", "quadrant"}


def test_classify_quadrant():
    assert classify_quadrant(101, 101) == "Leading"
    assert classify_quadrant(101, 99) == "Weakening"
    assert classify_quadrant(99, 99) == "Lagging"
    assert classify_quadrant(99, 101) == "Improving"
