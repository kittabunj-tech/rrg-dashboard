"""Unit tests for rrg_export — offline, using synthetic engine output."""

import json

import numpy as np
import pandas as pd
import pytest

from rrg_engine import BENCHMARK, TAIL_LENGTH, compute_rrg, get_tails
from rrg_export import build_payload, heading


@pytest.fixture
def payload():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-06", periods=160, freq="W-FRI")
    data = {BENCHMARK: 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(idx))))}
    for sym in ["AAA", "BBB"]:
        data[sym] = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(idx))))
    prices = pd.DataFrame(data, index=idx)
    tails = get_tails(compute_rrg(prices))
    return build_payload(tails, prices.index[0], prices.index[-1])


def test_payload_structure(payload):
    for key in ["schema_version", "generated_at", "benchmark", "params",
                "views", "assets", "data_start", "data_end"]:
        assert key in payload
    assert payload["benchmark"] == BENCHMARK
    assert BENCHMARK not in payload["assets"]


def test_tails_and_current(payload):
    for sym, asset in payload["assets"].items():
        tail = asset["tail"]
        assert len(tail) == TAIL_LENGTH
        dates = [p["date"] for p in tail]
        assert dates == sorted(dates)
        cur = asset["current"]
        assert cur["date"] == tail[-1]["date"]
        assert cur["ratio"] == tail[-1]["ratio"]
        assert cur["heading"] in {"E", "NE", "N", "NW", "W", "SW", "S", "SE", "flat"}
        assert cur["prev_quadrant"] == tail[-2]["quadrant"]


def test_payload_is_json_serializable(payload):
    text = json.dumps(payload)
    assert json.loads(text)["schema_version"] == payload["schema_version"]


def test_heading_compass():
    assert heading(1, 0) == "E"
    assert heading(1, 1) == "NE"
    assert heading(0, 1) == "N"
    assert heading(-1, 1) == "NW"
    assert heading(-1, 0) == "W"
    assert heading(-1, -1) == "SW"
    assert heading(0, -1) == "S"
    assert heading(1, -1) == "SE"
    assert heading(0, 0) == "flat"
