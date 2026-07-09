"""Phase 2 export layer — turn RRG engine output into data.json for the frontend.

Schema is documented in data_schema.md. Usage:

    python rrg_export.py            # fetch + compute + write data.json
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pandas as pd

from rrg_engine import (
    ASSET_DETAIL, BENCHMARK, BIG_PICTURE, TAIL_LENGTH, TICKERS, W1, W2,
    compute_rrg, fetch_weekly_closes, get_tails,
)

SCHEMA_VERSION = "1.0"

ASSET_NAMES = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "GLD": "Gold", "SLV": "Silver",
    "TLT": "US Treasury 20+Y", "IEF": "US Treasury 7-10Y",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
    "VT": "Global Equities", "AGG": "US Aggregate Bonds",
    "DBC": "Commodities", "BIL": "Cash (1-3M T-Bill)",
    "AOR": "60/40 Benchmark",
    "MCHI": "China", "INDA": "India", "EWJ": "Japan", "EWY": "South Korea",
    "EIDO": "Indonesia", "EWT": "Taiwan", "EWH": "Hong Kong", "VGK": "Europe",
}

_HEADINGS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def heading(d_ratio: float, d_momentum: float) -> str:
    """Compass heading of the tail's last move (8-way, E=+ratio, N=+momentum)."""
    if d_ratio == 0 and d_momentum == 0:
        return "flat"
    angle = math.degrees(math.atan2(d_momentum, d_ratio)) % 360
    return _HEADINGS[int((angle + 22.5) // 45) % 8]


def build_payload(tails: pd.DataFrame, data_start, data_end) -> dict:
    """Assemble the data.json payload from a get_tails() DataFrame."""
    assets = {}
    for sym, grp in tails.groupby("symbol"):
        grp = grp.sort_values("date")
        tail = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "ratio": round(float(r), 4),
                "momentum": round(float(m), 4),
                "quadrant": q,
            }
            for d, r, m, q in zip(grp["date"], grp["ratio"], grp["momentum"], grp["quadrant"])
        ]
        cur = dict(tail[-1])
        if len(tail) >= 2:
            cur["heading"] = heading(
                tail[-1]["ratio"] - tail[-2]["ratio"],
                tail[-1]["momentum"] - tail[-2]["momentum"],
            )
            cur["prev_quadrant"] = tail[-2]["quadrant"]
        assets[sym] = {"name": ASSET_NAMES.get(sym, sym), "tail": tail, "current": cur}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": BENCHMARK,
        "params": {"w1": W1, "w2": W2, "tail_length": TAIL_LENGTH},
        "data_start": pd.Timestamp(data_start).strftime("%Y-%m-%d"),
        "data_end": pd.Timestamp(data_end).strftime("%Y-%m-%d"),
        "views": {
            # only symbols that actually computed (a ticker skipped for short
            # history must not appear in a view the frontend will render)
            "big_picture": {"label": "Big Picture",
                            "symbols": [s for s in BIG_PICTURE if s in assets]},
            "asset_detail": {"label": "Asset Detail",
                             "symbols": [s for s in ASSET_DETAIL if s in assets]},
        },
        "assets": assets,
        "disclaimer": (
            "JdK RS-Ratio/RS-Momentum are approximated (normalized z-score method). "
            "Educational use only — not investment advice."
        ),
    }


def export(path: str = "data.json") -> dict:
    prices = fetch_weekly_closes(TICKERS)
    rrg = compute_rrg(prices)
    tails = get_tails(rrg)
    payload = build_payload(tails, prices.index[0], prices.index[-1])
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    payload = export()
    n = len(payload["assets"])
    print(f"Wrote data.json — {n} assets, data through {payload['data_end']}")
    for sym, a in sorted(payload["assets"].items()):
        c = a["current"]
        print(f"  {sym:8s} {c['quadrant']:10s} heading {c.get('heading', '-'):3s} "
              f"({c['ratio']:.2f}, {c['momentum']:.2f})")
