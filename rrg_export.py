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

SCHEMA_VERSION = "1.1"   # 1.1: assets may carry "daily" OHLCV history (additive)

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


def build_payload(tails: pd.DataFrame, data_start, data_end,
                  views: dict | None = None) -> dict:
    """Assemble a data payload from a get_tails() DataFrame.

    `views` maps view key -> {"label", "symbols"}; defaults to both views
    (single-view exports pass just their own so each JSON file stands alone).
    """
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
            key: {"label": v["label"],
                  "symbols": [s for s in v["symbols"] if s in assets]}
            for key, v in (views if views is not None else {
                "big_picture": {"label": "Big Picture", "symbols": BIG_PICTURE},
                "asset_detail": {"label": "Asset Detail", "symbols": ASSET_DETAIL},
            }).items()
        },
        "assets": assets,
        "disclaimer": (
            "JdK RS-Ratio/RS-Momentum are approximated (normalized z-score method). "
            "Educational use only — not investment advice."
        ),
    }


# Per-view export config: Big Picture is strictly weekly (completed Fridays
# only); Asset Detail refreshes daily, so its newest point uses the latest
# available close (include_partial). Separate files so the two update
# cadences never overwrite each other.
VIEWS = {
    "bigpicture": {"key": "big_picture", "label": "Big Picture",
                   "symbols": BIG_PICTURE, "include_partial": False,
                   "path": "data_bigpicture.json"},
    "assetdetail": {"key": "asset_detail", "label": "Asset Detail",
                    "symbols": ASSET_DETAIL, "include_partial": True,
                    "path": "data_assetdetail.json"},
}


def fetch_daily_history(symbols, period: str = "1y") -> dict:
    """1y of daily OHLCV per symbol, for the frontend's price-chart panel."""
    import yfinance as yf

    raw = yf.download(list(symbols), period=period, interval="1d",
                      auto_adjust=True, progress=False, group_by="ticker")
    out = {}
    for sym in symbols:
        try:
            df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df.dropna(subset=["Close"])
        if df.empty:
            continue
        out[sym] = {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "o": [round(float(x), 4) for x in df["Open"]],
            "h": [round(float(x), 4) for x in df["High"]],
            "l": [round(float(x), 4) for x in df["Low"]],
            "c": [round(float(x), 4) for x in df["Close"]],
            "v": [int(x) for x in df["Volume"].fillna(0)],
        }
    return out


def export_view(view: str) -> dict:
    cfg = VIEWS[view]
    tickers = list(dict.fromkeys([*cfg["symbols"], BENCHMARK]))
    prices = fetch_weekly_closes(tickers, include_partial=cfg["include_partial"])
    tails = get_tails(compute_rrg(prices))
    payload = build_payload(
        tails, prices.index[0], prices.index[-1],
        views={cfg["key"]: {"label": cfg["label"], "symbols": cfg["symbols"]}})
    for sym, hist in fetch_daily_history(cfg["symbols"]).items():
        if sym in payload["assets"]:
            payload["assets"][sym]["daily"] = hist
    with open(cfg["path"], "w") as f:
        json.dump(payload, f, indent=2)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export RRG view data to JSON")
    parser.add_argument("--view", choices=[*VIEWS, "both"], default="both",
                        help="which view file to generate (default: both)")
    args = parser.parse_args()

    for view in (list(VIEWS) if args.view == "both" else [args.view]):
        payload = export_view(view)
        n = len(payload["assets"])
        print(f"Wrote {VIEWS[view]['path']} — {n} assets, data through {payload['data_end']}")
        for sym, a in sorted(payload["assets"].items()):
            c = a["current"]
            print(f"  {sym:8s} {c['quadrant']:10s} heading {c.get('heading', '-'):3s} "
                  f"({c['ratio']:.2f}, {c['momentum']:.2f})")
