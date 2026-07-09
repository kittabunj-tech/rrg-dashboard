"""RRG Engine — Phase 0-1 of the RRG Asset Allocation Dashboard.

Computes JdK-style RS-Ratio / RS-Momentum (normalized approximation) for a
universe of ETFs + crypto against the AOR benchmark, using weekly (Friday)
closes from yfinance.

Formulas (see RRG_Dashboard_Process_Roadmap.md §4):

    RS(t)          = 100 * P_asset(t) / P_benchmark(t)
    RS_Ratio(t)    = 100 + (RS - SMA(RS, W1)) / StdDev(RS, W1)
    ROC(t)         = RS_Ratio pct-change over W2 weeks (in %)
    RS_Momentum(t) = 100 + (ROC - SMA(ROC, W1)) / StdDev(ROC, W1)

Usage:
    python rrg_engine.py            # fetch, compute, print table, save chart

The JdK formulas are proprietary; this is the commonly accepted normalized
approximation. Cross-check quadrants against StockCharts.com RRG (weekly,
benchmark AOR) and tune W1 (10-14) / W2 if rotation direction disagrees.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BENCHMARK = "AOR"

# ---------------------------------------------------------------------------
# Universe config — single source of truth for adding/removing tickers.
# A symbol may appear in several views; the benchmark is added automatically.
# ---------------------------------------------------------------------------
UNIVERSE = {
    "big_picture": ["VT", "AGG", "DBC", "BIL"],
    "asset_detail": [
        "SPY", "QQQ", "GLD", "SLV", "TLT", "IEF", "BTC-USD", "ETH-USD", "BIL",
        # country/region ETFs
        "MCHI", "INDA", "EWJ", "EWY", "EIDO", "EWT", "EWH", "VGK",
    ],
}

BIG_PICTURE = UNIVERSE["big_picture"]
ASSET_DETAIL = UNIVERSE["asset_detail"]
# ordered union of every view plus the benchmark
TICKERS = list(dict.fromkeys([*BIG_PICTURE, *ASSET_DETAIL, BENCHMARK]))
CRYPTO_TICKERS = {"BTC-USD", "ETH-USD"}

# Default tuning per roadmap §4 (W1 adjustable 10-14).
W1 = 14   # rolling window (weeks) for z-score normalization
W2 = 4    # look-back (weeks) for RS-Ratio rate of change
TAIL_LENGTH = 10


# ---------------------------------------------------------------------------
# 1. Data pipeline
# ---------------------------------------------------------------------------

def fetch_weekly_closes(
    tickers: Iterable[str] = TICKERS,
    years: int = 3,
    end: Optional[pd.Timestamp] = None,
    include_partial: bool = False,
) -> pd.DataFrame:
    """Fetch daily closes from yfinance and align to Friday weekly closes.

    Crypto trades 24/7 while ETFs only trade business days, so everything is
    resampled to W-FRI taking the last available close of each week; ETF
    holiday gaps are forward-filled before resampling so a Friday holiday
    falls back to Thursday's close.

    include_partial=False (weekly cadence) drops the in-progress week, so the
    series ends on the last completed Friday. include_partial=True (daily
    cadence) keeps it as one extra row built from the latest available close
    and dated with the actual last trading day — completed weeks are identical
    either way, so only the newest point moves between daily runs.
    """
    import yfinance as yf

    tickers = list(tickers)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    # Pad the start so the first weeks still have full rolling windows.
    start = end - pd.DateOffset(years=years) - pd.DateOffset(weeks=W1 + W2 + 2)

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    closes = closes.reindex(columns=tickers)

    # Forward-fill so ETF holidays (and any crypto gaps) inherit the prior
    # close, then take the last value of each Mon-Fri week.
    weekly = closes.ffill().resample("W-FRI").last()

    # The resample labels the current partial week with the upcoming Friday,
    # which hasn't closed yet: drop it (weekly cadence) or keep it relabeled
    # to the true last trading day (daily cadence).
    if len(weekly) and weekly.index[-1] > end:
        if include_partial:
            last_daily = closes.dropna(how="all").index[-1]
            weekly.index = weekly.index[:-1].append(pd.DatetimeIndex([last_daily]))
        else:
            weekly = weekly.iloc[:-1]

    # Drop leading rows where the benchmark has no data yet.
    if weekly[BENCHMARK].isna().all():
        raise ValueError(f"No data returned for benchmark {BENCHMARK}")
    weekly = weekly.loc[weekly[BENCHMARK].first_valid_index():]

    # A ticker with missing or short history (bad symbol, new listing) is
    # skipped with a warning instead of failing the whole pipeline.
    grace = weekly.index[min(4, len(weekly) - 1)]   # allow ≤4 weeks' slack
    drop = []
    for t in tickers:
        first = weekly[t].first_valid_index()
        if first is None:
            logger.warning("%s: no data returned from yfinance — skipped", t)
            drop.append(t)
        elif first > grace:
            logger.warning("%s: history starts %s, shorter than %dy — skipped",
                           t, first.date(), years)
            drop.append(t)
    return weekly.drop(columns=drop)


# ---------------------------------------------------------------------------
# 2. RRG math
# ---------------------------------------------------------------------------

def compute_rrg(
    prices: pd.DataFrame,
    benchmark: str = BENCHMARK,
    w1: int = W1,
    w2: int = W2,
) -> pd.DataFrame:
    """Compute normalized RS-Ratio and RS-Momentum for every non-benchmark column.

    Returns a DataFrame with MultiIndex columns (symbol, {"ratio", "momentum"})
    indexed by week.
    """
    if benchmark not in prices.columns:
        raise ValueError(f"benchmark {benchmark!r} not in price columns")

    assets = [c for c in prices.columns if c != benchmark]
    out = {}
    for sym in assets:
        rs = 100.0 * prices[sym] / prices[benchmark]

        mean = rs.rolling(w1).mean()
        std = rs.rolling(w1).std(ddof=0)
        # A near-zero std means RS is flat over the window (no relative move);
        # guard the division so float noise doesn't produce spurious z-scores.
        dev = ((rs - mean) / std).where(std > 1e-9 * mean.abs(), 0.0)
        ratio = 100.0 + dev

        roc = 100.0 * ratio.pct_change(w2, fill_method=None)
        roc_mean = roc.rolling(w1).mean()
        roc_std = roc.rolling(w1).std(ddof=0)
        mom_dev = ((roc - roc_mean) / roc_std).where(roc_std > 1e-12, 0.0)
        momentum = 100.0 + mom_dev

        out[(sym, "ratio")] = ratio
        out[(sym, "momentum")] = momentum

    rrg = pd.DataFrame(out)
    rrg.columns = pd.MultiIndex.from_tuples(rrg.columns, names=["symbol", "metric"])
    return rrg


def get_tails(rrg: pd.DataFrame, length: int = TAIL_LENGTH) -> pd.DataFrame:
    """Last `length` weekly (ratio, momentum) points per asset, long format.

    Columns: symbol, date, ratio, momentum, quadrant. Sorted oldest→newest
    within each symbol; the last row per symbol is the current position.
    """
    rows = []
    for sym in rrg.columns.get_level_values("symbol").unique():
        sub = rrg[sym].dropna().tail(length)
        for date, row in sub.iterrows():
            rows.append({
                "symbol": sym,
                "date": date,
                "ratio": row["ratio"],
                "momentum": row["momentum"],
                "quadrant": classify_quadrant(row["ratio"], row["momentum"]),
            })
    return pd.DataFrame(rows)


def classify_quadrant(ratio: float, momentum: float) -> str:
    if ratio >= 100 and momentum >= 100:
        return "Leading"
    if ratio >= 100:
        return "Weakening"
    if momentum >= 100:
        return "Improving"
    return "Lagging"


# ---------------------------------------------------------------------------
# 3. Diagnostic plot
# ---------------------------------------------------------------------------

QUADRANT_COLORS = {
    "Leading": "#2e7d32", "Weakening": "#f9a825",
    "Lagging": "#c62828", "Improving": "#1565c0",
}


def plot_rrg(
    tails: pd.DataFrame,
    symbols: Optional[Iterable[str]] = None,
    title: str = "RRG (weekly, benchmark AOR)",
    save_path: Optional[str] = None,
):
    """Static matplotlib RRG scatter with 4 shaded quadrants and fading tails."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if symbols is not None:
        tails = tails[tails["symbol"].isin(list(symbols))]

    fig, ax = plt.subplots(figsize=(10, 9))

    pad = 0.5
    lo_x = min(tails["ratio"].min(), 100 - 3) - pad
    hi_x = max(tails["ratio"].max(), 100 + 3) + pad
    lo_y = min(tails["momentum"].min(), 100 - 3) - pad
    hi_y = max(tails["momentum"].max(), 100 + 3) + pad

    ax.axhspan(100, hi_y, xmin=0, xmax=1, color="none")
    ax.fill_between([lo_x, 100], 100, hi_y, color="#1565c0", alpha=0.07)   # Improving
    ax.fill_between([100, hi_x], 100, hi_y, color="#2e7d32", alpha=0.07)   # Leading
    ax.fill_between([lo_x, 100], lo_y, 100, color="#c62828", alpha=0.07)   # Lagging
    ax.fill_between([100, hi_x], lo_y, 100, color="#f9a825", alpha=0.07)   # Weakening
    ax.axhline(100, color="gray", lw=1)
    ax.axvline(100, color="gray", lw=1)
    ax.text(hi_x, hi_y, "Leading", ha="right", va="top", color="#2e7d32", fontsize=11)
    ax.text(lo_x, hi_y, "Improving", ha="left", va="top", color="#1565c0", fontsize=11)
    ax.text(lo_x, lo_y, "Lagging", ha="left", va="bottom", color="#c62828", fontsize=11)
    ax.text(hi_x, lo_y, "Weakening", ha="right", va="bottom", color="#f9a825", fontsize=11)

    cmap = plt.get_cmap("tab10")
    for i, (sym, grp) in enumerate(tails.groupby("symbol")):
        grp = grp.sort_values("date")
        color = cmap(i % 10)
        n = len(grp)
        for j in range(n):  # fading tail dots
            ax.plot(grp["ratio"].iloc[j], grp["momentum"].iloc[j], "o",
                    color=color, ms=4, alpha=0.25 + 0.6 * j / max(n - 1, 1))
        ax.plot(grp["ratio"], grp["momentum"], "-", color=color, lw=1.2, alpha=0.6)
        ax.plot(grp["ratio"].iloc[-1], grp["momentum"].iloc[-1], "o",
                color=color, ms=10, mec="black", mew=0.8)
        if n >= 2:  # direction arrow at the head
            ax.annotate("", xy=(grp["ratio"].iloc[-1], grp["momentum"].iloc[-1]),
                        xytext=(grp["ratio"].iloc[-2], grp["momentum"].iloc[-2]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
        ax.annotate(sym, (grp["ratio"].iloc[-1], grp["momentum"].iloc[-1]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10,
                    fontweight="bold", color=color)

    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("JdK RS-Ratio (normalized approx.)")
    ax.set_ylabel("JdK RS-Momentum (normalized approx.)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# CLI view names → (view key, symbols, partial-week mode). Big Picture stays
# strictly weekly (completed Fridays only); Asset Detail refreshes daily, so
# its newest point is built from the latest available close.
VIEW_RUNS = {
    "bigpicture": ("big_picture", BIG_PICTURE, False),
    "assetdetail": ("asset_detail", ASSET_DETAIL, True),
}


def main():
    import argparse

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compute RRG data and diagnostic charts")
    parser.add_argument("--view", choices=[*VIEW_RUNS, "both"], default="both",
                        help="which view to compute (default: both)")
    args = parser.parse_args()
    runs = list(VIEW_RUNS) if args.view == "both" else [args.view]

    for run in runs:
        key, symbols, partial = VIEW_RUNS[run]
        tickers = list(dict.fromkeys([*symbols, BENCHMARK]))
        print(f"[{run}] fetching 3y weekly closes for {len(tickers)} tickers"
              f"{' (incl. partial week)' if partial else ''}...")
        prices = fetch_weekly_closes(tickers, include_partial=partial)
        print(f"  {len(prices)} weeks, {prices.index[0].date()} → {prices.index[-1].date()}, "
              f"{len(prices.columns)}/{len(tickers)} tickers usable")

        tails = get_tails(compute_rrg(prices))
        latest = (tails.sort_values("date").groupby("symbol").tail(1)
                  .set_index("symbol")[["date", "ratio", "momentum", "quadrant"]]
                  .sort_values("ratio", ascending=False))
        print(f"\n[{run}] latest positions (benchmark AOR):")
        print(latest.to_string(float_format="%.2f"))

        tails.to_csv(f"rrg_tails_{run}.csv", index=False)
        plot_rrg(tails, symbols=symbols,
                 title=f"RRG — {key.replace('_', ' ').title()} (weekly vs AOR)",
                 save_path=f"rrg_{run}.png")
        print(f"Saved: rrg_tails_{run}.csv, rrg_{run}.png\n")

    print("Validation: compare SPY/QQQ/GLD/TLT quadrants & rotation vs "
          "StockCharts.com RRG (weekly, benchmark AOR); tune W1/W2 if <8/10 match.")


if __name__ == "__main__":
    main()
