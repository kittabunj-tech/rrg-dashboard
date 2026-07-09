# Data JSON Schema (v1.0)

Output of `rrg_export.py`, consumed by the Phase 3 frontend. Since the
split-cadence update there are **two files with the same schema**, each
carrying only its own view (so the two update schedules never overwrite
each other):

| File | View | Cadence | Newest point |
|---|---|---|---|
| `data_bigpicture.json` | `big_picture` | weekly (Sat 07:30 ICT) | last completed Friday |
| `data_assetdetail.json` | `asset_detail` | daily (07:30 ICT) | latest available close (in-progress week; its `date` is the actual last trading day, not a Friday) |

```jsonc
{
  "schema_version": "1.0",
  "generated_at": "2026-07-07T09:00:00Z",   // UTC timestamp of generation
  "benchmark": "AOR",
  "params": { "w1": 14, "w2": 4, "tail_length": 10 },
  "data_start": "2023-02-17",               // first week in the price series
  "data_end": "2026-07-03",                 // last completed Friday close

  // Which symbols belong to each dashboard tab
  "views": {
    "big_picture":  { "label": "Big Picture",  "symbols": ["VT", "AGG", "DBC", "BIL"] },
    "asset_detail": { "label": "Asset Detail", "symbols": ["SPY", "QQQ", "..."] }
  },

  "assets": {
    "SPY": {
      "name": "S&P 500",
      // Oldest → newest, exactly tail_length points (fewer only if history is short)
      "tail": [
        { "date": "2026-05-01", "ratio": 101.23, "momentum": 100.45, "quadrant": "Leading" }
        // ... 10 points
      ],
      // Copy of the last tail point, plus:
      "current": {
        "date": "2026-07-03",
        "ratio": 100.28,
        "momentum": 99.25,
        "quadrant": "Weakening",
        "heading": "NE",              // 8-way compass of last week's move
                                      // (E = ratio up, N = momentum up), or "flat"
        "prev_quadrant": "Leading"    // quadrant one week earlier (for rotation alerts)
      }
    }
  },

  "disclaimer": "..."
}
```

## Field notes

- **quadrant** ∈ `Leading | Weakening | Lagging | Improving` (boundaries at 100 inclusive to Leading side).
- **ratio / momentum** rounded to 4 decimals; centered at 100 by construction.
- The benchmark itself (`AOR`) never appears in `assets`.
- Frontend should treat unknown extra fields as ignorable; breaking changes bump `schema_version`.
