# US30 Monitor

Intraday signal dashboard for the Dow Jones 30 spot index (US30), built on
Streamlit + yfinance. Standalone repo by design — it shares no code with
`mag7-monitor`, so a bug here can never white-screen the live NAS100 dashboard.

Companion to the pre-build research and architecture document.

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: point the app at `app.py` in the repo root. Push every changed
file in one commit — a partial deploy is what causes the crash-on-startup you
have hit before.

---

## Phase 0 first. Really.

```bash
streamlit run validate_tickers.py      # or: python validate_tickers.py
```

Every data assumption in this repo is documentation-sourced, **not empirically
confirmed** — Yahoo Finance was unreachable from the environment the code was
written in. `validate_tickers.py` probes all ~50 symbols at 1m/5m/1d, reports
which resolve, which carry real volume, which have usable option chains, and
whether the derived divisor reconciles. Run it on Streamlit Cloud before
trusting any panel. Its output will revise some assumptions.

The synthetic suite covers the part that breaks quietly — the maths:

```bash
python tests_synthetic.py       # 112 checks, no network required
```

---

## The one idea this repo is built around

The Dow is **price-weighted**. Every component's effect on the index is exact,
not estimated:

```
contribution_points(i) = (price_i − prev_close_i) / divisor
```

and the 30 contributions sum to the index change. There is no cap-weighted
equivalent, and it is the largest edge available on this instrument. It is why
L1 carries the biggest budget.

The divisor is **derived live** — `sum(component prices) / ^DJI` — never
hard-coded. It changed on 29 June 2026 when Alphabet replaced Verizon and it
changes on every split. Drift in the derived value *is* the constituent-change
detector, and it costs nothing.

---

## Three traps this code is written around

**1. `^DJI` has no volume.** Yahoo reports it as 0. A VWAP computed there is a
silent divide-by-zero that returns a plausible-looking number. `data_layer`
routes every volume-dependent calculation to `YM=F` or `DIA` and
`us30_technicals.vwap` returns NaN rather than garbage if you point it at the
index anyway. There is a test for exactly this.

**2. Options liquidity.** DIA options trade ~13.9k contracts/day against QQQ's
~1.53M; DJX index options ~3.0k. The QQQ GEX engine from `mag7-monitor` does
**not** port to DIA — strike-level OI is too thin for stable gamma walls.
Phase 5 builds it component-weighted across the top 8 liquid chains instead.
L6's budget is 10, not the NAS100 build's 19, on that evidence.

**3. The MT5 point trap.** On MT5, a "point" is a *per-symbol* property. The
USTEC bot's "1000 points max stop" could mean 10, 100 or 1000 index points on
US30 depending on quote digits. This repo defines risk in **ATR multiples**,
converts to **index points**, and leaves the broker-point conversion to
runtime via `mt5.symbol_info(symbol).point`. Never copy a point count across
instruments.

---

## Layers

Seven layers summing directly to ±100. No ±130-to-±100 rescale — that step was
a recurring source of off-by-a-bit bugs and there is no reason to repeat it.

| Layer | Budget | Status |
|---|---|---|
| L1 Attribution & breadth | ±25 | live |
| L2 Technicals | ±20 | live |
| L3 Futures & microstructure | ±15 | phase 4 |
| L4 Macro & rates | ±15 | live |
| L5 Regime | ±10 | live |
| L6 Options & gamma | ±10 | phase 5 |
| L7 Sector rotation | ±5 | live |

**Unavailable layers are redistributed, not zeroed.** A missing layer scoring 0
is not neutral — it silently drags the composite toward the midpoint and makes
every signal look weaker than the evidence supports. Its budget is reallocated
across the layers that did report, and the loss is shown as `coverage` in the
UI. Current coverage with phases 4 and 5 unbuilt: **75%**.

---

## Conflict resolver

| Code | Fires when | Action |
|---|---|---|
| C1 | Move >150pts, top-2 share >65%, weak participation | Block — narrow tape |
| C2 | 10y +>8bp while XLF lags SPY | Downgrade |
| C3 | Directional signal in CHOP regime | Block |
| C4 | Top-8 earnings within 24h | Cap lot 0.5×, widen stop 1 ATR |
| C5 | Top-8 ex-dividend today | Neutralise points, suppress gap reads |
| C6 | Options layer thin or absent | Redistribute budget |
| C7 | YM−cash basis beyond 2σ | Block until convergence |
| C8 | US30 vs NDX conflict at high correlation | Downgrade; upgrade if rotation |
| C9 | Expected move < 3× round-trip cost | Block regardless of score |
| C10 | ≥6 live layers agree | Upgrade |

C4, C5 and C7 need data phases 4–5 supply; the resolver reads them defensively
and simply does not fire until those layers exist.

---

## Files

```
config.py              constituents, sector map, tickers, thresholds, TTLs
data_layer.py          ALL yfinance access. Circuit breaker, column normalisation
dow_attribution.py     CORE — divisor, point contributions, breadth, concentration
us30_technicals.py     CPR, pivots, EMA, RSI + decay, ATR, VWAP, divergence
us30_macro.py          rates, curve, DXY, oil, VXD/VIX
us30_regime.py         trend/chop/revert + dispersion regime
us30_sectors.py        Dow-weighted sector RS (no XLU/XLRE — the Dow has neither)
us30_master_signal.py  7-layer aggregation, C1–C10, trade plan
app.py                 Streamlit UI, per-panel exception isolation
validate_tickers.py    PHASE 0 — run this first
tests_synthetic.py     112 checks against constructed data, no network
```

---

## Roadmap

- [x] Phase 0 — validation script
- [x] Phase 1 — attribution engine with live divisor
- [x] Phase 2 — technicals on the correct tickers
- [x] Phase 3 — macro, rates, sectors
- [ ] Phase 4 — YM basis, overnight levels, RVOL, sweeps, delta proxy
- [ ] Phase 5 — component-weighted options gamma across the top 8
- [ ] Phase 6 — full conflict resolver with C4/C5/C7 data
- [ ] Phase 7 — `us30_journal.py`, 60-day forward test, **no live trading**
- [ ] Phase 8 — MT5 wiring: point calibration, ATR sizing, R-aware breaker

---

## On the win-rate target

Break-even win rate is `(1 + cost/risk) / (R + 1)`. With a 40-point stop and 5
points of round-trip cost, **75% at 0.5:1 reward-to-risk is exactly
break-even** — and 0.5:1 is where a system drifts when it is optimised for hit
rate. The targets this repo is tuned toward are expectancy above 8 net index
points and profit factor above 1.4, with selectivity (2–4 signals a session)
as the mechanism. `C9` and `MIN_TARGET_PTS` exist to enforce that, and phase 7
exists to measure whether it worked.

Nothing here has been forward-tested. Do not wire it to the MT5 bot before
phase 7 is complete.
