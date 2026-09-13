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
file in one commit.

### Deploying without breaking it

Unzip over the whole folder and commit everything at once:

```bash
unzip -o us30-monitor.zip -d .. && git add -A && git commit -m "..." && git push
```

Pushing a subset is what causes the redacted `AttributeError` crashes: a module
lands before the config constants it reads, and Python resolves those at import
time, before any panel-level exception handling exists.

Three layers now protect against that, and they are ordered deliberately:

1. Each module carries **local defaults** for every constant its phase
   introduced, read through a `_cfg()` helper at call time rather than at
   import. A stale `config.py` costs a banner, not the app.
2. `app.py` asks each module for its `config_health()` through
   `data_layer.module_health()`, which treats a **missing or throwing**
   `config_health` as "this file is older than app.py" and says so. This
   matters because the first version of the guard called `config_health()`
   directly — so a module older than `app.py` made the partial-deploy guard
   itself the thing that crashed on a partial deploy. A guard that can crash
   is not a guard.
3. `CONFIG_VERSION` in `config.py` is bumped whenever a phase adds constants,
   and `app.py` compares it against what the code expects.

If you already pushed everything and still see the error, the app crashed on an
earlier commit and did not reload: **Manage app → Reboot app**. A crashed
Streamlit app does not reliably pick up the next push.

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
python tests_synthetic.py       # 403 checks, no network required
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
`dow_options.py` instead reads the top 8 names by price weight (~48% of the
index), all of which have deep chains, and weights each one's gamma by its DJIA
point contribution. DIA is fetched only as a cross-check and is discarded when
its OI is under 50k. L6's budget is 10, not the NAS100 build's 19, on that
evidence.

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
| L3 Futures & microstructure | ±15 | live |
| L4 Macro & rates | ±15 | live |
| L5 Regime | ±10 | live |
| L6 Options & gamma | ±10 | live |
| L7 Sector rotation | ±5 | live |

**Unavailable layers are redistributed, not zeroed.** A missing layer scoring 0
is not neutral — it silently drags the composite toward the midpoint and makes
every signal look weaker than the evidence supports. Its budget is reallocated
across the layers that did report, and the loss is shown as `coverage` in the
UI. With all seven layers built, coverage is **100%** — and drops to 90% on
its own whenever component option liquidity is too thin to trust, which is
the designed behaviour rather than a failure.

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
| C11 | Breakout into a long-gamma regime | Downgrade — dealers pin price |
| C12 | Mean-reversion setup in short gamma | Block — do not fade amplification |
| C13 | Today's move ≥95% of implied expected move | Block continuation |
| C14 | Inside a high-impact event blackout window | Block — no edge through a print |

All fourteen are live. C7 went live in phase 4, C11–C13 in phase 5, and
C4/C5/C8/C14 in phase 6. The resolver still reads every field defensively, so
a missing data source silences a conflict rather than crashing it.

---

## Phase 4 notes — what the microstructure layer can and cannot claim

**Delta is a proxy.** `us30_micro.compute_delta` measures where each bar closes
within its own range, mapped to ±1 and weighted by that bar's volume. That
correlates with real delta on trending bars and is close to meaningless on
inside bars. Real delta needs Level 2 / footprint data that yfinance does not
carry. It is weighted at half strength in the layer score, labelled a proxy in
the UI, and there is a test asserting a mid-range close contributes exactly
zero.

**RVOL is same-time-of-day.** Measured against the median volume in the same
5-minute slot across the prior 20 sessions — never a flat daily average. A
naive RVOL fires a volume-expansion signal at every open, because 09:35 always
looks like a spike beside 12:35. There is a test that builds four sessions with
a structurally heavy open and asserts RVOL reads ~1.0, not ~8.

**Basis is RTH-only.** `^DJI` does not tick outside 09:30–16:00 ET, so an
overnight "basis" is really the futures price minus a stale 16:00 print.
Outside RTH the basis returns NaN and C7 does not fire — a test covers exactly
that, because firing C7 on a meaningless number would block good entries every
morning.

**A sweep is a rejection, not a breakout.** Price must penetrate the level by
at least 0.05 × ATR *and* close back inside by half the bar's range. A level
that breaks and holds is a breakout and implies the opposite direction — the
tests assert both cases separately, since conflating them would invert the
signal.

Session boundaries follow Globex: a bar at or after 18:00 ET belongs to the
*next* session, and Sunday-evening bars roll to Monday. Getting this wrong
silently merges two sessions' overnight ranges.

---

## Phase 5 notes — three things the options layer will not pretend to know

**Dealer sign is a convention, not data.** Nobody outside the clearing system
observes which side dealers are on. This uses the standard assumption —
dealers long calls, short puts — so positive GEX means dealers are long gamma
and their hedging suppresses movement. Every surface that shows a gamma number
says so.

**yfinance ships no greeks.** Gamma is computed here from Black-Scholes using
the chain's own implied volatility, validated in the test suite against a
numerical second derivative of the BS price at four spot/strike/tenor
combinations. IV on illiquid strikes is frequently nonsense, so strikes outside
±15% moneyness are dropped and names under a 2,000-contract OI floor are
excluded entirely.

**Gamma is a volatility regime, not a direction.** Long gamma suppresses moves,
short gamma amplifies them; neither says which way price goes. The layer's
small directional score comes only from skew and put/call ratio. The gamma read
is exported as a regime that C11–C13 use to modulate other layers, plus a
`vol_scalar` that scales overall confidence the same way dispersion does.

If fewer than 30% of index weight has usable chains, the layer reports
unavailable and C6 redistributes its budget. That is the design working, not
an outage.

---

## Phase 6 notes — the calendar is a gate, not a layer

The seven layers still sum to ±100 and `us30_calendar.py` adds no score. That
is deliberate: "there is an FOMC statement in twenty minutes" is not a bullish
or bearish opinion, it is a reason not to have a position. The calendar feeds
C4, C5, C8 and C14, and nothing else.

**NFP is computed, not listed.** Non-farm payrolls is the first Friday of the
month, so it is derived from the rule and can never go stale. Anything
derivable is derived.

**FOMC and CPI are hardcoded, and hardcoded tables rot.** They come from
federalreserve.gov and the BLS schedule, verified on `CALENDAR_VERIFIED_ON`,
and each table carries its own end date. Past that end date the module reports
*"past the end of the table"* rather than returning no upcoming events — a
calendar that silently returns nothing reads as an all-clear, which is the
worst possible failure mode for an event gate. The app also warns once the
tables are more than `CALENDAR_STALE_AFTER_DAYS` old.

**yfinance earnings dates are frequently wrong** — stale by months in some
cases, missing entirely in others. Every entry is labelled with its source and
confidence, C4 caps size and widens the stop rather than blocking, and an
ex-dividend date inferred from historical payout cadence is marked ESTIMATED
and flagged. Verify against investor relations before leaning on either.

**Ex-dividend handling closes a real loop.** When a top-8 name goes ex-div, the
app recomputes attribution with that name's contribution neutralised — UNH
going ex at $3/share is an 18-point index drop carrying no information. The
underlying fetch is cached, so the second pass is free.

---

## Files

```
config.py              constituents, sector map, tickers, thresholds, TTLs
data_layer.py          ALL yfinance access. Circuit breaker, column normalisation
dow_attribution.py     CORE — divisor, point contributions, breadth, concentration
us30_technicals.py     CPR, pivots, EMA, RSI + decay, ATR, VWAP, divergence
us30_micro.py          basis, overnight/prior levels, RVOL, delta proxy, sweeps
dow_options.py         BS greeks, per-name GEX, point-weighted gamma regime
us30_macro.py          rates, curve, DXY, oil, VXD/VIX
us30_regime.py         trend/chop/revert + dispersion regime
us30_sectors.py        Dow-weighted sector RS (no XLU/XLRE — the Dow has neither)
us30_calendar.py       event risk, earnings, ex-dividends, DJIA/NDX correlation
us30_journal.py        signal ledger, realised expectancy, forward-test gates
us30_master_signal.py  7-layer aggregation, C1–C10, trade plan
app.py                 Streamlit UI, per-panel exception isolation
validate_tickers.py    PHASE 0 — run this first
tests_synthetic.py     403 checks against constructed data, no network
```

---

## Roadmap

- [x] Phase 0 — validation script
- [x] Phase 1 — attribution engine with live divisor
- [x] Phase 2 — technicals on the correct tickers
- [x] Phase 3 — macro, rates, sectors
- [x] Phase 4 — YM basis, overnight levels, RVOL, sweeps, delta proxy
- [x] Phase 5 — component-weighted options gamma across the top 8
- [x] Phase 6 — event calendar, earnings, ex-dividends, cross-index (C4/C5/C8/C14)
- [x] Phase 7 — `us30_journal.py`, forward-test gates, **no live trading yet**
- [ ] Phase 8 — MT5 wiring: point calibration, ATR sizing, R-aware breaker

---

## Phase 7 notes — the part that can actually answer the question

Everything before this is a hypothesis. Six phases of engines produce a number
whose relationship to money is entirely unmeasured. `us30_journal.py` records
what the dashboard claimed, what happened, and grades the difference.

**Win rate is never shown bare.** Every win rate carries its Wilson 95%
interval and its n. "100%" from one trade renders as `100% (95% CI 21–100%,
n=1)`, which is the truth. A bare percentage from a small sample is how a
system talks its owner into trusting noise. The interval is validated in the
tests against an independent quadratic solution of the Wilson equation, not
against a remembered constant.

**Expectancy is the headline.** The journal grades against expectancy in index
points and profit factor, with win rate as context. It also computes
`breakeven_win_rate(rr, cost, risk)` against the system's OWN realised R and
reports whether the achieved rate clears it — the identity from the research
doc applied to real numbers rather than restated as a claim.

**Every layer score is stored, not just the composite.** Seven extra columns
per row, so you can afterwards ask which layers actually predicted anything.
The Journal tab charts each layer's correlation with realised points once n≥10.
The honest answer may be that some layers are decoration, and that question is
worth more than the win rate.

**Both forward-test gates must clear: 30 trades AND 60 days.** Thirty trades
from one week is one market regime wearing a sample's clothing.

**Blocked signals are never logged.** A block is the system working correctly,
not a trade. Logging them would make every rate computed from the ledger
meaningless.

### The persistence trap

**Streamlit Cloud's filesystem is ephemeral.** It is wiped on every reboot and
redeploy. A journal written only there will lose the forward test silently —
you would not find out until you went looking for the results. The app detects
the host and shows a red banner when storage is not durable.

Two working options:

1. **Export regularly.** The Journal tab has a download button and a restore
   uploader that merges by `signal_id` + `logged_at`. Low-tech and reliable.
2. **Run the durable copy on the MT5 host.** Your bot already imports these
   modules directly, and that machine has a real filesystem. `CsvStore` works
   there unchanged and `durable` reports `True`. This is the better path.

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
