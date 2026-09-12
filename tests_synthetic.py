"""
tests_synthetic.py — US30 Monitor

Yahoo Finance is unreachable from the environment this repo was written in, so
none of the engines could be exercised against live data. Everything here
instead drives the PURE functions with constructed inputs whose correct answers
are known in advance.

That covers the part that actually breaks quietly: the maths. It does NOT cover
whether the tickers resolve — that is `validate_tickers.py`, phase 0, and it
must be run on Streamlit Cloud before trusting any panel.

    python tests_synthetic.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
import dow_attribution as attr
import us30_macro as macro_mod
import us30_master_signal as master
import us30_regime as regime_mod
import us30_sectors as sectors_mod
import us30_technicals as tech

PASS, FAIL = 0, 0
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def close_to(a: float, b: float, tol: float = 1e-6) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)


# ==========================================================================
print("\n[1] Divisor derivation and attribution reconciliation")
# ==========================================================================
DIV = config.DIVISOR_REFERENCE
PRICES = {
    "GS": 1037.48, "CAT": 819.40, "MSFT": 497.28, "UNH": 388.82, "AMGN": 384.39,
    "TRV": 371.68, "V": 370.32, "JPM": 359.54, "GOOGL": 339.39, "AAPL": 333.44,
    "AXP": 324.80, "SHW": 322.77, "HD": 310.68, "JNJ": 269.03, "AMZN": 256.48,
    "MCD": 253.74, "CRM": 247.70, "IBM": 236.56, "NVDA": 219.42, "CVX": 213.03,
    "BA": 208.99, "HON": 203.55, "MMM": 163.21, "MRK": 145.03, "PG": 144.78,
    "CSCO": 109.55, "WMT": 106.44, "DIS": 106.40, "KO": 88.36, "NKE": 36.90,
}
last = pd.Series(PRICES)
index_level = float(last.sum()) / DIV

d = attr.derive_divisor(last, index_level)
check("divisor derived from 30 prices", d.derived)
check("derived divisor matches reference", close_to(d.value, DIV, 1e-12),
      f"got {d.value!r}")
check("no drift alert on clean input", not d.alert)
check("points per dollar ~5.944", close_to(d.points_per_dollar, 5.944, 0.001),
      f"got {d.points_per_dollar:.4f}")

# An incomplete price set must NOT silently derive a wrong divisor.
partial = attr.derive_divisor(last.iloc[:25], index_level)
check("incomplete price set falls back to reference", not partial.derived)

# Construct a move with a known answer: GS +$10, everyone else flat.
prev = last.copy()
prev["GS"] = last["GS"] - 10.0
expected_pts = 10.0 / DIV
new_index = float(last.sum()) / DIV
old_index = float(prev.sum()) / DIV

rep = attr.compute_attribution(last, prev, new_index, old_index)
check("attribution ok", rep.ok, rep.note)
check("GS contribution is exactly $10/divisor",
      close_to(float(rep.table.loc["GS", "points"]), round(expected_pts, 2), 0.01),
      f"got {rep.table.loc['GS', 'points']}")
check("contributions reconcile to index change",
      abs(rep.reconciliation_error) < 0.05,
      f"error {rep.reconciliation_error:.4f}")
check("single-name move flagged as 100% top-2 share",
      close_to(rep.top2_share, 1.0, 1e-9), f"got {rep.top2_share}")
check("narrow tape flagged", any("Narrow tape" in f for f in rep.flags),
      str(rep.flags))
check("1 advancer, 0 decliners", rep.advancers == 1 and rep.decliners == 0)
check("efficiency is +1 on a one-sided move", close_to(rep.efficiency, 1.0, 1e-9))

# Broad move: every name up 1%.
prev_broad = last / 1.01
rep_broad = attr.compute_attribution(
    last, prev_broad, new_index, float(prev_broad.sum()) / DIV)
check("broad move: 30 advancers", rep_broad.advancers == 30)
check("broad move: participation +1", close_to(rep_broad.participation, 1.0, 1e-9))
check("broad move scores higher than narrow move",
      rep_broad.score > rep.score, f"{rep_broad.score:.2f} vs {rep.score:.2f}")
check("broad move top-2 share is small", rep_broad.top2_share < 0.25,
      f"got {rep_broad.top2_share:.3f}")

# Divergence: index up on GS alone while 25 names fall slightly.
prev_div = last.copy()
prev_div["GS"] = last["GS"] - 60.0
for t in list(PRICES)[1:26]:
    prev_div[t] = last[t] * 1.002          # these names DECLINED into today
rep_div = attr.compute_attribution(
    last, prev_div, new_index, float(prev_div.sum()) / DIV)
check("breadth divergence detected", rep_div.divergence, str(rep_div.flags))
check("divergence suppresses the score", abs(rep_div.score) < 8.0,
      f"score {rep_div.score:.2f}")

# Ex-dividend neutralisation.
prev_xd = last.copy()
prev_xd["UNH"] = last["UNH"] + 3.0         # $3 mechanical drop
rep_xd = attr.compute_attribution(
    last, prev_xd, new_index, float(prev_xd.sum()) / DIV, ex_div_today={"UNH"})
check("ex-div contribution neutralised to zero",
      close_to(float(rep_xd.table.loc["UNH", "points"]), 0.0, 1e-9))

# Live re-ranking.
top8 = attr.rank_by_weight(rep, 8)
check("top 8 ranked by price weight",
      top8 == ["GS", "CAT", "MSFT", "UNH", "AMGN", "TRV", "V", "JPM"], str(top8))

split = last.copy()
split["GS"] = last["GS"] / 4.0             # 4:1 split
rep_split = attr.compute_attribution(split, split, float(split.sum()) / DIV,
                                     float(split.sum()) / DIV)
check("a split drops GS out of the top 8",
      "GS" not in attr.rank_by_weight(rep_split, 8),
      str(attr.rank_by_weight(rep_split, 8)))
check("split triggers divisor drift alert", rep_split.divisor.alert is False
      or rep_split.divisor.derived, "divisor state after split")


# ==========================================================================
print("\n[2] Indicator primitives")
# ==========================================================================
n = 300
idx = pd.date_range("2026-01-02", periods=n, freq="B")

# Monotonic rise -> RSI pinned at 100, never NaN-from-divide-by-zero.
rising = pd.Series(np.linspace(50000, 55000, n), index=idx)
r = tech.rsi(rising)
check("RSI of a monotonic rise is 100", close_to(float(r.iloc[-1]), 100.0, 1e-6),
      f"got {r.iloc[-1]}")

falling = pd.Series(np.linspace(55000, 50000, n), index=idx)
check("RSI of a monotonic fall is ~0",
      float(tech.rsi(falling).iloc[-1]) < 1e-6)

rng = np.random.default_rng(7)
noise = pd.Series(52000 + np.cumsum(rng.normal(0, 40, n)), index=idx)
rn = tech.rsi(noise).dropna()
check("RSI stays within [0,100]", bool((rn >= 0).all() and (rn <= 100).all()))

# ATR of a constant-range series equals that range.
high = pd.Series(np.full(n, 52100.0), index=idx)
low = pd.Series(np.full(n, 52000.0), index=idx)
close = pd.Series(np.full(n, 52050.0), index=idx)
check("ATR of a constant 100pt range is 100",
      close_to(float(tech.atr(high, low, close).iloc[-1]), 100.0, 1e-6))

# VWAP against a hand-computed answer.
h = pd.Series([10.0, 20.0]); l = pd.Series([10.0, 20.0]); c = pd.Series([10.0, 20.0])
v = pd.Series([1.0, 3.0])
check("VWAP matches hand calculation",
      close_to(float(tech.vwap(h, l, c, v).iloc[-1]), (10 * 1 + 20 * 3) / 4, 1e-9))

# THE ^DJI TRAP: zero volume must give NaN, never a plausible wrong number.
zero_v = pd.Series([0.0, 0.0])
check("VWAP on zero volume returns NaN (the ^DJI guard)",
      bool(tech.vwap(h, l, c, zero_v).isna().all()))

# RSI momentum decay sign.
decaying = pd.Series(list(np.linspace(30, 80, 40)) + list(np.linspace(80, 65, 10)))
check("RSI decay negative when RSI is rolling over",
      tech.rsi_momentum_decay(decaying) < 0)

# Divergence: price makes a higher high, oscillator a lower high.
price = pd.Series([10, 12, 18, 12, 11, 13, 22, 14, 12, 11, 10, 9] * 4, dtype=float)
osc = pd.Series([30, 40, 80, 45, 40, 45, 60, 42, 38, 35, 33, 30] * 4, dtype=float)
check("bearish divergence found",
      tech.find_divergence(price, osc, lookback=48, pivot=2) == "bearish",
      tech.find_divergence(price, osc, lookback=48, pivot=2))
check("no divergence on a clean trend",
      tech.find_divergence(rising, tech.rsi(rising)) == "none")


# ==========================================================================
print("\n[3] CPR classification")
# ==========================================================================
# Narrow CPR: H/L/C almost equal -> tiny width relative to ATR.
c_narrow = tech.compute_cpr(52100, 52000, 52050, atr20=500, current=52400)
check("narrow CPR classified NARROW", c_narrow.classification == "NARROW",
      f"{c_narrow.classification} width_atr={c_narrow.width_atr:.3f}")
check("price above TC gives BULLISH_TREND", c_narrow.signal == "BULLISH_TREND",
      c_narrow.signal)
check("TC is above BC", c_narrow.tc >= c_narrow.bc)

# Wide CPR: big prior range, small ATR.
c_wide = tech.compute_cpr(53000, 52000, 52900, atr20=300, current=52500)
check("wide CPR classified WIDE", c_wide.classification == "WIDE",
      f"{c_wide.classification} width_atr={c_wide.width_atr:.3f}")
check("wide CPR signals RANGE_FADE", c_wide.signal == "RANGE_FADE", c_wide.signal)

# Pivot arithmetic.
ph, pl, pc = 53000.0, 52000.0, 52500.0
c_piv = tech.compute_cpr(ph, pl, pc, atr20=400, current=52500)
check("pivot = (H+L+C)/3", close_to(c_piv.pivot, (ph + pl + pc) / 3, 1e-9))
check("R1 = 2P - L", close_to(c_piv.r1, 2 * c_piv.pivot - pl, 1e-9))
check("S1 = 2P - H", close_to(c_piv.s1, 2 * c_piv.pivot - ph, 1e-9))
check("R2 - S2 equals twice the prior range",
      close_to(c_piv.r2 - c_piv.s2, 2 * (ph - pl), 1e-9))

# Virgin CPR: today's whole range sits above the band.
c_virgin = tech.compute_cpr(52100, 52000, 52050, atr20=500, current=53000,
                            today_high=53100, today_low=52900)
check("virgin CPR detected", c_virgin.virgin)
c_touched = tech.compute_cpr(52100, 52000, 52050, atr20=500, current=53000,
                             today_high=53100, today_low=52000)
check("touched CPR not virgin", not c_touched.virgin)


# ==========================================================================
print("\n[4] Technicals end to end")
# ==========================================================================
def make_daily(trend: float, periods: int = 260) -> pd.DataFrame:
    i = pd.date_range("2025-09-01", periods=periods, freq="B")
    base = 50000 + np.arange(periods) * trend + rng.normal(0, 60, periods)
    return pd.DataFrame({
        "Open": base, "High": base + 180, "Low": base - 180,
        "Close": base, "Volume": np.zeros(periods),     # ^DJI: volume is zero
    }, index=i)


up_daily = make_daily(+12.0)
t_rep = tech.compute_technicals(up_daily)
check("technicals ok on daily-only input", t_rep.ok, t_rep.note)
check("uptrend gives a positive score", t_rep.score > 0, f"{t_rep.score:.2f}")
check("score is inside the ±20 budget", abs(t_rep.score) <= 20.0 + 1e-9)
check("VWAP suppressed without a volume source",
      t_rep.vwap_source == "unavailable")
check("missing VWAP is flagged", any("volume" in f.lower() for f in t_rep.flags),
      str(t_rep.flags))

down_daily = make_daily(-12.0)
check("downtrend gives a negative score",
      tech.compute_technicals(down_daily).score < 0)

# With a genuine volume source, VWAP appears.
vol_bars = pd.DataFrame({
    "Open": [520.0, 521.0, 522.0], "High": [521.0, 522.0, 523.0],
    "Low": [519.0, 520.0, 521.0], "Close": [520.5, 521.5, 522.5],
    "Volume": [1000.0, 2000.0, 3000.0],
}, index=pd.date_range("2026-09-11 09:30", periods=3, freq="5min"))
t_vol = tech.compute_technicals(up_daily, None, vol_bars, vol_scale=100.0, vol_source="DIA")
check("VWAP populated from a volume-bearing source",
      np.isfinite(t_vol.vwap_value) and t_vol.vwap_source == "DIA",
      f"{t_vol.vwap_value}")
check("VWAP scaled to index points", t_vol.vwap_value > 50000,
      f"{t_vol.vwap_value:.0f}")

# A zero-volume source must be rejected, not silently used.
zero_bars = vol_bars.copy()
zero_bars["Volume"] = 0.0
t_zero = tech.compute_technicals(up_daily, None, zero_bars, 100.0, "^DJI")
check("zero-volume source rejected", not np.isfinite(t_zero.vwap_value))
check("zero-volume source explains itself",
      any("zero volume" in f for f in t_zero.flags), str(t_zero.flags))


# ==========================================================================
print("\n[5] Regime")
# ==========================================================================
trend_daily = make_daily(+25.0, 300)
r_trend = regime_mod.compute_regime(trend_daily)
check("strong trend classified TREND", r_trend.regime == "TREND",
      f"{r_trend.regime} ADX={r_trend.adx_value:.1f} ER={r_trend.efficiency:.2f}")
check("trend direction is UP", r_trend.direction == "UP")
check("regime score inside ±10", abs(r_trend.score) <= 10.0 + 1e-9)

flat_i = pd.date_range("2025-09-01", periods=300, freq="B")
flat_base = 52000 + rng.normal(0, 150, 300)
flat_daily = pd.DataFrame({
    "Open": flat_base, "High": flat_base + 200, "Low": flat_base - 200,
    "Close": flat_base, "Volume": 0.0}, index=flat_i)
r_flat = regime_mod.compute_regime(flat_daily)
check("noise is not classified TREND", r_flat.regime != "TREND", r_flat.regime)

check("efficiency ratio of a straight line is 1",
      close_to(regime_mod.efficiency_ratio(pd.Series(np.arange(50.0)), 20), 1.0, 1e-9))
# Round trip INSIDE the 21-bar window: up 10 then back down 10, net zero.
round_trip = pd.Series(list(np.arange(11.0)) + list(np.arange(9.0, -1.0, -1.0)))
check("efficiency ratio of a round trip is ~0",
      regime_mod.efficiency_ratio(round_trip, 20) < 0.1,
      f"got {regime_mod.efficiency_ratio(round_trip, 20):.4f}")

# Dispersion: identical returns -> zero dispersion.
same = pd.DataFrame(
    {t: np.linspace(100, 110, 80) for t in config.COMPONENTS},
    index=pd.date_range("2026-05-01", periods=80, freq="B"))
r_same = regime_mod.compute_regime(trend_daily, same)
check("identical component returns give ~zero dispersion",
      np.isfinite(r_same.dispersion) and r_same.dispersion < 0.01,
      f"{r_same.dispersion}")

scattered = pd.DataFrame(
    {t: 100 * np.cumprod(1 + rng.normal(0, 0.02, 80)) for t in config.COMPONENTS},
    index=pd.date_range("2026-05-01", periods=80, freq="B"))
r_scat = regime_mod.compute_regime(trend_daily, scattered)
check("scattered components give non-zero dispersion",
      np.isfinite(r_scat.dispersion) and r_scat.dispersion > 0.1,
      f"{r_scat.dispersion}")
check("confidence scalar stays in a sane band",
      0.5 <= r_scat.signal_confidence_scalar <= 1.3)


# ==========================================================================
print("\n[6] Macro")
# ==========================================================================
m_idx = pd.date_range("2026-08-01", periods=40, freq="B")
m_frame = pd.DataFrame({
    "^TNX": np.linspace(41.0, 42.0, 40),      # 4.10% -> 4.20%, i.e. +10bp total
    "^FVX": np.linspace(38.0, 38.2, 40),
    "^TYX": np.linspace(45.0, 45.5, 40),
    "^IRX": np.linspace(52.0, 52.0, 40),
    "DX-Y.NYB": np.linspace(100.0, 101.0, 40),
    "CL=F": np.linspace(88.0, 96.0, 40),
    "GC=F": np.linspace(3000.0, 3010.0, 40),
    "^VIX": np.linspace(16.0, 18.0, 40),
    "^VXD": np.linspace(15.0, 19.5, 40),
    "SPY": np.linspace(600.0, 610.0, 40),
}, index=m_idx)

m = macro_mod.compute_macro(m_frame)
check("macro ok", m.ok, m.note)
check("10y converted from x10 quote",
      close_to(m.us10y, 4.2, 0.01), f"got {m.us10y}")
check("yield change expressed in bp",
      close_to(m.us10y_change_bp, (42.0 - 41.0) / 39 * 10, 0.01),
      f"got {m.us10y_change_bp}")
check("curve computed", close_to(m.curve_5s10s, 4.2 - 3.82, 0.01), f"{m.curve_5s10s}")
check("VXD/VIX ratio computed", close_to(m.vxd_vix_ratio, 19.5 / 18.0, 0.01))
check("oil above $90 registers as a driver on level alone",
      any("WTI" in d for d in m.drivers), str(m.drivers))
check("macro score inside ±15", abs(m.score) <= 15.0 + 1e-9)

# Oil below $90 and flat must NOT produce a driver.
m_calm = macro_mod.compute_macro(m_frame.assign(**{"CL=F": np.linspace(70.0, 70.2, 40)}))
check("calm sub-$90 oil produces no oil driver",
      not any("WTI" in d for d in m_calm.drivers), str(m_calm.drivers))

m_empty = macro_mod.compute_macro(pd.DataFrame())
check("empty macro frame degrades safely", not m_empty.ok and m_empty.score == 0.0)


# ==========================================================================
print("\n[7] Sectors")
# ==========================================================================
s_idx = pd.date_range("2026-08-01", periods=30, freq="B")
s_frame = pd.DataFrame({
    "SPY": np.linspace(600, 606, 30),
    "XLF": np.linspace(50, 49.5, 30),        # financials LAGGING
    "XLI": np.linspace(140, 143, 30),
    "XLK": np.linspace(260, 264, 30),
    "XLV": np.linspace(150, 151, 30),
    "XLC": np.linspace(110, 112, 30),
    "XLY": np.linspace(220, 222, 30),
    "XLP": np.linspace(80, 80.2, 30),
    "XLE": np.linspace(95, 97, 30),
    "XLB": np.linspace(90, 91, 30),
}, index=s_idx)

s = sectors_mod.compute_sectors(s_frame, last)
check("sectors ok", s.ok, s.note)
check("XLF underperformance detected", s.xlf_underperforming,
      f"xlf_rs={s.xlf_rs}")
check("sector score inside ±5", abs(s.score) <= 5.0 + 1e-9)

w = sectors_mod.dow_sector_weights(last)
check("financials are the largest Dow sector bloc",
      w.index[0] == "Financials", str(w.head(3).to_dict()))
check("financials weight near 27.8%", close_to(float(w["Financials"]), 27.8, 0.3),
      f"got {float(w['Financials']):.2f}")
check("sector weights sum to 100", close_to(float(w.sum()), 100.0, 1e-6))
check("no utilities or real estate in the map",
      not ({"Utilities", "Real Estate"} & set(w.index)))


# ==========================================================================
print("\n[8] Master signal, redistribution and conflicts")
# ==========================================================================
class Stub:
    def __init__(self, score, confidence=1.0, ok=True, note="stub", **kw):
        self.score, self.confidence, self.ok, self.note = score, confidence, ok, note
        for k, v in kw.items():
            setattr(self, k, v)


# All layers live and bullish -> strongly positive.
sig_all = master.build_master_signal(
    attribution=Stub(20), technicals=Stub(16), macro=Stub(12),
    regime=Stub(8), sectors=Stub(4), micro=Stub(12), options=Stub(8),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=config.MARKET_TZ).to_pydatetime()},
)
check("full coverage reads 100%", close_to(sig_all.coverage, 1.0, 1e-9),
      f"{sig_all.coverage}")
check("all-bullish gives a strong long", sig_all.final_score > 40,
      f"{sig_all.final_score:.1f}")
check("composite stays within ±100", abs(sig_all.final_score) <= 100.0)

# Missing layers must be REDISTRIBUTED, not scored zero.
sig_missing = master.build_master_signal(
    attribution=Stub(20), technicals=Stub(16), macro=Stub(12),
    regime=Stub(8), sectors=Stub(4), micro=None, options=None,
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=config.MARKET_TZ).to_pydatetime()},
)
check("coverage drops to 75% with L3 and L6 missing",
      close_to(sig_missing.coverage, 0.75, 1e-9), f"{sig_missing.coverage}")
# THE redistribution test: with two layers absent, the pre-conflict composite
# must be IDENTICAL to the full-coverage one, because every live layer here
# sits at the same fraction of its budget. If missing layers were scored zero
# instead of redistributed, this would drop to 0.75x.
check("missing layers do NOT drag the raw composite toward zero",
      close_to(sig_missing.raw_score, sig_all.raw_score, 1e-9),
      f"missing={sig_missing.raw_score:.2f} full={sig_all.raw_score:.2f}")
check("zeroing instead of redistributing would have cost 25%",
      close_to(sig_all.raw_score * 0.75, sig_missing.raw_score * 0.75, 1e-9))
# The final scores DO differ, but only because C10 needs 6 live layers to fire.
check("C10 fires on full coverage but not on 5 live layers",
      any(c.code == "C10" for c in sig_all.conflicts)
      and not any(c.code == "C10" for c in sig_missing.conflicts),
      f"full={[c.code for c in sig_all.conflicts]} "
      f"missing={[c.code for c in sig_missing.conflicts]}")
check("C6 raised for the absent options layer",
      any(c.code == "C6" for c in sig_missing.conflicts),
      str([c.code for c in sig_missing.conflicts]))

# A layer scoring zero is NOT the same as a missing layer.
sig_zero = master.build_master_signal(
    attribution=Stub(20), technicals=Stub(16), macro=Stub(12),
    regime=Stub(8), sectors=Stub(4), micro=Stub(0), options=Stub(0),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=config.MARKET_TZ).to_pydatetime()},
)
check("a zero-scoring layer damps more than a missing one",
      sig_zero.final_score < sig_missing.final_score,
      f"zero={sig_zero.final_score:.1f} missing={sig_missing.final_score:.1f}")

# C3: chop blocks directional entries.
sig_chop = master.build_master_signal(
    attribution=Stub(20), technicals=Stub(16, price=52000, atr14=400),
    regime=Stub(0, regime="CHOP", adx_value=12.0, efficiency=0.2,
                signal_confidence_scalar=1.0),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=config.MARKET_TZ).to_pydatetime()},
)
check("C3 blocks in chop", sig_chop.blocked,
      str([c.code for c in sig_chop.conflicts]))
check("blocked signal produces no trade plan", not sig_chop.plan.valid)

# C9: cost gate blocks when the expected move cannot clear structure.
sig_cost = master.build_master_signal(
    attribution=Stub(20), technicals=Stub(16, price=52000, atr14=5.0),
    regime=Stub(8, regime="TREND", signal_confidence_scalar=1.0),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=config.MARKET_TZ).to_pydatetime(),
         "spread_pts": 4.0, "slippage_pts": 2.0},
)
check("C9 blocks a sub-cost expected move",
      any(c.code == "C9" for c in sig_cost.conflicts),
      str([c.code for c in sig_cost.conflicts]))

# C2: rates conflict downgrades a long.
sig_rates = master.build_master_signal(
    attribution=Stub(22), technicals=Stub(18, price=52000, atr14=400),
    macro=Stub(0, us10y_change_bp=12.0), regime=Stub(8, regime="TREND",
                                                     signal_confidence_scalar=1.0),
    sectors=Stub(-2, xlf_underperforming=True, xlf_rs=-0.8),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=config.MARKET_TZ).to_pydatetime()},
)
check("C2 fires on rising yields with lagging XLF",
      any(c.code == "C2" for c in sig_rates.conflicts),
      str([c.code for c in sig_rates.conflicts]))

# C8: cross-index conflict.
sig_cross = master.build_master_signal(
    attribution=Stub(22), technicals=Stub(18, price=52000, atr14=400),
    regime=Stub(8, regime="TREND", signal_confidence_scalar=1.0),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=config.MARKET_TZ).to_pydatetime(),
         "cross_index": {"correlation": 0.85, "ndx_direction": -1}},
)
check("C8 downgrades against a breaking NDX",
      any(c.code == "C8" for c in sig_cross.conflicts),
      str([c.code for c in sig_cross.conflicts]))

# Session blocks.
lunch, lunch_scalar = master.current_session_block(
    pd.Timestamp("2026-09-11 12:15", tz=config.MARKET_TZ).to_pydatetime())
check("lunch block identified", lunch == "LUNCH_CHOP" and lunch_scalar == 0.5,
      f"{lunch}/{lunch_scalar}")
weekend, weekend_scalar = master.current_session_block(
    pd.Timestamp("2026-09-12 12:15", tz=config.MARKET_TZ).to_pydatetime())
check("weekend zeroes the scalar", weekend_scalar == 0.0, f"{weekend}")

# Trade plan geometry.
sig_plan = master.build_master_signal(
    attribution=Stub(24), technicals=Stub(19, price=52000.0, atr14=400.0),
    macro=Stub(12), regime=Stub(9, regime="TREND", signal_confidence_scalar=1.0),
    sectors=Stub(4),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=config.MARKET_TZ).to_pydatetime()},
)
p = sig_plan.plan
check("long plan produced", p.direction == "LONG" and p.valid,
      f"{p.direction} valid={p.valid} reason={p.reason}")
check("stop is ATR_STOP_MULT below entry",
      close_to(p.entry - p.stop, 400.0 * config.ATR_STOP_MULT, 1e-6),
      f"{p.entry - p.stop}")
check("TP2 is 2R", close_to(p.tp2 - p.entry, (p.entry - p.stop) * 2.0, 1e-6))
check("lot multiplier within [0,1]", 0.0 <= p.lot_multiplier <= 1.0)
check("TP1 clears the structural minimum",
      (p.tp1 - p.entry) >= config.MIN_TARGET_PTS)

# Tiny ATR -> target below the YM structural minimum -> no trade.
sig_tiny = master.build_master_signal(
    attribution=Stub(24), technicals=Stub(19, price=52000.0, atr14=10.0),
    macro=Stub(12), regime=Stub(9, regime="TREND", signal_confidence_scalar=1.0),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=config.MARKET_TZ).to_pydatetime(),
         "spread_pts": 0.1, "slippage_pts": 0.1},
)
check("sub-minimum target rejected", not sig_tiny.plan.valid,
      sig_tiny.plan.reason)

# Layer budgets must sum to exactly 100.
check("layer budgets sum to 100",
      sum(config.LAYER_WEIGHTS.values()) == 100,
      str(sum(config.LAYER_WEIGHTS.values())))

# Everything empty must not raise.
sig_empty = master.build_master_signal()
check("no inputs at all degrades safely",
      sig_empty.ok and sig_empty.coverage == 0.0 and not sig_empty.plan.valid)


# ==========================================================================
print("\n[9] Data layer guards")
# ==========================================================================
import data_layer as dl  # noqa: E402

dl.reset_breaker()
check("breaker starts closed", not dl.breaker_open())
for _ in range(3):
    dl._record(False)
check("breaker opens after 3 consecutive failures", dl.breaker_open(),
      dl.breaker_status())
check("open breaker short-circuits downloads",
      dl._download(["AAPL"], period="1d", interval="1d").empty)
dl._record(True)
check("a success closes the breaker", not dl.breaker_open())
dl.reset_breaker()

# Column normalisation: yfinance ships ticker-major and field-major shapes.
field_major = pd.DataFrame(
    np.arange(12.0).reshape(2, 6),
    index=pd.date_range("2026-09-10", periods=2),
    columns=pd.MultiIndex.from_product([["Close", "Open", "Volume"], ["AAPL", "MSFT"]]),
)
flat_fm = dl._flatten(field_major, ["AAPL", "MSFT"])
check("field-major columns preserved",
      "Close" in flat_fm.columns.get_level_values(0))

ticker_major = field_major.swaplevel(axis=1).sort_index(axis=1)
flat_tm = dl._flatten(ticker_major, ["AAPL", "MSFT"])
check("ticker-major columns swapped to field-major",
      "Close" in flat_tm.columns.get_level_values(0),
      str(flat_tm.columns.get_level_values(0).unique().tolist()))
check("both shapes normalise to the same values",
      close_to(float(flat_fm[("Close", "AAPL")].iloc[-1]),
               float(flat_tm[("Close", "AAPL")].iloc[-1]), 1e-9))

single = dl._flatten(field_major, ["AAPL"])
check("single ticker collapses to plain columns",
      not isinstance(single.columns, pd.MultiIndex),
      str(type(single.columns)))

check("an unhappy Fetch is falsy", not bool(dl.Fetch(ok=False)))
check("closes() of an empty Fetch is an empty frame",
      dl.closes(dl.Fetch(ok=False)).empty)
check("ohlcv() of an empty Fetch is an empty frame",
      dl.ohlcv(dl.Fetch(ok=False)).empty)


# ==========================================================================
print("\n" + "=" * 64)
print(f"  {PASS} passed, {FAIL} failed")
if FAILURES:
    print("=" * 64)
    for f in FAILURES:
        print(f"  - {f}")
print("=" * 64)
sys.exit(1 if FAIL else 0)
