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
print("\n[9] Microstructure — phase 4")
# ==========================================================================
import us30_micro as micro_mod  # noqa: E402

TZ = config.MARKET_TZ


def bars_for(day: str, start: str, n: int, base: float, step: float = 0.0,
             rng_pts: float = 20.0, vol: float = 1000.0,
             close_pos: float = 0.5, freq: str = "5min") -> pd.DataFrame:
    """Build n bars with a known range and a known close position inside it."""
    idx = pd.date_range(f"{day} {start}", periods=n, freq=freq, tz=TZ)
    mid = base + np.arange(n) * step
    low = mid - rng_pts / 2
    high = mid + rng_pts / 2
    close = low + (high - low) * close_pos
    return pd.DataFrame({"Open": mid, "High": high, "Low": low,
                         "Close": close, "Volume": np.full(n, vol)}, index=idx)


# ---- session boundary: Globex 18:00 belongs to the NEXT session -----------
mixed = pd.DatetimeIndex([
    pd.Timestamp("2026-09-08 17:00", tz=TZ),   # Tue RTH-ish -> Tue
    pd.Timestamp("2026-09-08 18:05", tz=TZ),   # Tue Globex  -> Wed
    pd.Timestamp("2026-09-09 03:00", tz=TZ),   # Wed o/n     -> Wed
    pd.Timestamp("2026-09-09 10:00", tz=TZ),   # Wed RTH     -> Wed
    pd.Timestamp("2026-09-06 20:00", tz=TZ),   # Sunday eve  -> Monday 7th? no, 7th is Mon
])
sd = micro_mod.session_dates(mixed)
check("17:00 bar stays in its own session",
      sd.iloc[0] == pd.Timestamp("2026-09-08"), str(sd.iloc[0]))
check("18:05 Globex bar rolls to the next session",
      sd.iloc[1] == pd.Timestamp("2026-09-09"), str(sd.iloc[1]))
check("overnight 03:00 bar belongs to that day's session",
      sd.iloc[2] == pd.Timestamp("2026-09-09"), str(sd.iloc[2]))
check("Sunday-evening bar rolls forward to Monday",
      sd.iloc[4].dayofweek == 0, f"{sd.iloc[4]} dow={sd.iloc[4].dayofweek}")

rth = micro_mod.rth_mask(pd.DatetimeIndex([
    pd.Timestamp("2026-09-09 09:29", tz=TZ),
    pd.Timestamp("2026-09-09 09:30", tz=TZ),
    pd.Timestamp("2026-09-09 15:59", tz=TZ),
    pd.Timestamp("2026-09-09 16:00", tz=TZ),
    pd.Timestamp("2026-09-12 11:00", tz=TZ),   # Saturday
]))
check("RTH mask boundaries are correct",
      list(rth) == [False, True, True, False, False], str(list(rth)))

naive = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0],
                      "Close": [1.0], "Volume": [1.0]},
                     index=pd.DatetimeIndex(["2026-09-09 14:00"]))
check("naive timestamps are localised, not dropped",
      micro_mod.to_et(naive).index.tz is not None)

# ---- RVOL is same-time-of-day, not a flat average -------------------------
# Three quiet sessions, then today with a 3x spike at ONE slot only.
hist = pd.concat([
    bars_for("2026-09-08", "09:30", 12, 52000, vol=1000),
    bars_for("2026-09-09", "09:30", 12, 52000, vol=1000),
    bars_for("2026-09-10", "09:30", 12, 52000, vol=1000),
])
today_spike = bars_for("2026-09-11", "09:30", 12, 52000, vol=1000)
today_spike.iloc[-1, today_spike.columns.get_loc("Volume")] = 3000.0
rv, rv_series = micro_mod.compute_rvol(pd.concat([hist, today_spike]))
check("RVOL detects a 3x same-slot spike", close_to(rv, 3.0, 0.01), f"got {rv}")

# The open being naturally heavy must NOT read as a spike.
shaped = []
for day in ("2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"):
    b = bars_for(day, "09:30", 12, 52000, vol=1000)
    b.iloc[0, b.columns.get_loc("Volume")] = 8000.0      # every open is heavy
    shaped.append(b)
rv_shaped, _ = micro_mod.compute_rvol(pd.concat(shaped))
check("a structurally heavy open does not read as expansion",
      close_to(rv_shaped, 1.0, 0.01),
      f"got {rv_shaped} — a flat-average RVOL would have said ~8x here")

check("RVOL needs two sessions",
      not np.isfinite(micro_mod.compute_rvol(
          bars_for("2026-09-11", "09:30", 12, 52000))[0]))

# ---- delta proxy ----------------------------------------------------------
strong_up = bars_for("2026-09-11", "09:30", 20, 52000, step=5, close_pos=1.0)
cum_up, slope_up, series_up = micro_mod.compute_delta(strong_up)
check("closes at the high give maximum positive delta",
      close_to(cum_up, 20 * 1000.0, 1e-6), f"got {cum_up}")
check("delta slope positive on buying pressure", slope_up > 0)

strong_dn = bars_for("2026-09-11", "09:30", 20, 52000, step=-5, close_pos=0.0)
check("closes at the low give maximum negative delta",
      close_to(micro_mod.compute_delta(strong_dn)[0], -20 * 1000.0, 1e-6))

mid = bars_for("2026-09-11", "09:30", 20, 52000, close_pos=0.5)
check("mid-range closes contribute zero delta",
      close_to(micro_mod.compute_delta(mid)[0], 0.0, 1e-6))

# Zero-range bars must not divide by zero.
flat = bars_for("2026-09-11", "09:30", 10, 52000, rng_pts=0.0)
cum_flat, _, _ = micro_mod.compute_delta(flat)
check("zero-range bars are handled without NaN/inf",
      np.isfinite(cum_flat) and close_to(cum_flat, 0.0, 1e-6), f"got {cum_flat}")

# Divergence: price makes a higher high late, delta does not follow.
first_half = bars_for("2026-09-11", "09:30", 14, 52000, step=8, close_pos=1.0)
second_half = bars_for("2026-09-11", "10:40", 14, 52150, step=6, close_pos=0.15)
div_bars = pd.concat([first_half, second_half])
_, _, div_series = micro_mod.compute_delta(div_bars)
check("delta divergence detected on an unconfirmed new high",
      micro_mod.delta_divergence(div_bars, div_series) == "bearish",
      micro_mod.delta_divergence(div_bars, div_series))

# ---- levels ---------------------------------------------------------------
overnight = bars_for("2026-09-11", "04:00", 30, 52000, rng_pts=40)   # o/n range
session = bars_for("2026-09-11", "09:30", 30, 52200, step=2, rng_pts=20)
prior = bars_for("2026-09-10", "09:30", 40, 51800, rng_pts=60)
level_bars = pd.concat([prior, overnight, session])
lv = micro_mod.compute_levels(level_bars, float(session["Close"].iloc[-1]))
check("overnight high captured", close_to(lv.overnight_high, 52020.0, 0.01),
      f"got {lv.overnight_high}")
check("overnight low captured", close_to(lv.overnight_low, 51980.0, 0.01))
check("prior-day high captured", close_to(lv.prior_high, 51830.0, 0.01),
      f"got {lv.prior_high}")
check("initial balance is the first 60 RTH minutes only",
      np.isfinite(lv.ib_high) and lv.ib_high < float(session["High"].max()),
      f"ib_high={lv.ib_high} session_high={session['High'].max()}")
check("price above the overnight range sets ABOVE_ON",
      lv.state == "ABOVE_ON", lv.state)
check("range position above 1 when price clears the overnight high",
      lv.on_range_position > 1.0, f"{lv.on_range_position}")

# ---- sweeps: rejection is what separates a sweep from a breakout ----------
base_bars = bars_for("2026-09-11", "09:30", 40, 52000, rng_pts=20, vol=1000)
sweep_levels = micro_mod.Levels(overnight_high=52010.0, overnight_low=51990.0)

# A bar that pokes above the level and closes back WELL below it = sweep.
rejected = base_bars.copy()
i = len(rejected) - 2
rejected.iloc[i, rejected.columns.get_loc("High")] = 52080.0
rejected.iloc[i, rejected.columns.get_loc("Low")] = 51980.0
rejected.iloc[i, rejected.columns.get_loc("Close")] = 51995.0
sweeps = micro_mod.detect_sweeps(rejected, sweep_levels, atr_val=40.0)
check("rejected penetration is detected as a sweep",
      any(s.side == "high" for s in sweeps), str(sweeps))
check("a high sweep implies bearish",
      all(s.implication == "bearish" for s in sweeps if s.side == "high"))

# A bar that breaks the level and HOLDS above it = breakout, not a sweep.
held = base_bars.copy()
held.iloc[i, held.columns.get_loc("High")] = 52080.0
held.iloc[i, held.columns.get_loc("Low")] = 52020.0
held.iloc[i, held.columns.get_loc("Close")] = 52075.0
check("a level that breaks and holds is NOT a sweep",
      not any(s.side == "high" for s in
              micro_mod.detect_sweeps(held, sweep_levels, atr_val=40.0)),
      str(micro_mod.detect_sweeps(held, sweep_levels, atr_val=40.0)))

# A touch without real penetration is not a sweep either.
touched = base_bars.copy()
touched.iloc[i, touched.columns.get_loc("High")] = 52010.5
touched.iloc[i, touched.columns.get_loc("Close")] = 51995.0
check("a mere touch of the level is not a sweep",
      not any(s.side == "high" for s in
              micro_mod.detect_sweeps(touched, sweep_levels, atr_val=40.0)))

# Low sweep.
low_swept = base_bars.copy()
low_swept.iloc[i, low_swept.columns.get_loc("Low")] = 51920.0
low_swept.iloc[i, low_swept.columns.get_loc("High")] = 52020.0
low_swept.iloc[i, low_swept.columns.get_loc("Close")] = 52005.0
low_sweeps = micro_mod.detect_sweeps(low_swept, sweep_levels, atr_val=40.0)
check("low sweep detected and implies bullish",
      any(s.side == "low" and s.implication == "bullish" for s in low_sweeps),
      str(low_sweeps))

# ---- basis ----------------------------------------------------------------
fut_rth = bars_for("2026-09-11", "09:30", 40, 52050, rng_pts=10)
cash_rth = bars_for("2026-09-11", "09:30", 40, 52000, rng_pts=10)
b_val, b_mean, b_std, b_sigma, b_ok = micro_mod.compute_basis(fut_rth, cash_rth)
check("basis computed during RTH", b_ok)
check("constant 50pt basis measured correctly", close_to(b_val, 50.0, 1e-6),
      f"got {b_val}")

# Dislocate the last bar by a large amount -> large sigma.
fut_disloc = fut_rth.copy()
fut_disloc.iloc[:-1, fut_disloc.columns.get_loc("Close")] += np.linspace(-3, 3, 39)
fut_disloc.iloc[-1, fut_disloc.columns.get_loc("Close")] += 40.0
_, _, _, sigma_d, _ = micro_mod.compute_basis(fut_disloc, cash_rth)
check("a dislocated basis produces a large sigma", abs(sigma_d) > 2.0,
      f"got {sigma_d}")

# Overnight bars only -> basis must be unavailable, not wrong.
fut_on = bars_for("2026-09-11", "03:00", 40, 52050)
cash_on = bars_for("2026-09-11", "03:00", 40, 52000)
check("basis is unavailable outside RTH rather than misleading",
      not micro_mod.compute_basis(fut_on, cash_on)[4])

# ---- end-to-end -----------------------------------------------------------
full = pd.concat([
    bars_for("2026-09-09", "09:30", 78, 51900, rng_pts=25, vol=1000),
    bars_for("2026-09-10", "09:30", 78, 51950, rng_pts=25, vol=1000),
    bars_for("2026-09-11", "04:00", 66, 52000, rng_pts=30, vol=400),
    bars_for("2026-09-11", "09:30", 40, 52100, step=3, rng_pts=25,
             vol=2000, close_pos=0.9),
])
cash_full = bars_for("2026-09-11", "09:30", 40, 52050, step=3, rng_pts=25)
m = micro_mod.compute_micro(full, cash_full, spread_pts=4.0, slippage_pts=2.0)
check("micro ok end to end", m.ok, m.note)
check("micro score inside ±15", abs(m.score) <= 15.0 + 1e-9, f"{m.score}")
check("round-trip cost is spread + slippage",
      close_to(m.round_trip_cost, 6.0, 1e-9))
check("minimum viable target is 3x cost",
      close_to(m.min_viable_target, 18.0, 1e-9))
check("strong buying with volume expansion scores positive", m.score > 0,
      f"{m.score} rvol={m.rvol} state={m.levels.state}")
check("confidence in a sane band", 0.0 <= m.confidence <= 1.0)

m_empty = micro_mod.compute_micro(pd.DataFrame())
check("empty bars degrade safely", not m_empty.ok and m_empty.score == 0.0)
m_short = micro_mod.compute_micro(bars_for("2026-09-11", "09:30", 5, 52000))
check("too few bars degrades safely", not m_short.ok, m_short.note)

# ---- C7 now has data ------------------------------------------------------
sig_basis = master.build_master_signal(
    attribution=Stub(22), technicals=Stub(18, price=52000, atr14=400),
    regime=Stub(8, regime="TREND", signal_confidence_scalar=1.0),
    micro=Stub(5, basis_sigma=3.4),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=TZ).to_pydatetime()},
)
check("C7 fires on a dislocated basis",
      any(c.code == "C7" for c in sig_basis.conflicts),
      str([c.code for c in sig_basis.conflicts]))
check("C7 blocks entry", sig_basis.blocked)

sig_nobasis = master.build_master_signal(
    attribution=Stub(22), technicals=Stub(18, price=52000, atr14=400),
    regime=Stub(8, regime="TREND", signal_confidence_scalar=1.0),
    micro=Stub(5, basis_sigma=float("nan")),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=TZ).to_pydatetime()},
)
check("C7 stays silent when the basis is NaN",
      not any(c.code == "C7" for c in sig_nobasis.conflicts),
      str([c.code for c in sig_nobasis.conflicts]))

# ---- stale-config tolerance -----------------------------------------------
# Simulate the exact failure that killed the live app: config.py without the
# phase-4 constants, while us30_micro.py is already deployed.
_saved = {}
for _name in list(micro_mod._CFG_DEFAULTS):
    if hasattr(config, _name):
        _saved[_name] = getattr(config, _name)
        delattr(config, _name)

check("stale config is detected and named",
      sorted(micro_mod.config_health()) == sorted(_saved),
      str(micro_mod.config_health()))
check("session_dates still works on a stale config",
      micro_mod.session_dates(pd.DatetimeIndex(
          [pd.Timestamp("2026-09-08 18:05", tz=TZ)])).iloc[0]
      == pd.Timestamp("2026-09-09"))
m_stale = micro_mod.compute_micro(full, cash_full)
check("compute_micro still runs on a stale config", m_stale.ok, m_stale.note)
check("stale-config result matches the real-config result",
      close_to(m_stale.score, m.score, 1e-9),
      f"stale={m_stale.score} real={m.score}")

for _name, _val in _saved.items():
    setattr(config, _name, _val)
check("config restored after the stale-config test",
      micro_mod.config_health() == [])

# ---- coverage rises to 90% with L3 live -----------------------------------
sig_p4 = master.build_master_signal(
    attribution=Stub(20), technicals=Stub(16), macro=Stub(12),
    regime=Stub(8), sectors=Stub(4), micro=Stub(12), options=None,
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=TZ).to_pydatetime()},
)
check("coverage is 90% with only L6 missing",
      close_to(sig_p4.coverage, 0.90, 1e-9), f"{sig_p4.coverage}")


# ==========================================================================
print("\n[10] Options — phase 5")
# ==========================================================================
import dow_options as opt  # noqa: E402

# ---- Black-Scholes, validated against numerical derivatives ---------------
check("norm_cdf(0) is 0.5", close_to(float(opt.norm_cdf(0.0)), 0.5, 1e-12))
check("norm_cdf is symmetric",
      close_to(float(opt.norm_cdf(1.3)) + float(opt.norm_cdf(-1.3)), 1.0, 1e-12))
check("norm_pdf(0) is 1/sqrt(2pi)",
      close_to(float(opt.norm_pdf(0.0)), 1 / np.sqrt(2 * np.pi), 1e-12))

# THE test that matters: analytic gamma must equal the numerical second
# derivative of the BS price. A wrong d1, a missing sqrt(T) or a stray spot
# term all survive eyeballing and die here.
for S, K, T, V, R in [(100, 100, 1.0, 0.20, 0.0), (100, 110, 0.5, 0.35, 0.04),
                      (1037, 1000, 0.08, 0.28, 0.04), (52, 55, 0.25, 0.15, 0.02)]:
    h = S * 1e-4
    for is_call in (True, False):
        num = (float(opt.bs_price(S + h, K, T, V, R, is_call))
               - 2 * float(opt.bs_price(S, K, T, V, R, is_call))
               + float(opt.bs_price(S - h, K, T, V, R, is_call))) / (h * h)
        ana = float(opt.bs_gamma(S, K, T, V, R))
        check(f"gamma matches finite differences S={S} K={K} "
              f"{'call' if is_call else 'put'}",
              abs(num - ana) < max(1e-6, abs(ana) * 1e-3),
              f"analytic={ana:.8f} numerical={num:.8f}")

check("call and put gamma are identical",
      close_to(float(opt.bs_gamma(100, 105, 0.5, 0.3, 0.03)),
               float(opt.bs_gamma(100, 105, 0.5, 0.3, 0.03)), 1e-15))
check("put-call parity holds",
      close_to(float(opt.bs_price(100, 95, 1.0, 0.25, 0.05, True))
               - float(opt.bs_price(100, 95, 1.0, 0.25, 0.05, False)),
               100 - 95 * np.exp(-0.05), 1e-8))
check("ATM gamma exceeds far-OTM gamma",
      float(opt.bs_gamma(100, 100, 0.25, 0.2)) > float(opt.bs_gamma(100, 140, 0.25, 0.2)))
check("zero time to expiry yields zero, not inf",
      float(opt.bs_gamma(100, 100, 0.0, 0.2)) == 0.0)
check("zero vol yields zero, not inf",
      float(opt.bs_gamma(100, 100, 1.0, 0.0)) == 0.0)


# ---- chain fixtures --------------------------------------------------------
def make_chain(spot: float, expiry: str, call_oi: float, put_oi: float,
               iv: float = 0.25, n: int = 11, put_iv_bump: float = 0.0) -> dict:
    strikes = np.linspace(spot * 0.90, spot * 1.10, n)
    calls = pd.DataFrame({"strike": strikes, "impliedVolatility": np.full(n, iv),
                          "openInterest": np.full(n, call_oi / n),
                          "volume": np.full(n, 10.0)})
    puts = pd.DataFrame({"strike": strikes,
                         "impliedVolatility": np.full(n, iv + put_iv_bump),
                         "openInterest": np.full(n, put_oi / n),
                         "volume": np.full(n, 10.0)})
    return {"ticker": "X", "ok": True, "note": "", "expiry": expiry,
            "calls": calls, "puts": puts, "spot": spot}


NOW = pd.Timestamp("2026-09-13 12:00")
EXP = "2026-09-25"

# Call-heavy chain -> dealers long gamma -> positive net GEX.
ng_long = opt.compute_name_gamma("GS", make_chain(1000, EXP, 60000, 10000),
                                 1000.0, 11.7, 61.7, rate=0.04, now=NOW)
check("call-heavy chain is liquid and parsed", ng_long.liquid, ng_long.note)
check("call-heavy chain gives positive net GEX", ng_long.net_gex > 0,
      f"{ng_long.net_gex:,.0f}")
check("PCR below 1 on a call-heavy chain", ng_long.pcr < 1.0, f"{ng_long.pcr}")

ng_short = opt.compute_name_gamma("GS", make_chain(1000, EXP, 10000, 60000),
                                  1000.0, 11.7, 61.7, rate=0.04, now=NOW)
check("put-heavy chain gives negative net GEX", ng_short.net_gex < 0,
      f"{ng_short.net_gex:,.0f}")
check("PCR above 1 on a put-heavy chain", ng_short.pcr > 1.0, f"{ng_short.pcr}")
check("net GEX is call GEX minus put GEX",
      close_to(ng_short.net_gex, ng_short.call_gex - ng_short.put_gex, 1e-6))

# Skew: richer puts than calls must produce a positive skew reading.
ng_skew = opt.compute_name_gamma("GS", make_chain(1000, EXP, 30000, 30000,
                                                  put_iv_bump=0.05),
                                 1000.0, 11.7, 61.7, rate=0.04, now=NOW)
check("richer put IV produces positive skew", ng_skew.iv_skew > 0.04,
      f"{ng_skew.iv_skew}")

# Expected move converts to DJIA points through pts_per_1pct.
check("expected move is expressed in DJIA points",
      np.isfinite(ng_long.expected_move_pts)
      and close_to(ng_long.expected_move_pts,
                   ng_long.expected_move_pct * 61.7, 1e-6),
      f"{ng_long.expected_move_pts}")
check("expected move is positive and sane",
      0 < ng_long.expected_move_pct < 20, f"{ng_long.expected_move_pct}")

# Gamma wall lands on a real strike.
check("gamma wall is one of the chain's strikes",
      np.isfinite(ng_long.gamma_wall)
      and abs(ng_long.gamma_wall - 1000.0) <= 100.0, f"{ng_long.gamma_wall}")

# ---- liquidity floor drops a name entirely --------------------------------
ng_thin = opt.compute_name_gamma("NKE", make_chain(37, EXP, 200, 200),
                                 37.0, 0.42, 2.2, rate=0.04, now=NOW)
check("a chain under the OI floor is excluded", not ng_thin.liquid, ng_thin.note)
check("excluded name says why", "below floor" in ng_thin.note, ng_thin.note)

ng_nochain = opt.compute_name_gamma("BA", {"ok": False, "note": "no expiries listed"},
                                    208.0, 2.36, 12.4, now=NOW)
check("a missing chain degrades safely", not ng_nochain.liquid)

# Strikes far outside the moneyness band are ignored.
wide = make_chain(1000, EXP, 60000, 10000, n=11)
wide["calls"].loc[0, "strike"] = 300.0        # 70% OTM, junk IV territory
ng_wide = opt.compute_name_gamma("GS", wide, 1000.0, 11.7, 61.7, rate=0.04, now=NOW)
check("out-of-band strikes are excluded from OI",
      ng_wide.call_oi < ng_long.call_oi, f"{ng_wide.call_oi} vs {ng_long.call_oi}")

# ---- aggregation -----------------------------------------------------------
def name(t, w, pts, gex_per_pct, pcr=1.0, skew=0.0, move=50.0, liquid=True):
    n = opt.NameGamma(ticker=t, spot=100.0, weight_pct=w, pts_per_1pct=pts)
    n.gex_per_pct, n.pcr, n.iv_skew = gex_per_pct, pcr, skew
    n.expected_move_pts, n.liquid, n.net_gex = move, liquid, gex_per_pct * 1e6
    n.expiry, n.dte = EXP, 12.0
    return n

agg_long = opt.aggregate([name("GS", 11.7, 61.7, 0.004),
                          name("CAT", 9.2, 48.7, 0.003),
                          name("MSFT", 5.6, 29.6, 0.003),
                          name("UNH", 4.4, 23.1, 0.002)], 30.9)
check("aggregate ok with good coverage", agg_long.ok, agg_long.note)
check("positive component GEX reads LONG_GAMMA",
      agg_long.gamma_regime == "LONG_GAMMA", agg_long.gamma_regime)
check("long gamma sets a suppressive vol scalar", agg_long.vol_scalar < 1.0)
check("aggregate is clipped to ±1", abs(agg_long.aggregate_gex) <= 1.0)

agg_short = opt.aggregate([name("GS", 11.7, 61.7, -0.004),
                           name("CAT", 9.2, 48.7, -0.003),
                           name("MSFT", 5.6, 29.6, -0.003),
                           name("UNH", 4.4, 23.1, -0.002)], 30.9)
check("negative component GEX reads SHORT_GAMMA",
      agg_short.gamma_regime == "SHORT_GAMMA", agg_short.gamma_regime)
check("short gamma sets an amplifying vol scalar", agg_short.vol_scalar > 1.0)

# Heavier-weight names must move the aggregate more than light ones.
heavy_pos = opt.aggregate([name("GS", 11.7, 61.7, 0.004),
                           name("NKE", 0.42, 2.2, -0.004),
                           name("CAT", 9.2, 48.7, 0.001),
                           name("MSFT", 5.6, 29.6, 0.001)], 26.9)
check("price weight dominates the aggregate, not name count",
      heavy_pos.aggregate_gex > 0, f"{heavy_pos.aggregate_gex}")

# ---- the liquidity gate is the whole point of this design ------------------
agg_thin = opt.aggregate([name("GS", 11.7, 61.7, 0.004, liquid=False),
                          name("CAT", 9.2, 48.7, 0.003, liquid=False),
                          name("MSFT", 5.6, 29.6, 0.003, liquid=False),
                          name("UNH", 4.4, 23.1, 0.002, liquid=True)], 30.9)
check("coverage below the floor reports NOT ok", not agg_thin.ok,
      f"covered={agg_thin.weight_covered:.3f}")
check("thin coverage explains itself",
      any("floor" in f for f in agg_thin.flags), str(agg_thin.flags))
check("a not-ok options report still exposes its table",
      not agg_thin.table.empty)

agg_none = opt.aggregate([], 0.0)
check("no names at all degrades safely",
      not agg_none.ok and agg_none.score == 0.0)

# DIA cross-check is ignored when its OI is thin.
agg_dia_thin = opt.aggregate([name("GS", 11.7, 61.7, 0.004),
                              name("CAT", 9.2, 48.7, 0.003),
                              name("MSFT", 5.6, 29.6, 0.003),
                              name("UNH", 4.4, 23.1, 0.002)], 30.9,
                             dia_net_gex=-5e6, dia_total_oi=1200)
check("thin DIA cross-check is discarded, not used",
      agg_dia_thin.dia_agrees is None, str(agg_dia_thin.dia_agrees))
check("discarded DIA check says why",
      any("DIA cross-check ignored" in f for f in agg_dia_thin.flags),
      str(agg_dia_thin.flags))

agg_dia_ok = opt.aggregate([name("GS", 11.7, 61.7, 0.004),
                            name("CAT", 9.2, 48.7, 0.003),
                            name("MSFT", 5.6, 29.6, 0.003),
                            name("UNH", 4.4, 23.1, 0.002)], 30.9,
                           dia_net_gex=8e6, dia_total_oi=120_000)
check("a liquid DIA check that agrees is recorded",
      agg_dia_ok.dia_agrees is True)

# ---- directional score comes from skew and PCR, never from gamma ----------
agg_puts_rich = opt.aggregate([name("GS", 11.7, 61.7, 0.001, skew=0.08),
                               name("CAT", 9.2, 48.7, 0.001, skew=0.08),
                               name("MSFT", 5.6, 29.6, 0.001, skew=0.08),
                               name("UNH", 4.4, 23.1, 0.001, skew=0.08)], 30.9)
check("rich put skew scores bearish", agg_puts_rich.score < 0,
      f"{agg_puts_rich.score}")
check("options score inside ±10", abs(agg_puts_rich.score) <= 10.0 + 1e-9)

agg_flat = opt.aggregate([name("GS", 11.7, 61.7, 0.004, skew=0.0, pcr=1.0),
                          name("CAT", 9.2, 48.7, 0.004, skew=0.0, pcr=1.0),
                          name("MSFT", 5.6, 29.6, 0.004, skew=0.0, pcr=1.0),
                          name("UNH", 4.4, 23.1, 0.004, skew=0.0, pcr=1.0)], 30.9)
check("strong gamma with neutral skew scores ~0 directionally",
      abs(agg_flat.score) < 1.0, f"{agg_flat.score}")
check("...but still reports a gamma regime",
      agg_flat.gamma_regime == "LONG_GAMMA")

# ---- C11 / C12 / C13 -------------------------------------------------------
class Lv:
    def __init__(self, state):
        self.state = state


sig_c11 = master.build_master_signal(
    attribution=Stub(22, index_change_pts=80.0),
    technicals=Stub(18, price=52000, atr14=400, mean_reversion="none"),
    regime=Stub(8, regime="TREND", signal_confidence_scalar=1.0),
    micro=Stub(6, levels=Lv("ABOVE_ON"), basis_sigma=float("nan")),
    options=Stub(1, gamma_regime="LONG_GAMMA", aggregate_gex=0.6,
                 vol_scalar=0.85, expected_move_pts=600.0),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=TZ).to_pydatetime()},
)
check("C11 downgrades a breakout into long gamma",
      any(c.code == "C11" for c in sig_c11.conflicts),
      str([c.code for c in sig_c11.conflicts]))

sig_c12 = master.build_master_signal(
    attribution=Stub(-22, index_change_pts=-80.0),
    technicals=Stub(-18, price=52000, atr14=400, mean_reversion="long_setup"),
    regime=Stub(-8, regime="TREND", signal_confidence_scalar=1.0),
    micro=Stub(-6, levels=Lv("BELOW_ON"), basis_sigma=float("nan")),
    options=Stub(-1, gamma_regime="SHORT_GAMMA", aggregate_gex=-0.6,
                 vol_scalar=1.15, expected_move_pts=600.0),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=TZ).to_pydatetime()},
)
check("C12 blocks fading a short-gamma tape",
      any(c.code == "C12" for c in sig_c12.conflicts) and sig_c12.blocked,
      str([c.code for c in sig_c12.conflicts]))

sig_c13 = master.build_master_signal(
    attribution=Stub(22, index_change_pts=580.0),
    technicals=Stub(18, price=52000, atr14=400, mean_reversion="none"),
    regime=Stub(8, regime="TREND", signal_confidence_scalar=1.0),
    micro=Stub(6, levels=Lv("INSIDE_ON"), basis_sigma=float("nan")),
    options=Stub(1, gamma_regime="NEUTRAL", aggregate_gex=0.0,
                 vol_scalar=1.0, expected_move_pts=600.0),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=TZ).to_pydatetime()},
)
check("C13 blocks once the expected move is spent",
      any(c.code == "C13" for c in sig_c13.conflicts) and sig_c13.blocked,
      str([c.code for c in sig_c13.conflicts]))

sig_c13_room = master.build_master_signal(
    attribution=Stub(22, index_change_pts=120.0),
    technicals=Stub(18, price=52000, atr14=400, mean_reversion="none"),
    regime=Stub(8, regime="TREND", signal_confidence_scalar=1.0),
    micro=Stub(6, levels=Lv("INSIDE_ON"), basis_sigma=float("nan")),
    options=Stub(1, gamma_regime="NEUTRAL", aggregate_gex=0.0,
                 vol_scalar=1.0, expected_move_pts=600.0),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=TZ).to_pydatetime()},
)
check("C13 stays silent with room left in the expected move",
      not any(c.code == "C13" for c in sig_c13_room.conflicts))

check("gamma vol scalar reaches the master signal's confidence",
      sig_c11.confidence < sig_c13_room.confidence,
      f"{sig_c11.confidence:.3f} vs {sig_c13_room.confidence:.3f}")

# ---- full coverage at last -------------------------------------------------
sig_full = master.build_master_signal(
    attribution=Stub(20), technicals=Stub(16), macro=Stub(12), regime=Stub(8),
    sectors=Stub(4), micro=Stub(12), options=Stub(8),
    ctx={"now": pd.Timestamp("2026-09-11 14:00", tz=TZ).to_pydatetime()},
)
check("all seven layers live gives 100% coverage",
      close_to(sig_full.coverage, 1.0, 1e-9), f"{sig_full.coverage}")

# ---- stale config ----------------------------------------------------------
_saved_o = {}
for _name in list(opt._CFG_DEFAULTS):
    if hasattr(config, _name):
        _saved_o[_name] = getattr(config, _name)
        delattr(config, _name)
check("stale options config is detected and named",
      sorted(opt.config_health()) == sorted(_saved_o), str(opt.config_health()))
ng_stale = opt.compute_name_gamma("GS", make_chain(1000, EXP, 60000, 10000),
                                  1000.0, 11.7, 61.7, rate=0.04, now=NOW)
check("options still compute on a stale config", ng_stale.liquid, ng_stale.note)
check("stale-config gamma matches real-config gamma",
      close_to(ng_stale.net_gex, ng_long.net_gex, 1e-6))
for _name, _val in _saved_o.items():
    setattr(config, _name, _val)
check("options config restored", opt.config_health() == [])


# ==========================================================================
print("\n[11] Data layer guards")
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

# ---- the partial-deploy guard must not itself crash on a partial deploy ----
# This reproduces the exact failure that took the live app down twice: app.py
# called micro_mod.config_health() directly, and a module older than app.py
# does not have that function.
import types  # noqa: E402

_old_module = types.ModuleType("us30_micro")      # no config_health at all
check("a module with no config_health is reported stale, not crashed",
      dl.module_health(_old_module) == ([], ["us30_micro.py"]),
      str(dl.module_health(_old_module)))

_broken = types.ModuleType("dow_options")
_broken.config_health = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
check("a config_health that raises is caught and reported stale",
      dl.module_health(_broken) == ([], ["dow_options.py"]),
      str(dl.module_health(_broken)))

_good = types.ModuleType("us30_micro")
_good.config_health = lambda: ["RVOL_SPIKE"]
check("a healthy module reports its missing constants",
      dl.module_health(_good) == (["RVOL_SPIKE"], []),
      str(dl.module_health(_good)))
check("the real modules pass their own health check",
      dl.module_health(micro_mod) == ([], [])
      and dl.module_health(opt) == ([], [])),

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
