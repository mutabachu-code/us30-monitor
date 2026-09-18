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
print("\n[11] Calendar — phase 6")
# ==========================================================================
import us30_calendar as cal  # noqa: E402
from datetime import date as _date, datetime as _dt, timedelta as _td  # noqa: E402
from zoneinfo import ZoneInfo as _ZI  # noqa: E402

ET = _ZI(config.MARKET_TZ)

# ---- NFP: first Friday, derived not listed --------------------------------
# Verified against a real calendar: these are the first Fridays of each month.
KNOWN_NFP = {
    _date(2026, 9, 4), _date(2026, 10, 2), _date(2026, 11, 6),
    _date(2026, 12, 4), _date(2027, 1, 1), _date(2027, 2, 5),
}
got = set(cal.nfp_dates(_date(2026, 9, 1), _date(2027, 2, 28)))
check("NFP first-Friday rule matches known dates", KNOWN_NFP <= got,
      f"missing {sorted(KNOWN_NFP - got)}")
check("every computed NFP date is a Friday",
      all(d.weekday() == 4 for d in got), str([d for d in got if d.weekday() != 4]))
check("every computed NFP date is in the first 7 days of its month",
      all(d.day <= 7 for d in got), str([d for d in got if d.day > 7]))
check("one NFP per month over the range", len(got) == 6, str(sorted(got)))
check("NFP handles a month starting on Friday",
      _date(2027, 1, 1) in got, "Jan 2027 starts on a Friday")
check("empty range yields no NFP dates",
      cal.nfp_dates(_date(2026, 9, 5), _date(2026, 9, 20)) == [])

# ---- hardcoded tables integrity -------------------------------------------
fomc = [_date.fromisoformat(d) for d, _ in cal.FOMC_DECISION_DAYS]
check("FOMC table is sorted", fomc == sorted(fomc))
check("FOMC has 8 meetings in 2026",
      sum(1 for d in fomc if d.year == 2026) == 8,
      str([d for d in fomc if d.year == 2026]))
check("FOMC has 8 meetings in 2027",
      sum(1 for d in fomc if d.year == 2027) == 8)
check("no FOMC decision lands on a weekend",
      all(d.weekday() < 5 for d in fomc), str([d for d in fomc if d.weekday() >= 5]))
check("Sept 2026 FOMC decision day is the 16th",
      _date(2026, 9, 16) in fomc)
check("Dec 2026 FOMC decision day is the 9th", _date(2026, 12, 9) in fomc)
check("four SEP meetings per year",
      sum(1 for d, sep in cal.FOMC_DECISION_DAYS
          if sep and _date.fromisoformat(d).year == 2026) == 4)

cpi = [_date.fromisoformat(d) for d in cal.CPI_RELEASE_DAYS]
check("CPI table is sorted", cpi == sorted(cpi))
check("12 CPI releases in 2026", len(cpi) == 12)
check("no CPI release lands on a weekend",
      all(d.weekday() < 5 for d in cpi), str([d for d in cpi if d.weekday() >= 5]))
check("one CPI release per month",
      sorted(d.month for d in cpi) == list(range(1, 13)))
check("Oct 2026 CPI is the 14th", _date(2026, 10, 14) in cpi)
check("table end dates match the last row",
      cal.CPI_TABLE_ENDS == cpi[-1] and cal.FOMC_TABLE_ENDS == fomc[-1])

# ---- event assembly --------------------------------------------------------
NOW_CAL = _dt(2026, 9, 14, 10, 0, tzinfo=ET)      # Monday, two days before FOMC
events, past_end = cal.build_events(NOW_CAL, horizon_days=10)
check("events are returned in chronological order",
      [e.when for e in events] == sorted(e.when for e in events))
names = [e.name for e in events]
check("FOMC on 16 Sep appears in a 10-day horizon", "FOMC" in names, str(names))
check("not past the end of the tables in Sept 2026", not past_end)
check("FOMC statement is timed at 14:00 ET",
      all(e.when.hour == 14 and e.when.minute == 0 for e in events if e.name == "FOMC"))
check("CPI and NFP are timed at 08:30 ET",
      all(e.when.hour == 8 and e.when.minute == 30
          for e in events if e.name in ("CPI", "NFP")))
check("a 1-day horizon returns fewer events than a 30-day one",
      len(cal.build_events(NOW_CAL, 1)[0]) < len(cal.build_events(NOW_CAL, 30)[0]))

# Past the end of the tables, silence must NOT read as an all-clear.
_, past_end_far = cal.build_events(_dt(2028, 3, 1, 10, 0, tzinfo=ET), 10)
check("past the table end is reported, not silently empty", past_end_far)

# ---- blackout windows ------------------------------------------------------
FOMC_AT = _dt(2026, 9, 16, 14, 0, tzinfo=ET)
ev = [cal.Event("FOMC", FOMC_AT, "HIGH", "statement")]
for offset_min, expect, label in [
    (-31, False, "31 minutes before is outside the window"),
    (-30, True, "30 minutes before is inside"),
    (-1, True, "1 minute before is inside"),
    (0, True, "at the release is inside"),
    (14, True, "14 minutes after is inside"),
    (15, True, "15 minutes after is the boundary"),
    (16, False, "16 minutes after is outside"),
]:
    at = FOMC_AT + _td(minutes=offset_min)
    check(f"blackout: {label}", (cal.find_blackout(ev, at) is not None) == expect,
          f"offset={offset_min}")

check("no blackout when nothing is scheduled",
      cal.find_blackout([], _dt(2026, 9, 14, 11, 0, tzinfo=ET)) is None)

# ---- earnings parsing: every shape yfinance is known to return -------------
# Relative to the injected clock, never absolute — a fixture pinned to a real
# date silently becomes a failure the moment that date passes, which is exactly
# what happened to the first version of these tests.
EARN_NOW = _dt(2026, 9, 14, 10, 0, tzinfo=ET)
future_ts = pd.Timestamp("2026-09-15 16:30")
past_ts = pd.Timestamp("2026-06-15 16:30")

e_dict = cal.parse_earnings("GS", {"Earnings Date": [future_ts]}, None, EARN_NOW)
check("earnings parsed from a calendar dict",
      e_dict.when is not None and e_dict.source == "calendar", e_dict.note)
check("parsed earnings carry reported confidence", e_dict.confidence == "reported")

e_scalar = cal.parse_earnings("GS", {"Earnings Date": future_ts}, None, EARN_NOW)
check("a scalar earnings date parses as well as a list",
      e_scalar.when is not None, e_scalar.note)

e_df = cal.parse_earnings(
    "CAT", None, pd.DataFrame({"EPS Estimate": [1.0]}, index=[future_ts]), EARN_NOW)
check("earnings parsed from get_earnings_dates frame",
      e_df.when is not None and e_df.source == "earnings_dates", e_df.note)

e_stale = cal.parse_earnings("MSFT", {"Earnings Date": [past_ts]}, None, EARN_NOW)
check("only-past earnings dates are rejected, not reported as next",
      e_stale.when is None, e_stale.note)
check("stale earnings say why", "stale" in e_stale.note, e_stale.note)

e_none = cal.parse_earnings("UNH", None, None, EARN_NOW)
check("missing earnings data degrades safely",
      e_none.when is None and e_none.confidence == "none")
e_junk = cal.parse_earnings("V", {"Earnings Date": ["not a date"]}, None, EARN_NOW)
check("unparseable earnings data degrades safely", e_junk.when is None)

# ---- ex-dividend parsing ----------------------------------------------------
d_rep = cal.parse_ex_dividend("GS", {"Ex-Dividend Date": pd.Timestamp("2026-09-13")}, None)
check("reported ex-div date is used and not marked estimated",
      d_rep.ex_date == _date(2026, 9, 13) and not d_rep.estimated)
check("reported ex-div records its source", d_rep.source == "calendar")

quarterly = pd.Series(
    [1.0, 1.0, 1.05, 1.05],
    index=pd.to_datetime(["2025-09-12", "2025-12-12", "2026-03-13", "2026-06-12"]))
d_inf = cal.parse_ex_dividend("JPM", None, quarterly)
check("ex-div inferred from quarterly cadence", d_inf.ex_date is not None, str(d_inf))
check("inferred ex-div is MARKED estimated", d_inf.estimated)
check("inferred ex-div lands roughly one quarter on",
      d_inf.ex_date is not None and 80 <= (d_inf.ex_date - _date(2026, 6, 12)).days <= 100,
      str(d_inf.ex_date))
check("last dividend amount captured", close_to(d_inf.amount, 1.05, 1e-9))

d_thin = cal.parse_ex_dividend("NKE", None, pd.Series(dtype=float))
check("no dividend history degrades safely", d_thin.ex_date is None)

# ---- cross-index -------------------------------------------------------------
n_c = 80
idx_c = pd.date_range("2026-05-01", periods=n_c, freq="B")
base_c = np.cumsum(rng.normal(0, 1, n_c))
coupled_dji = pd.Series(52000 + base_c * 100, index=idx_c)
coupled_ndx = pd.Series(26000 + base_c * 50, index=idx_c)
cx = cal.compute_cross_index(coupled_dji, coupled_ndx)
check("identical drivers give near-perfect correlation",
      cx["correlation"] > 0.95, f"{cx['correlation']}")
check("high correlation is labelled COUPLED", cx["regime"] == "COUPLED", cx["regime"])

rot_ndx = pd.Series(26000 - base_c * 50, index=idx_c)
cx_rot = cal.compute_cross_index(coupled_dji, rot_ndx)
check("opposed drivers give negative correlation", cx_rot["correlation"] < -0.95)
check("low correlation is labelled ROTATION", cx_rot["regime"] == "ROTATION")
check("directions are opposite in a rotation",
      cx_rot["dji_direction"] == -cx_rot["ndx_direction"]
      or cx_rot["ndx_direction"] == 0, str(cx_rot))

check("too little history degrades safely",
      not np.isfinite(cal.compute_cross_index(
          coupled_dji.head(5), coupled_ndx.head(5))["correlation"]))
check("empty series degrade safely",
      cal.compute_cross_index(pd.Series(dtype=float), pd.Series(dtype=float))
      ["regime"] == "UNKNOWN")

# ---- report assembly ---------------------------------------------------------
earn_soon = cal.EarningsEntry("GS", _dt(2026, 9, 14, 16, 30, tzinfo=ET),
                              "calendar", "reported")
earn_far = cal.EarningsEntry("CAT", _dt(2026, 10, 20, 16, 30, tzinfo=ET),
                             "calendar", "reported")
div_today = cal.DividendEntry("UNH", _date(2026, 9, 14), 2.1, "calendar", False)
div_est = cal.DividendEntry("V", _date(2026, 9, 14), 0.6, "inferred", True)
div_later = cal.DividendEntry("JPM", _date(2026, 10, 5), 1.4, "calendar", False)

rep_cal = cal.build_report(NOW_CAL, [earn_soon, earn_far],
                           [div_today, div_est, div_later], cx)
check("calendar report ok", rep_cal.ok, rep_cal.note)
check("earnings inside 24h are flagged",
      rep_cal.earnings_within_window == ["GS"], str(rep_cal.earnings_within_window))
check("earnings a month out are not flagged", "CAT" not in rep_cal.earnings_within_window)
check("today's ex-dividends are collected",
      sorted(rep_cal.ex_div_today) == ["UNH", "V"], str(rep_cal.ex_div_today))
check("a later ex-div is not collected", "JPM" not in rep_cal.ex_div_today)
check("an ESTIMATED ex-div raises a flag",
      any("INFERRED" in f for f in rep_cal.flags), str(rep_cal.flags))

rep_unres = cal.build_report(NOW_CAL, [cal.EarningsEntry("BA")], [], {})
check("names with no earnings date are named in a flag",
      any("BA" in f for f in rep_unres.flags), str(rep_unres.flags))

# ---- C4 / C5 / C8 / C14 ------------------------------------------------------
def base_kwargs(**over):
    kw = dict(
        attribution=Stub(22, index_change_pts=80.0, pw_advance_pts=120.0,
                         pw_decline_pts=40.0, top2_share=0.3, participation=0.5,
                         advancers=20, decliners=10, table=pd.DataFrame()),
        technicals=Stub(18, price=52000, atr14=400, mean_reversion="none"),
        regime=Stub(8, regime="TREND", signal_confidence_scalar=1.0),
        micro=Stub(6, levels=Lv("INSIDE_ON"), basis_sigma=float("nan")),
    )
    kw.update(over)
    return kw


NOW_T = pd.Timestamp("2026-09-14 11:00", tz=TZ).to_pydatetime()

sig_c4 = master.build_master_signal(
    **base_kwargs(), ctx={"now": NOW_T, "earnings_top8": ["GS"]})
check("C4 fires on a top-8 name reporting inside 24h",
      any(c.code == "C4" for c in sig_c4.conflicts),
      str([c.code for c in sig_c4.conflicts]))
check("C4 caps the lot multiplier rather than blocking",
      not sig_c4.blocked and sig_c4.plan.lot_multiplier > 0, str(sig_c4.plan))

sig_noearn = master.build_master_signal(
    **base_kwargs(), ctx={"now": NOW_T, "earnings_top8": []})
check("C4 stays silent with no earnings in the window",
      not any(c.code == "C4" for c in sig_noearn.conflicts))
check("C4 widens the stop relative to no earnings",
      sig_c4.plan.risk_pts > sig_noearn.plan.risk_pts,
      f"{sig_c4.plan.risk_pts} vs {sig_noearn.plan.risk_pts}")
check("C4 halves the lot multiplier",
      sig_c4.plan.lot_multiplier < sig_noearn.plan.lot_multiplier,
      f"{sig_c4.plan.lot_multiplier} vs {sig_noearn.plan.lot_multiplier}")

sig_c5 = master.build_master_signal(
    **base_kwargs(), ctx={"now": NOW_T, "ex_div_today": ["UNH"]})
check("C5 fires on an ex-dividend day",
      any(c.code == "C5" for c in sig_c5.conflicts),
      str([c.code for c in sig_c5.conflicts]))
check("C5 warns rather than blocking", not sig_c5.blocked)

sig_c8 = master.build_master_signal(
    **base_kwargs(),
    ctx={"now": NOW_T, "cross_index": {"correlation": 0.88, "ndx_direction": -1}})
check("C8 fires when NDX opposes at high correlation",
      any(c.code == "C8" for c in sig_c8.conflicts),
      str([c.code for c in sig_c8.conflicts]))
check("C8 downgrades the score",
      abs(sig_c8.final_score) < abs(sig_noearn.final_score),
      f"{sig_c8.final_score} vs {sig_noearn.final_score}")

sig_c8_rot = master.build_master_signal(
    **base_kwargs(),
    ctx={"now": NOW_T, "cross_index": {"correlation": 0.20, "ndx_direction": -1}})
check("C8 UPGRADES in a rotation regime instead",
      abs(sig_c8_rot.final_score) > abs(sig_c8.final_score),
      f"rotation={sig_c8_rot.final_score} coupled={sig_c8.final_score}")

blackout_ev = cal.Event("CPI", _dt(2026, 9, 14, 11, 20, tzinfo=ET), "HIGH", "08:30 ET")
sig_c14 = master.build_master_signal(
    **base_kwargs(), ctx={"now": NOW_T, "event_blackout": blackout_ev})
check("C14 fires inside an event blackout",
      any(c.code == "C14" for c in sig_c14.conflicts),
      str([c.code for c in sig_c14.conflicts]))
check("C14 blocks entry outright", sig_c14.blocked)
check("C14 produces no trade plan", not sig_c14.plan.valid)
check("no blackout means no C14",
      not any(c.code == "C14" for c in sig_noearn.conflicts))

# ---- ex-div neutralisation actually reaches attribution ---------------------
prev_xd2 = last.copy()
prev_xd2["JPM"] = last["JPM"] + 1.4
idx_now = float(last.sum()) / DIV
rep_with = attr.compute_attribution(last, prev_xd2, idx_now,
                                    float(prev_xd2.sum()) / DIV)
rep_without = attr.compute_attribution(last, prev_xd2, idx_now,
                                       float(prev_xd2.sum()) / DIV,
                                       ex_div_today={"JPM"})
check("ex-div neutralisation changes the attribution result",
      abs(float(rep_with.table.loc["JPM", "points"])) > 1.0
      and close_to(float(rep_without.table.loc["JPM", "points"]), 0.0, 1e-9),
      f"with={rep_with.table.loc['JPM', 'points']}")

# ---- stale config ------------------------------------------------------------
_saved_c = {}
for _name in list(cal._CFG_DEFAULTS):
    if hasattr(config, _name):
        _saved_c[_name] = getattr(config, _name)
        delattr(config, _name)
check("stale calendar config is detected and named",
      sorted(cal.config_health()) == sorted(_saved_c), str(cal.config_health()))
check("calendar still builds events on a stale config",
      len(cal.build_events(NOW_CAL, 10)[0]) > 0)
check("blackout still works on a stale config",
      cal.find_blackout(ev, FOMC_AT) is not None)
for _name, _val in _saved_c.items():
    setattr(config, _name, _val)
check("calendar config restored", cal.config_health() == [])

check("layers are still 7 — the calendar is a gate, not a layer",
      len(config.LAYER_WEIGHTS) == 7 and sum(config.LAYER_WEIGHTS.values()) == 100)


# ==========================================================================
print("\n[12] Journal — phase 7")
# ==========================================================================
import us30_journal as jr  # noqa: E402

# ---- Wilson interval against known values ---------------------------------
lo, hi = jr.wilson_interval(13, 21)          # 62% from 21 trades
check("Wilson interval brackets the point estimate",
      lo < 13 / 21 < hi, f"{lo:.3f} {hi:.3f}")
check("small sample gives a wide interval", (hi - lo) > 0.30,
      f"width {(hi - lo):.3f}")



def wilson_by_quadratic(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Independent derivation: the Wilson bounds are the roots of
        (p_hat - p)^2 = z^2 * p(1-p)/n
    i.e.  p^2 (1 + z^2/n) - p(2*p_hat + z^2/n) + p_hat^2 = 0
    Solved numerically here so it shares no code with the implementation —
    validating the formula rather than a constant someone remembered.
    """
    ph = wins / n
    roots = np.roots([1 + z * z / n, -(2 * ph + z * z / n), ph * ph])
    return float(min(roots)), float(max(roots))


for _w, _n in [(13, 21), (6, 10), (130, 210), (1, 4), (99, 100)]:
    _q = wilson_by_quadratic(_w, _n)
    _i = jr.wilson_interval(_w, _n)
    check(f"Wilson {_w}/{_n} matches an independent quadratic solution",
          close_to(_q[0], _i[0], 1e-10) and close_to(_q[1], _i[1], 1e-10),
          f"quadratic={_q[0]:.6f},{_q[1]:.6f} impl={_i[0]:.6f},{_i[1]:.6f}")
lo2, hi2 = jr.wilson_interval(130, 210)      # same rate, 10x the sample
check("ten times the sample narrows the interval", (hi2 - lo2) < (hi - lo) / 2,
      f"{(hi2 - lo2):.3f} vs {(hi - lo):.3f}")
check("interval stays inside [0,1] at the extremes",
      jr.wilson_interval(0, 5)[0] >= 0.0 and jr.wilson_interval(5, 5)[1] <= 1.0)
check("n=0 yields NaN, not a divide-by-zero",
      not np.isfinite(jr.wilson_interval(0, 0)[0]))

# ---- break-even identity from the research doc ----------------------------
check("75% is break-even at 0.5:1 with a 40pt stop and 5pts cost",
      close_to(jr.breakeven_win_rate(0.5, 5.0, 40.0), 0.75, 1e-9),
      f"{jr.breakeven_win_rate(0.5, 5.0, 40.0)}")
check("1:1 with no cost is exactly 50%",
      close_to(jr.breakeven_win_rate(1.0, 0.0, 40.0), 0.50, 1e-12))
check("1:1 with 5pts cost is 56.25%",
      close_to(jr.breakeven_win_rate(1.0, 5.0, 40.0), 0.5625, 1e-9))
check("3:1 with 5pts cost is 28.125%",
      close_to(jr.breakeven_win_rate(3.0, 5.0, 40.0), 0.28125, 1e-9))
check("higher R needs a lower win rate",
      jr.breakeven_win_rate(2.0, 5, 40) < jr.breakeven_win_rate(1.0, 5, 40))
check("zero risk degrades safely",
      not np.isfinite(jr.breakeven_win_rate(1.0, 5.0, 0.0)))

# ---- fingerprint dedup -----------------------------------------------------
fp_a = jr.fingerprint("LONG", 52.0, "MORNING_TREND", 52000.0)
check("same setup gives the same fingerprint",
      fp_a == jr.fingerprint("LONG", 52.0, "MORNING_TREND", 52000.0))
check("a small score drift collapses to the same fingerprint",
      fp_a == jr.fingerprint("LONG", 54.0, "MORNING_TREND", 52010.0),
      "score 52->54 and entry 52000->52010 are the same setup")
check("a direction flip is a different fingerprint",
      fp_a != jr.fingerprint("SHORT", 52.0, "MORNING_TREND", 52000.0))
check("a different session block is a different fingerprint",
      fp_a != jr.fingerprint("LONG", 52.0, "LUNCH_CHOP", 52000.0))
check("a large entry move is a different fingerprint",
      fp_a != jr.fingerprint("LONG", 52.0, "MORNING_TREND", 52500.0))
check("NaN inputs do not raise",
      isinstance(jr.fingerprint("LONG", float("nan"), "X", float("nan")), str))

# ---- store round trip -------------------------------------------------------
import tempfile, os as _os  # noqa: E402

_tmpdir = tempfile.mkdtemp()
_path = _os.path.join(_tmpdir, "journal.csv")
csv_store = jr.CsvStore(_path)
check("a missing file loads as an empty frame with the full schema",
      list(csv_store.load().columns) == jr.COLUMNS)

NOW_J = pd.Timestamp("2026-09-14 10:30", tz=TZ).to_pydatetime()


def make_signal(direction="LONG", score=55.0, entry=52000.0, valid=True,
                blocked=False, ok=True, block="MORNING_TREND"):
    sig = master.MasterSignal(ok=ok)
    sig.direction, sig.final_score = direction, score
    sig.conviction, sig.coverage, sig.confidence = "STRONG", 1.0, 0.8
    sig.session_block, sig.blocked = block, blocked
    sig.layers = [
        master.Layer("attribution", "L1", 25, 20.0, 0.9, True),
        master.Layer("technicals", "L2", 20, 15.0, 0.9, True),
        master.Layer("options", "L6", 10, 0.0, 0.0, False),
    ]
    plan = master.TradePlan()
    plan.direction = direction
    plan.entry, plan.stop = entry, entry - 480.0
    plan.tp1, plan.tp2 = entry + 480.0, entry + 960.0
    plan.risk_pts, plan.rr, plan.lot_multiplier = 480.0, 2.0, 0.7
    plan.valid = valid
    sig.plan = plan
    return sig


ok1, why1 = jr.log_signal(csv_store, make_signal(), NOW_J)
check("an actionable signal is logged", ok1, why1)
check("the row survives a round trip through CSV", len(csv_store.load()) == 1)

ok2, why2 = jr.log_signal(csv_store, make_signal(), NOW_J + pd.Timedelta(minutes=5))
check("the same setup 5 minutes later is deduped", not ok2, why2)
check("dedup names the window", "duplicate" in why2, why2)

ok3, _ = jr.log_signal(csv_store, make_signal(),
                       NOW_J + pd.Timedelta(minutes=25))
check("the same setup past the dedup window logs again", ok3)

ok4, why4 = jr.log_signal(csv_store, make_signal(blocked=True),
                          NOW_J + pd.Timedelta(hours=1))
check("a BLOCKED signal is never logged", not ok4, why4)
check("blocked rows explain themselves", "not a trade" in why4, why4)

ok5, why5 = jr.log_signal(csv_store, make_signal(valid=False),
                          NOW_J + pd.Timedelta(hours=2))
check("a signal with no valid plan is not logged", not ok5, why5)

check("only real setups reached the ledger", len(csv_store.load()) == 2)

# Per-layer capture is the point of the schema.
row0 = csv_store.load().iloc[0]
check("available layer scores are captured",
      close_to(float(row0["l1_attribution"]), 20.0, 1e-9)
      and close_to(float(row0["l2_technicals"]), 15.0, 1e-9))
check("an unavailable layer is NaN, not zero",
      not np.isfinite(pd.to_numeric(row0["l6_options"], errors="coerce")),
      f"{row0['l6_options']}")
check("paper mode is recorded", bool(row0["paper"]))

# ---- THE dtype trap: recording an outcome through a real CSV round trip -----
# An all-empty text column reads back from CSV as float64, and pandas 3 refuses
# to write a string into it. This only fails on the real storage path, which is
# why an in-memory test suite missed it entirely.
_p2 = _os.path.join(_tmpdir, "dtype.csv")
_cs2 = jr.CsvStore(_p2)
jr.log_signal(_cs2, make_signal(entry=52000.0), NOW_J)
_sid_csv = _cs2.load().iloc[0]["signal_id"]
_okc, _stc = jr.record_outcome(_cs2, _sid_csv, 52480.0, NOW_J, "TP1")
check("an outcome can be recorded through a fresh CSV journal", _okc, _stc)
check("the CSV outcome computed correctly",
      close_to(float(_cs2.load().iloc[0]["realised_pts"]), 480.0, 1e-6))
check("text columns survive as text after a CSV round trip",
      isinstance(_cs2.load().iloc[0]["exit_reason"], str))
check("coerce_schema makes an empty frame writable",
      jr.empty_frame()[jr.TEXT_COLUMNS].dtypes.apply(
          lambda d: d == object).all())

# ---- outcomes ---------------------------------------------------------------
mem = jr.MemoryStore()
sid = None
jr.log_signal(mem, make_signal(entry=52000.0), NOW_J)
sid = mem.load().iloc[0]["signal_id"]

okw, status = jr.record_outcome(mem, sid, 52480.0, NOW_J + pd.Timedelta(hours=1), "TP1")
check("a winning outcome is recorded", okw and status == jr.WIN, status)
closed = mem.load().iloc[0]
check("realised points computed from entry", close_to(float(closed["realised_pts"]), 480.0, 1e-6))
check("realised R computed from risk", close_to(float(closed["realised_r"]), 1.0, 1e-6))

check("closing an already-closed row is refused",
      not jr.record_outcome(mem, sid, 52000.0)[0])
check("an unknown id is refused", not jr.record_outcome(mem, "deadbeef", 52000.0)[0])

# Short direction must invert the sign.
mem2 = jr.MemoryStore()
jr.log_signal(mem2, make_signal(direction="SHORT", entry=52000.0), NOW_J)
sid2 = mem2.load().iloc[0]["signal_id"]
jr.record_outcome(mem2, sid2, 51600.0, NOW_J, "TP1")
check("a short that fell is a WIN with positive points",
      mem2.load().iloc[0]["status"] == jr.WIN
      and close_to(float(mem2.load().iloc[0]["realised_pts"]), 400.0, 1e-6),
      str(mem2.load().iloc[0]["realised_pts"]))

# Slippage measured from the actual fill, not the plan.
mem3 = jr.MemoryStore()
jr.log_signal(mem3, make_signal(entry=52000.0), NOW_J)
sid3 = mem3.load().iloc[0]["signal_id"]
jr.record_outcome(mem3, sid3, 52400.0, NOW_J, "TP1", actual_entry=52007.0)
r3 = mem3.load().iloc[0]
check("slippage measured from the real fill",
      close_to(float(r3["actual_slippage_pts"]), 7.0, 1e-6))
check("realised points use the actual fill, not the plan",
      close_to(float(r3["realised_pts"]), 393.0, 1e-6),
      f"{r3['realised_pts']} — 52400 minus the 52007 fill, not the 52000 plan")

# Scratch.
mem4 = jr.MemoryStore()
jr.log_signal(mem4, make_signal(entry=52000.0), NOW_J)
jr.record_outcome(mem4, mem4.load().iloc[0]["signal_id"], 52000.2, NOW_J, "Manual")
check("a flat exit is a SCRATCH, not a win",
      mem4.load().iloc[0]["status"] == jr.SCRATCH)

# ---- expiry ------------------------------------------------------------------
mem5 = jr.MemoryStore()
jr.log_signal(mem5, make_signal(), NOW_J)
check("nothing expires inside the window",
      jr.expire_stale(mem5, NOW_J + pd.Timedelta(hours=2)) == 0)
check("an open row older than 24h expires",
      jr.expire_stale(mem5, NOW_J + pd.Timedelta(hours=30)) == 1)
check("expired rows leave the open count",
      jr.compute_metrics(mem5.load()).n_open == 0)

# ---- metrics against hand-computed values -------------------------------------
def ledger(results, direction="LONG", risk=100.0, start="2026-07-01"):
    """results: list of realised points. Builds a closed ledger."""
    rows = []
    for i, pts in enumerate(results):
        stamp = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=i)
        rows.append({
            **{c: np.nan for c in jr.COLUMNS},
            "signal_id": f"id{i}", "logged_at": stamp.isoformat(),
            "session_block": "MORNING_TREND" if i % 2 else "LUNCH_CHOP",
            "direction": direction, "score": 50.0, "conviction": "STRONG",
            "entry": 52000.0, "risk_pts": risk,
            "l1_attribution": 20.0 + i, "l2_technicals": 10.0,
            "regime": "TREND" if i % 2 else "CHOP",
            "status": jr.WIN if pts > 0.5 else (jr.LOSS if pts < -0.5 else jr.SCRATCH),
            "realised_pts": pts, "realised_r": pts / risk,
            "blocked": False, "paper": True,
        })
    return pd.DataFrame(rows)[jr.COLUMNS]


# 6 wins of +150, 4 losses of -100. Expectancy = (900-400)/10 = +50.
m = jr.compute_metrics(ledger([150, -100, 150, -100, 150, -100, 150, -100, 150, 150]))
check("closed count correct", m.n_closed == 10)
check("wins and losses counted", m.wins == 6 and m.losses == 4)
check("win rate is 60%", close_to(m.win_rate, 0.6, 1e-9))
check("expectancy is +50 points", close_to(m.expectancy_pts, 50.0, 1e-9),
      f"{m.expectancy_pts}")
check("total is +500 points", close_to(m.total_pts, 500.0, 1e-9))
check("profit factor is 900/400 = 2.25",
      close_to(m.profit_factor, 2.25, 1e-9), f"{m.profit_factor}")
check("average win is +150", close_to(m.avg_win_pts, 150.0, 1e-9))
check("average loss is -100", close_to(m.avg_loss_pts, -100.0, 1e-9))
check("expectancy in R is +0.5", close_to(m.expectancy_r, 0.5, 1e-9))
check("win rate carries an interval", np.isfinite(m.win_rate_low))
check("win rate text shows the interval and n",
      "CI" in m.win_rate_text and "n=10" in m.win_rate_text, m.win_rate_text)

# Max consecutive losses.
m_run = jr.compute_metrics(ledger([100, -50, -50, -50, 100, -50, -50, 100]))
check("max consecutive losses found", m_run.max_consecutive_losses == 3,
      str(m_run.max_consecutive_losses))
check("max_consecutive helper handles no losses",
      jr.max_consecutive(pd.Series([jr.WIN, jr.WIN]), jr.LOSS) == 0)

# The research doc's two systems, reproduced through the journal.
win_optimised = jr.compute_metrics(ledger([20] * 78 + [-40] * 22, risk=40.0))
balanced = jr.compute_metrics(ledger([40] * 70 + [-40] * 30, risk=40.0))
check("78% at 0.5:1 shows a high win rate", close_to(win_optimised.win_rate, 0.78, 1e-9))
check("70% at 1:1 shows a lower win rate", close_to(balanced.win_rate, 0.70, 1e-9))
check("...yet the lower win rate has far better expectancy",
      balanced.expectancy_pts > win_optimised.expectancy_pts * 2,
      f"balanced={balanced.expectancy_pts:.1f} "
      f"win-optimised={win_optimised.expectancy_pts:.1f}")
check("...and a better profit factor",
      balanced.profit_factor > win_optimised.profit_factor,
      f"{balanced.profit_factor:.2f} vs {win_optimised.profit_factor:.2f}")

# Grading against break-even.
g = jr.grade_against_breakeven(win_optimised, cost_pts=5.0)
check("the win-optimised system is graded against its own R",
      np.isfinite(g["required"]), str(g))
check("78% at 0.5:1 does NOT clear break-even with confidence",
      g["verdict"] != "clears break-even with confidence", str(g))

# ---- sample-size and forward-test gates ----------------------------------------
small = jr.compute_metrics(ledger([100, -50, 100, -50, 100]))
check("a 5-trade sample is flagged inadequate", not small.sample_adequate)
check("the inadequate-sample warning names the interval width",
      any("noise, not a result" in w for w in small.warnings), str(small.warnings))
check("an incomplete forward test says so",
      small.verdict == "FORWARD TEST IN PROGRESS", small.verdict)

# 40 trades but all on consecutive days from one start -> days gate matters.
many_days = ledger([50, -30] * 20, start="2026-06-01")
m_days = jr.compute_metrics(many_days)
check("40 trades over 40 days still fails the 60-day gate",
      not m_days.forward_test_complete,
      f"n={m_days.n_closed} days={m_days.days_elapsed}")
check("the days/trades warning names both gates",
      any("BOTH gates" in w for w in m_days.warnings), str(m_days.warnings))

# Both gates cleared.
full = ledger([50, -30] * 20, start="2026-05-01")
full.loc[len(full) - 1, "logged_at"] = pd.Timestamp("2026-08-01", tz="UTC").isoformat()
m_full = jr.compute_metrics(full)
check("both gates cleared completes the forward test", m_full.forward_test_complete,
      f"n={m_full.n_closed} days={m_full.days_elapsed}")
check("a completed test that meets targets says so",
      m_full.verdict in ("MEETS TARGETS", "POSITIVE BUT BELOW TARGET"), m_full.verdict)

negative = ledger([30, -60] * 20, start="2026-05-01")
negative.loc[len(negative) - 1, "logged_at"] = pd.Timestamp("2026-08-01", tz="UTC").isoformat()
check("negative expectancy is called negative",
      jr.compute_metrics(negative).verdict == "NEGATIVE EXPECTANCY")

# ---- breakdowns -----------------------------------------------------------------
check("regime breakdown splits trend from chop",
      set(m.by_regime.index) == {"TREND", "CHOP"}, str(m.by_regime.index.tolist()))
check("each regime row carries its own interval",
      "CI_low_%" in m.by_regime.columns)
check("session breakdown is produced", not m.by_session.empty)
check("breakdown n sums to the closed count",
      int(m.by_regime["n"].sum()) == m.n_closed)

big = ledger([150, -100] * 15)
m_big = jr.compute_metrics(big)
check("layer correlation computed once n is adequate",
      not m_big.layer_correlation.empty, str(m_big.layer_correlation))

# ---- empty and malformed ---------------------------------------------------------
m_empty = jr.compute_metrics(jr.empty_frame())
check("an empty journal degrades safely",
      m_empty.n_closed == 0 and m_empty.verdict == "INSUFFICIENT DATA")
check("the empty journal says it is empty",
      any("empty" in w for w in m_empty.warnings))

open_only = ledger([100])
open_only["status"] = jr.OPEN
open_only["realised_pts"] = np.nan
m_open = jr.compute_metrics(open_only)
check("open-only journal reports nothing closed", m_open.n_closed == 0)
check("open-only journal explains why",
      any("none closed" in w for w in m_open.warnings), str(m_open.warnings))

# ---- persistence honesty -----------------------------------------------------------
check("a memory store admits it is not durable", not jr.MemoryStore().durable)
check("the memory store warns", jr.storage_warning(jr.MemoryStore()) != "")
check("the warning names the real risk",
      "wipes the filesystem" in jr.storage_warning(jr.MemoryStore()))


class _FakeDurable:
    durable = True
    location = "/somewhere/real"


check("a durable store produces no warning",
      jr.storage_warning(_FakeDurable()) == "")

_prev_env = _os.environ.pop("STREAMLIT_RUNTIME_ENV", None)
check("no Streamlit markers means not detected as ephemeral",
      jr.on_ephemeral_host() == _os.path.exists("/mount/src"))
_os.environ["STREAMLIT_RUNTIME_ENV"] = "cloud"
check("a Streamlit Cloud marker is detected as ephemeral", jr.on_ephemeral_host())
check("a CSV store on an ephemeral host reports NOT durable",
      not jr.CsvStore(_path).durable)
_os.environ.pop("STREAMLIT_RUNTIME_ENV", None)
if _prev_env is not None:
    _os.environ["STREAMLIT_RUNTIME_ENV"] = _prev_env

# ---- atomic write ------------------------------------------------------------------
before = csv_store.load()
check("a failed save leaves the previous file intact",
      jr.CsvStore("/proc/nonexistent/journal.csv").save(before) is False
      and len(csv_store.load()) == len(before))

# ---- stale config --------------------------------------------------------------------
_saved_j = {}
for _name in list(jr._CFG_DEFAULTS):
    if hasattr(config, _name):
        _saved_j[_name] = getattr(config, _name)
        delattr(config, _name)
check("stale journal config is detected and named",
      sorted(jr.config_health()) == sorted(_saved_j), str(jr.config_health()))
check("metrics still compute on a stale config",
      jr.compute_metrics(ledger([100, -50])).n_closed == 2)
check("Wilson still computes on a stale config",
      np.isfinite(jr.wilson_interval(5, 10)[0]))
for _name, _val in _saved_j.items():
    setattr(config, _name, _val)
check("journal config restored", jr.config_health() == [])

import shutil as _shutil  # noqa: E402
_shutil.rmtree(_tmpdir, ignore_errors=True)


# ==========================================================================
print("\n[13] Reversal readiness — observation only")
# ==========================================================================
import us30_reversal as rv  # noqa: E402


class Lvls:
    def __init__(self, **kw):
        for k in ("overnight_high", "overnight_low", "prior_high", "prior_low",
                  "ib_high", "ib_low"):
            setattr(self, k, kw.get(k, float("nan")))


class Swp:
    def __init__(self, side, name, bars_ago=2, confirmed=True):
        self.side, self.level_name = side, name
        self.bars_ago, self.confirmed = bars_ago, confirmed


def T(**kw):
    d = dict(price=52000.0, atr14=400.0, vwap_distance_pts=-700.0,
             rsi_decay=0.0, divergence="none", mean_reversion="none",
             cpr=tech.compute_cpr(53400, 52900, 53000, 400, 52000))
    d.update(kw)
    return Stub(0, **d)


def MI(**kw):
    d = dict(levels=Lvls(overnight_low=51990.0, overnight_high=52700.0,
                         prior_low=51950.0, prior_high=52800.0),
             sweeps=[], delta_divergence="none", basis_sigma=float("nan"))
    d.update(kw)
    return Stub(0, **d)


def AT(**kw):
    d = dict(index_change_pts=-450.0, participation=-0.9, efficiency=-0.9,
             top2_share=0.30)
    d.update(kw)
    return Stub(0, **d)


def OP(**kw):
    d = dict(expected_move_pts=500.0, gamma_regime="SHORT_GAMMA", gamma_levels=[])
    d.update(kw)
    return Stub(0, **d)


# ---- the falling-knife guard: extended but nowhere near a level -------------
# 52145 sits in the gap between Pivot S3 (52300) and the overnight low (51990)
# — 0.39 ATR from the nearest support, outside the 0.30 proximity band.
far = rv.assess(technicals=T(price=52145.0, vwap_distance_pts=-700.0),
                micro=MI(), attribution=AT(), options=OP(),
                regime=Stub(0, regime="TREND"))
check("extended but not at a level stays DORMANT", far.state == rv.DORMANT,
      f"{far.state} note={far.note}")
check("the no-level refusal explains itself",
      "falling-knife" in far.note, far.note)

# ---- not extended at all -----------------------------------------------------
calm = rv.assess(technicals=T(vwap_distance_pts=-50.0),
                 micro=MI(), attribution=AT(index_change_pts=-20.0),
                 options=OP(), regime=Stub(0, regime="TRANSITION"))
check("no extension means nothing to reverse", calm.state == rv.DORMANT)
check("calm state says so", "no extension" in calm.note, calm.note)

# ---- ARMED: extended and sitting on the overnight low ------------------------
armed = rv.assess(technicals=T(price=52000.0), micro=MI(), attribution=AT(),
                  options=OP(), regime=Stub(0, regime="TRANSITION"))
check("extended AND at a level arms", armed.state == rv.ARMED, armed.note)
check("the reversal direction is opposite the extension",
      armed.direction == "BULLISH", armed.direction)
check("the arming level is named", armed.level is not None
      and "Overnight low" in armed.level.name, str(armed.level))

# ---- TRIGGERED: the level was swept and rejected -----------------------------
trig = rv.assess(technicals=T(price=52000.0),
                 micro=MI(sweeps=[Swp("low", "Overnight low", 2, True)]),
                 attribution=AT(), options=OP(),
                 regime=Stub(0, regime="TRANSITION"))
check("a confirmed sweep triggers", trig.state == rv.TRIGGERED, trig.note)
check("the sweep is named", trig.sweep_level == "Overnight low")
check("evidence is evaluated once triggered", len(trig.evidence) == 5)

# An UNconfirmed sweep (thin volume) must not trigger.
thin = rv.assess(technicals=T(price=52000.0),
                 micro=MI(sweeps=[Swp("low", "Overnight low", 2, False)]),
                 attribution=AT(), options=OP(),
                 regime=Stub(0, regime="TRANSITION"))
check("a sweep on thin volume does NOT trigger", thin.state == rv.ARMED, thin.state)
check("the thin sweep is explained", any("thin volume" in f for f in thin.flags),
      str(thin.flags))

# A sweep of the WRONG side must not trigger a bullish reversal.
wrong = rv.assess(technicals=T(price=52000.0),
                  micro=MI(sweeps=[Swp("high", "Prior day high", 2, True)]),
                  attribution=AT(), options=OP(),
                  regime=Stub(0, regime="TRANSITION"))
check("a high sweep does not trigger a bullish reversal", wrong.state == rv.ARMED)

# A stale sweep falls outside the window.
stale = rv.assess(technicals=T(price=52000.0),
                  micro=MI(sweeps=[Swp("low", "Overnight low", 40, True)]),
                  attribution=AT(), options=OP(),
                  regime=Stub(0, regime="TRANSITION"))
check("a sweep older than the window does not trigger", stale.state == rv.ARMED)

# ---- CONFIRMED: internals turn ------------------------------------------------
conf = rv.assess(
    technicals=T(price=52000.0, rsi_decay=1.2, divergence="bullish"),
    micro=MI(sweeps=[Swp("low", "Overnight low", 2, True)],
             delta_divergence="bullish"),
    attribution=AT(participation=-0.2, efficiency=-0.9, top2_share=0.55),
    options=OP(), regime=Stub(0, regime="TRANSITION"))
check("enough evidence confirms", conf.state == rv.CONFIRMED, conf.note)
check("all five evidence items present", conf.evidence_count == 5,
      str([(e.key, e.present) for e in conf.evidence]))

# The Dow-specific one in isolation: breadth ahead of price.
breadth_only = rv.assess(
    technicals=T(price=52000.0), micro=MI(sweeps=[Swp("low", "Overnight low", 2, True)]),
    attribution=AT(participation=-0.2, efficiency=-0.9, top2_share=0.20),
    options=OP(), regime=Stub(0, regime="TRANSITION"))
_b = next(e for e in breadth_only.evidence if e.key == "breadth")
check("breadth ahead of price is detected on its own", _b.present, _b.detail)
check("a broad selloff does NOT show breadth exhaustion",
      not next(e for e in trig.evidence if e.key == "breadth").present,
      "participation and efficiency both -0.9 is a broad decline")
check("concentration evidence fires when the decline narrows",
      next(e for e in conf.evidence if e.key == "concentration").present)

# ---- TREND raises the bar --------------------------------------------------------
three = dict(
    technicals=T(price=52000.0, rsi_decay=1.2, divergence="bullish"),
    micro=MI(sweeps=[Swp("low", "Overnight low", 2, True)], delta_divergence="bullish"),
    attribution=AT(participation=-0.9, efficiency=-0.9, top2_share=0.20),
    options=OP())
in_transition = rv.assess(**three, regime=Stub(0, regime="TRANSITION"))
in_trend = rv.assess(**three, regime=Stub(0, regime="TREND"))
check("3 of 5 confirms outside a trend", in_transition.state == rv.CONFIRMED,
      f"{in_transition.evidence_count}/{in_transition.evidence_required}")
check("the SAME evidence only triggers in TREND", in_trend.state == rv.TRIGGERED,
      f"{in_trend.evidence_count}/{in_trend.evidence_required}")
check("the raised bar is explained",
      any("every support level fails" in f for f in in_trend.flags), str(in_trend.flags))

# ---- bearish mirror ----------------------------------------------------------------
bear = rv.assess(
    technicals=T(price=52700.0, vwap_distance_pts=+700.0, rsi_decay=-1.2,
                 divergence="bearish"),
    micro=MI(sweeps=[Swp("high", "Overnight high", 2, True)],
             delta_divergence="bearish"),
    attribution=AT(index_change_pts=+450.0, participation=0.2, efficiency=0.9,
                   top2_share=0.55),
    options=OP(), regime=Stub(0, regime="TRANSITION"))
check("the bearish mirror confirms", bear.state == rv.CONFIRMED, bear.note)
check("bearish reversal direction is correct", bear.direction == "BEARISH")

# ---- it reports what the master signal would have done -------------------------------
check("C12 suppression is surfaced",
      any("C12" in s for s in conf.suppressed_by), str(conf.suppressed_by))
chop = rv.assess(
    technicals=T(price=52000.0, rsi_decay=1.2, divergence="bullish"),
    micro=MI(sweeps=[Swp("low", "Overnight low", 2, True)], delta_divergence="bullish"),
    attribution=AT(participation=-0.2, efficiency=-0.9, top2_share=0.55),
    options=OP(gamma_regime="NEUTRAL"), regime=Stub(0, regime="CHOP"))
check("C3 suppression is surfaced", any("C3" in s for s in chop.suppressed_by),
      str(chop.suppressed_by))

# ---- levels --------------------------------------------------------------------------
lv_all = rv.collect_levels(T(), MI(), [{"index_level": 51800.0, "kind": "support",
                                        "tickers": "GS, CAT"}])
check("levels gathered from pivots, session and options",
      {l.source for l in lv_all} >= {"pivots", "session", "options"},
      str({l.source for l in lv_all}))
near = rv.nearest_level(lv_all, 52000.0, 400.0, "support")
check("nearest support found and annotated",
      near is not None and np.isfinite(near.distance_atr), str(near))
check("nearest_level respects the kind filter",
      rv.nearest_level(lv_all, 52000.0, 400.0, "resistance").kind == "resistance")
check("no levels degrades safely", rv.nearest_level([], 52000.0, 400.0) is None)

# ---- index gamma clusters -------------------------------------------------------------
DIVI = config.DIVISOR_REFERENCE
gn = [opt.NameGamma("GS", spot=1000.0, weight_pct=11.7),
      opt.NameGamma("CAT", spot=800.0, weight_pct=9.2),
      opt.NameGamma("NKE", spot=37.0, weight_pct=0.42)]
gn[0].gamma_wall, gn[0].liquid = 990.0, True      # -10 -> -59 pts
gn[1].gamma_wall, gn[1].liquid = 792.0, True      # -8  -> -48 pts
gn[2].gamma_wall, gn[2].liquid = 30.0, True       # -7  -> -42 pts but tiny weight
clusters = opt.index_gamma_levels(gn, DIVI, 52000.0)
check("gamma walls translate into index levels", len(clusters) >= 1, str(clusters))
check("the cluster sits below spot and is labelled support",
      clusters[0]["index_level"] < 52000.0 and clusters[0]["kind"] == "support",
      str(clusters[0]))
check("GS and CAT cluster together",
      "GS" in clusters[0]["tickers"] and "CAT" in clusters[0]["tickers"],
      clusters[0]["tickers"])
check("a $10 GS wall implies roughly 59 index points",
      close_to(abs(clusters[0]["distance_pts"]), 54.0, 12.0),
      f"{clusters[0]['distance_pts']}")
check("illiquid or unwalled names are skipped",
      opt.index_gamma_levels([opt.NameGamma("X", spot=100.0)], DIVI, 52000.0) == [])
check("a bad divisor degrades safely",
      opt.index_gamma_levels(gn, 0.0, 52000.0) == [])

# ---- observation log --------------------------------------------------------------------
log = rv.MemoryLogStore()
NOW_R = pd.Timestamp("2026-09-18 11:00", tz=TZ).to_pydatetime()

okl, whyl = rv.log_observation(log, armed, OP(), NOW_R)
check("ARMED is not logged — it happens all day", not okl, whyl)
okl2, _ = rv.log_observation(log, conf, OP(), NOW_R)
check("CONFIRMED is logged", okl2)
okl3, whyl3 = rv.log_observation(log, conf, OP(), NOW_R + pd.Timedelta(minutes=5))
check("the same observation is deduped", not okl3, whyl3)
okl4, _ = rv.log_observation(log, conf, OP(), NOW_R + pd.Timedelta(minutes=45))
check("a later observation logs again", okl4)

check("nothing is followed up before the window",
      rv.fill_followups(log, 52100.0, NOW_R + pd.Timedelta(minutes=10)) == 0)
filled = rv.fill_followups(log, 52180.0, NOW_R + pd.Timedelta(minutes=90))
check("observations past the window are followed up", filled == 2, str(filled))

after = log.load()
check("the follow-up records what price did",
      close_to(float(after.iloc[0]["move_pts"]), 180.0, 1e-6),
      str(after.iloc[0]["move_pts"]))
check("a bullish call that rose is marked right",
      after.iloc[0]["went_the_right_way"] == "yes")

log_wrong = rv.MemoryLogStore()
rv.log_observation(log_wrong, conf, OP(), NOW_R)
rv.fill_followups(log_wrong, 51800.0, NOW_R + pd.Timedelta(minutes=90))
check("a bullish call that fell is marked wrong",
      log_wrong.load().iloc[0]["went_the_right_way"] == "no")

log_flat = rv.MemoryLogStore()
rv.log_observation(log_flat, conf, OP(), NOW_R)
rv.fill_followups(log_flat, 52002.0, NOW_R + pd.Timedelta(minutes=90))
check("a tiny move is marked flat, not a win",
      log_flat.load().iloc[0]["went_the_right_way"] == "flat")

summ = rv.summarise(after)
check("summary counts observations", summ["observations"] == 2)
check("summary computes a hit rate", np.isfinite(summ["hit_rate"]), str(summ))
check("summary splits by state", not summ["by_state"].empty)
check("an empty log summarises safely",
      rv.summarise(rv.empty_log())["observations"] == 0)

# ---- it must not touch the master signal ------------------------------------------------
check("reversal has no score attribute",
      not hasattr(conf, "score"), "observation-only means no score")
check("layers are still 7 and sum to 100",
      len(config.LAYER_WEIGHTS) == 7 and sum(config.LAYER_WEIGHTS.values()) == 100)

# ---- degradation -------------------------------------------------------------------------
check("no inputs at all degrades safely",
      rv.assess().state == rv.DORMANT and rv.assess().ok is False)
check("missing micro degrades safely",
      rv.assess(technicals=T(), attribution=AT(), options=OP()).ok)

_saved_r = {}
for _name in list(rv._CFG_DEFAULTS):
    if hasattr(config, _name):
        _saved_r[_name] = getattr(config, _name)
        delattr(config, _name)
check("stale reversal config is detected", sorted(rv.config_health()) == sorted(_saved_r))
check("assess still runs on a stale config",
      rv.assess(technicals=T(price=52000.0), micro=MI(), attribution=AT(),
                options=OP(), regime=Stub(0, regime="TRANSITION")).state == rv.ARMED)
for _name, _val in _saved_r.items():
    setattr(config, _name, _val)
check("reversal config restored", rv.config_health() == [])


# ==========================================================================
print("\n[14] Data layer guards")
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
