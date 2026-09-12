"""
app.py — US30 Monitor

Resilience patterns carried over from the NAS100 build, because they were
expensive to learn there:
  * regime detection runs FIRST, inside its own try/except
  * every panel is individually wrapped, so one bad panel cannot white-screen
    the whole app
  * every engine returns a report object with ok=False rather than raising

Phases 4 (microstructure) and 5 (options) are not built yet. Their layer
budgets are redistributed, not zeroed — see us30_master_signal.
"""

from __future__ import annotations

import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import config
import data_layer as dl
import dow_attribution as attr
import us30_macro as macro_mod
import us30_micro as micro_mod
import us30_regime as regime_mod
import us30_sectors as sectors_mod
import us30_technicals as tech_mod
import us30_master_signal as master

st.set_page_config(page_title="US30 Monitor", page_icon="📉", layout="wide")


# --------------------------------------------------------------------------
def panel(title: str):
    """Decorator: any panel that throws shows an error box and nothing more."""
    def _wrap(fn):
        def _inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{title} failed: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                return None
        return _inner
    return _wrap


def signed(value: float, digits: int = 0) -> str:
    return "—" if not np.isfinite(value) else f"{value:+,.{digits}f}"


# ==========================================================================
# Header
# ==========================================================================
now_et = datetime.now(ZoneInfo(config.MARKET_TZ))
block, scalar = master.current_session_block(now_et)

st.title("US30 Monitor")
st.caption(
    f"Dow Jones 30 spot index signals · price-weighted attribution engine · "
    f"{now_et:%Y-%m-%d %H:%M} ET · session block **{block}** (scalar {scalar:.1f})"
)

with st.sidebar:
    st.header("Controls")
    spread_pts = st.number_input(
        "Live spread (index points)", 0.0, 50.0, config.DEFAULT_SPREAD_PTS, 0.5,
        help="Read this from MT5 symbol_info_tick. yfinance cannot supply it reliably.",
    )
    slippage_pts = st.number_input(
        "Expected slippage (index points)", 0.0, 50.0, config.DEFAULT_SLIPPAGE_PTS, 0.5,
        help="Measure from your own MT5 fills. Do not model it optimistically.",
    )
    st.divider()
    st.caption(
        "**Not yet built:** L6 options (phase 5). Its ±10 budget is "
        "redistributed across live layers rather than scored zero."
    )
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()


# ==========================================================================
# Engines — regime first, as on the NAS100 build
# ==========================================================================
with st.spinner("Loading engines..."):
    try:
        regime = regime_mod.get_regime()
    except Exception as exc:  # noqa: BLE001
        regime = regime_mod.RegimeReport(note=f"regime crashed: {exc}")

    try:
        attribution = attr.get_attribution()
    except Exception as exc:  # noqa: BLE001
        attribution = attr.AttributionReport(note=f"attribution crashed: {exc}")

    try:
        technicals = tech_mod.get_technicals()
    except Exception as exc:  # noqa: BLE001
        technicals = tech_mod.TechnicalReport(note=f"technicals crashed: {exc}")

    try:
        macro = macro_mod.get_macro()
    except Exception as exc:  # noqa: BLE001
        macro = macro_mod.MacroReport(note=f"macro crashed: {exc}")

    try:
        sectors = sectors_mod.get_sectors()
    except Exception as exc:  # noqa: BLE001
        sectors = sectors_mod.SectorReport(note=f"sectors crashed: {exc}")

    try:
        micro = micro_mod.get_micro(spread_pts, slippage_pts)
    except Exception as exc:  # noqa: BLE001
        micro = micro_mod.MicroReport(note=f"microstructure crashed: {exc}")

    try:
        signal = master.build_master_signal(
            attribution=attribution, technicals=technicals, macro=macro,
            regime=regime, sectors=sectors, micro=micro,
            ctx={"spread_pts": spread_pts, "slippage_pts": slippage_pts},
        )
    except Exception as exc:  # noqa: BLE001
        signal = master.MasterSignal()
        st.error(f"Master signal failed: {exc}")


# ==========================================================================
# Master signal
# ==========================================================================
@panel("Master signal")
def render_signal():
    st.subheader("Master signal")

    colour = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}[signal.direction]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score", f"{signal.final_score:+.0f}", help="±100 scale, post-conflict")
    c2.metric("Direction", f"{colour} {signal.direction}")
    c3.metric("Conviction", signal.conviction)
    c4.metric("Coverage", f"{signal.coverage * 100:.0f}%",
              help="Share of the ±100 budget from layers that actually reported")
    c5.metric("Confidence", f"{signal.confidence * 100:.0f}%")

    if signal.blocked:
        st.error("**Entry blocked** — " + " · ".join(signal.block_reasons))

    plan = signal.plan
    if plan.valid:
        st.success(f"**{plan.direction}** setup")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Entry", f"{plan.entry:,.0f}",
                  help=f"Zone {plan.entry_low:,.0f} – {plan.entry_high:,.0f}")
        p2.metric("Stop", f"{plan.stop:,.0f}", f"{plan.risk_pts:,.0f} pts risk")
        p3.metric("TP1", f"{plan.tp1:,.0f}")
        p4.metric("TP2", f"{plan.tp2:,.0f}", f"{plan.rr:.1f}R")
        p5.metric("Lot multiplier", f"{plan.lot_multiplier:.2f}")
        st.caption(
            "Risk is defined in ATR multiples and expressed in INDEX points. "
            "Convert to broker points with `mt5.symbol_info(symbol).point` at "
            "runtime — never copy a point count from the USTEC bot."
        )
    elif plan.reason:
        st.info(f"No actionable setup: {plan.reason}")

    if not signal.layer_frame.empty:
        st.dataframe(signal.layer_frame, width="stretch", hide_index=True)

    if signal.conflicts:
        st.markdown("**Conflict resolver**")
        for c in signal.conflicts:
            icon = {"block": "⛔", "downgrade": "⬇️", "upgrade": "⬆️", "warn": "⚠️"}[c.severity]
            st.markdown(f"{icon} **{c.code}** — {c.description}  \n&nbsp;&nbsp;→ _{c.action}_")

    for n in signal.notes:
        st.caption(n)


# ==========================================================================
# Attribution
# ==========================================================================
@panel("Attribution")
def render_attribution():
    st.subheader("Point attribution — who is moving the Dow")
    if not attribution.ok:
        st.warning(attribution.note or "attribution unavailable")
        return

    d = attribution.divisor
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Index", f"{attribution.index_level:,.0f}",
              signed(attribution.index_change_pts))
    c2.metric("Divisor", f"{d.value:.10f}",
              "derived" if d.derived else "reference")
    c3.metric("$1 move", f"{d.points_per_dollar:.3f} pts")
    c4.metric("A / D", f"{attribution.advancers} / {attribution.decliners}")
    c5.metric("Top-2 share", f"{attribution.top2_share * 100:.0f}%",
              help="Share of the gross move from the two biggest contributors")

    err = attribution.reconciliation_error
    if np.isfinite(err):
        if abs(err) <= 5:
            st.success(
                f"Reconciliation ✓ — contributions sum to {attribution.reconciled_pts:+,.1f}pts "
                f"against an index change of {attribution.index_change_pts:+,.1f}pts "
                f"(error {err:+.2f})"
            )
        else:
            st.warning(f"Reconciliation error {err:+.1f}pts — stale or missing prices")

    if d.alert:
        st.error(f"Divisor drift: {d.message}")

    for f in attribution.flags:
        st.warning(f)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Contribution ledger** (index points)")
        table = attribution.top_contributors[
            ["price", "pct", "points", "weight_pct", "pts_per_1pct", "sector"]
        ].rename(columns={
            "price": "Price", "pct": "%", "points": "DJIA pts",
            "weight_pct": "Weight %", "pts_per_1pct": "Pts/1%", "sector": "Sector",
        })
        st.dataframe(table, width="stretch", height=420)
    with right:
        st.markdown("**Points by sector**")
        st.bar_chart(attribution.sector_points)
        st.metric("Efficiency", f"{attribution.efficiency:+.2f}",
                  help="Net over gross move. ±1 = perfectly one-sided tape.")
        st.metric("Participation", f"{attribution.participation:+.2f}",
                  help="(advancers − decliners) / 30")

    top8 = attr.rank_by_weight(attribution, 8)
    st.caption(
        f"Top 8 by live price weight: **{', '.join(top8)}** — "
        f"re-ranked every refresh, never hard-coded, because a split reorders this instantly."
    )


# ==========================================================================
# Technicals
# ==========================================================================
@panel("Technicals")
def render_technicals():
    st.subheader("Technicals")
    if not technicals.ok:
        st.warning(technicals.note or "technicals unavailable")
        return

    cpr = technicals.cpr
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"{technicals.price:,.0f}")
    c2.metric("ATR(14)", f"{technicals.atr14:,.0f} pts")
    c3.metric("CPR", cpr.classification,
              f"{cpr.position}{' · virgin' if cpr.virgin else ''}")
    c4.metric("RSI", f"{technicals.rsi_value:.1f}" if np.isfinite(technicals.rsi_value) else "—",
              f"decay {technicals.rsi_decay:+.1f}/bar")
    c5.metric("EMA stack", technicals.ema_stack)

    if technicals.vwap_source == "unavailable":
        st.warning(
            "VWAP unavailable — no volume-bearing source resolved. "
            "This is expected if only ^DJI came back, since its volume is always 0."
        )
    else:
        v1, v2, v3 = st.columns(3)
        v1.metric("VWAP", f"{technicals.vwap_value:,.0f}", f"via {technicals.vwap_source}")
        v2.metric("Distance", f"{signed(technicals.vwap_distance_pts)} pts")
        v3.metric("VWAP slope", f"{technicals.vwap_slope:+.2f}")

    st.markdown("**CPR and pivot levels**")
    levels = pd.DataFrame({
        "level": ["R3", "R2", "R1", "TC", "Pivot", "BC", "S1", "S2", "S3"],
        "value": [cpr.r3, cpr.r2, cpr.r1, cpr.tc, cpr.pivot, cpr.bc, cpr.s1, cpr.s2, cpr.s3],
    })
    levels["distance"] = (levels["value"] - technicals.price).round(0)
    st.dataframe(levels.round(1), width="stretch", hide_index=True)
    st.caption(
        f"CPR width {cpr.width:,.0f} pts = {cpr.width_atr:.2f} × ATR(20). "
        f"Classified against ATR, not absolute points — a fixed threshold ported "
        f"from NAS100 would misclassify every session at DJIA's scale."
    )

    if technicals.divergence != "none":
        st.info(f"Divergence detected: **{technicals.divergence}**")
    if technicals.mean_reversion != "none":
        st.info(f"Mean-reversion setup: **{technicals.mean_reversion}**")
    for f in technicals.flags:
        st.caption(f"⚠️ {f}")


# ==========================================================================
# Macro / regime / sectors
# ==========================================================================
@panel("Macro")
def render_macro():
    st.subheader("Macro & rates")
    if not macro.ok:
        st.warning(macro.note or "macro unavailable")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("10y", f"{macro.us10y:.2f}%" if np.isfinite(macro.us10y) else "—",
              f"{macro.us10y_change_bp:+.1f} bp")
    c2.metric("5s10s", f"{macro.curve_5s10s:+.2f}%" if np.isfinite(macro.curve_5s10s) else "—",
              f"{macro.curve_change_bp:+.1f} bp")
    c3.metric("DXY", f"{macro.dxy:.2f}" if np.isfinite(macro.dxy) else "—",
              f"{macro.dxy_change_pct:+.2f}%")
    c4.metric("WTI", f"${macro.wti:.1f}" if np.isfinite(macro.wti) else "—",
              f"{macro.wti_change_pct:+.1f}%")
    c5.metric("VXD/VIX", f"{macro.vxd_vix_ratio:.2f}" if np.isfinite(macro.vxd_vix_ratio) else "—",
              macro.regime_label)
    for d in macro.drivers:
        st.markdown(f"- {d}")
    for f in macro.flags:
        st.caption(f"⚠️ {f}")
    st.caption(
        "Rates are treated as two-sided here. Financials are 27.8% of the Dow, "
        "so rising yields can help the largest bloc while hurting the 18.5% tech "
        "bloc — C2 resolves it using XLF relative strength."
    )


@panel("Regime")
def render_regime():
    st.subheader("Regime")
    if not regime.ok:
        st.warning(regime.note or "regime unavailable")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Regime", regime.regime, regime.direction)
    c2.metric("ADX", f"{regime.adx_value:.1f}" if np.isfinite(regime.adx_value) else "—")
    c3.metric("Efficiency", f"{regime.efficiency:.2f}" if np.isfinite(regime.efficiency) else "—")
    c4.metric("Dispersion", regime.dispersion_regime,
              f"{regime.dispersion:.2f}%" if np.isfinite(regime.dispersion) else "—")
    c5.metric("Confidence scalar", f"{regime.signal_confidence_scalar:.2f}×")
    for f in regime.flags:
        st.caption(f"• {f}")
    st.caption(
        "Dispersion scales the signal's confidence, never its direction — "
        "treating a confidence input as directional is how confluence systems "
        "double-count."
    )


@panel("Microstructure")
def render_micro():
    st.subheader("Futures & microstructure")
    if not micro.ok:
        st.warning(micro.note or "microstructure unavailable")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Source", micro.source)
    c2.metric("RVOL", f"{micro.rvol:.2f}" if np.isfinite(micro.rvol) else "—",
              micro.rvol_state,
              help="Against the same 5-minute slot on prior sessions, not a flat daily average")
    c3.metric("Basis", f"{micro.basis:+.0f}" if micro.basis_available else "n/a",
              f"{micro.basis_sigma:+.1f}σ" if micro.basis_available else "outside RTH")
    c4.metric("Round-trip cost", f"{micro.round_trip_cost:.1f} pts",
              f"min target {micro.min_viable_target:.0f}")
    c5.metric("Overnight state", micro.levels.state.replace("_", " ").title())

    lv = micro.levels
    st.markdown("**Session levels**")
    levels = pd.DataFrame({
        "level": ["Overnight high", "Initial balance high", "Prior day high",
                  "Prior day close", "Prior day low", "Initial balance low",
                  "Overnight low"],
        "value": [lv.overnight_high, lv.ib_high, lv.prior_high, lv.prior_close,
                  lv.prior_low, lv.ib_low, lv.overnight_low],
    })
    levels["distance"] = (levels["value"] - micro.price).round(0)
    st.dataframe(levels.round(1), width="stretch", hide_index=True)
    if np.isfinite(lv.on_range_position):
        st.progress(float(np.clip(lv.on_range_position, 0.0, 1.0)),
                    text=f"Position in overnight range: {lv.on_range_position * 100:.0f}%")

    if micro.sweeps:
        st.markdown("**Liquidity sweeps** — level taken out *and rejected*")
        st.dataframe(pd.DataFrame([{
            "level": s.level_name, "side": s.side, "price": round(s.level, 1),
            "penetration": round(s.penetration, 1),
            "RVOL": round(s.rvol, 2) if np.isfinite(s.rvol) else None,
            "confirmed": s.confirmed, "bars ago": s.bars_ago,
            "implies": s.implication,
        } for s in micro.sweeps]), width="stretch", hide_index=True)
    else:
        st.caption("No sweeps detected in the last 12 bars.")

    d1, d2 = st.columns(2)
    d1.metric("Cumulative delta (proxy)",
              f"{micro.cum_delta:,.0f}" if np.isfinite(micro.cum_delta) else "—",
              f"slope {micro.delta_slope:+,.0f}")
    d2.metric("Delta divergence", micro.delta_divergence)
    st.warning(
        "**Delta here is a bar-derived proxy, not order flow.** It measures where "
        "each bar closes within its range, weighted by volume. Real delta needs "
        "Level 2 data that yfinance does not carry — this correlates on trending "
        "bars and is near-meaningless on inside bars. It is weighted at half "
        "strength in the layer score for that reason."
    )
    for f in micro.flags:
        st.caption(f"• {f}")


@panel("Sectors")
def render_sectors():
    st.subheader("Sector rotation — Dow-weighted")
    if not sectors.ok:
        st.warning(sectors.note or "sectors unavailable")
        return
    c1, c2 = st.columns([1, 3])
    c1.metric("Risk tone", sectors.risk_tone)
    c1.metric("XLF vs SPY", f"{sectors.xlf_rs:+.2f}%" if np.isfinite(sectors.xlf_rs) else "—")
    with c2:
        st.dataframe(sectors.table, width="stretch")
    for f in sectors.flags:
        st.warning(f)
    st.caption(
        "No XLU or XLRE: the Dow contains neither utilities nor real estate. "
        "Weights are derived live from component prices."
    )


# ==========================================================================
# Layout
# ==========================================================================
render_signal()
st.divider()

tabs = st.tabs(["Attribution", "Technicals", "Microstructure", "Macro",
                "Regime", "Sectors", "Diagnostics"])
with tabs[0]:
    render_attribution()
with tabs[1]:
    render_technicals()
with tabs[2]:
    render_micro()
with tabs[3]:
    render_macro()
with tabs[4]:
    render_regime()
with tabs[5]:
    render_sectors()
with tabs[6]:
    st.subheader("Diagnostics")
    st.write({
        "attribution": attribution.note,
        "technicals": technicals.note,
        "microstructure": micro.note,
        "macro": macro.note,
        "regime": regime.note,
        "sectors": sectors.note,
        "session_block": f"{block} (scalar {scalar})",
        "coverage": f"{signal.coverage * 100:.0f}%",
        "data_breaker": dl.breaker_status(),
    })
    if dl.breaker_open():
        st.error(
            "Data circuit breaker is OPEN — Yahoo is failing, so every panel is "
            "showing its degraded state rather than stale or wrong numbers. "
            "It closes itself on the next successful fetch."
        )
        if st.button("Reset breaker now"):
            dl.reset_breaker()
            st.cache_data.clear()
            st.rerun()
    st.caption(
        "If a panel is empty, run `validate_tickers.py` — it will tell you "
        "whether the symbol resolves at all before you debug the engine."
    )
