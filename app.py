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
import dow_options as options_mod
import us30_calendar as cal_mod
import us30_journal as journal_mod
import us30_macro as macro_mod
import us30_micro as micro_mod
import us30_regime as regime_mod
import us30_sectors as sectors_mod
import us30_technicals as tech_mod
import us30_master_signal as master

st.set_page_config(page_title="US30 Monitor", page_icon="📉", layout="wide")

# --------------------------------------------------------------------------
# Deploy health check — runs before anything can crash on a stale constant
# --------------------------------------------------------------------------
# Streamlit Cloud deploys whatever is in the repo at that moment. A push that
# lands a module before its config constants used to kill the app at import
# time with a redacted AttributeError. Now the modules carry their own
# fallbacks and this banner names exactly which file is behind.
EXPECTED_CONFIG_VERSION = 5


# Lives in data_layer so it is unit-testable; reached through getattr so that a
# stale data_layer cannot break the check either.
_module_health = getattr(dl, "module_health", lambda m: ([], []))

_missing: list[str] = []
_stale_modules: list[str] = []
for _mod in (micro_mod, options_mod, cal_mod, journal_mod):
    _m, _s = _module_health(_mod)
    _missing += _m
    _stale_modules += _s

_version = getattr(config, "CONFIG_VERSION", 1)

if _stale_modules:
    st.error(
        f"**Partial deploy detected.** These files are older than `app.py`: "
        f"`{'`, `'.join(_stale_modules)}`. The app is running, but those layers "
        f"may misbehave. Push every changed file in ONE commit, then "
        f"**Manage app → Reboot**."
    )
if _missing or _version < EXPECTED_CONFIG_VERSION:
    st.warning(
        f"**config.py looks stale** — it reports version {_version}, the code "
        f"expects {EXPECTED_CONFIG_VERSION}."
        + (f" Missing: `{'`, `'.join(_missing)}`." if _missing else "")
        + " The app is running on built-in defaults, so nothing is broken, but "
        "push the current `config.py` and reboot to pick up your real settings."
    )


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
try:
    block, scalar = master.current_session_block(now_et)
except Exception:  # noqa: BLE001 — a stale master must not kill the header
    block, scalar = "UNKNOWN", 1.0

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
    st.subheader("Forward test")
    paper_mode = st.toggle(
        "Paper mode", value=True,
        help="Rows are marked paper. Leave this on until the forward test clears "
             "both gates — 30 trades AND 60 days.")
    auto_log = st.toggle(
        "Auto-log actionable signals", value=True,
        help="Blocked signals are never logged — a block is the system working, "
             "not a trade.")
    st.divider()
    st.caption(
        "**All seven layers built.** L6 options still reports unavailable when "
        "component chain liquidity is too thin — that is designed behaviour, "
        "and C6 redistributes its ±10 budget when it happens."
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
        options = options_mod.get_options(attribution)
    except Exception as exc:  # noqa: BLE001
        options = options_mod.OptionsReport(note=f"options crashed: {exc}")

    # Calendar needs the live top-8 ranking, which only attribution can give.
    try:
        top8 = attr.rank_by_weight(attribution, 8) if attribution.ok else config.COMPONENTS[:8]
        calendar = cal_mod.get_calendar(top8)
    except Exception as exc:  # noqa: BLE001
        calendar = cal_mod.CalendarReport(note=f"calendar crashed: {exc}")

    # An ex-dividend drop is mechanical, not information. Recompute attribution
    # with those contributions neutralised — the underlying fetch is cached, so
    # the second pass costs nothing.
    if calendar.ex_div_today:
        try:
            attribution = attr.get_attribution(ex_div_today=set(calendar.ex_div_today))
        except Exception:  # noqa: BLE001
            pass

    try:
        signal = master.build_master_signal(
            attribution=attribution, technicals=technicals, macro=macro,
            regime=regime, sectors=sectors, micro=micro, options=options,
            ctx={
                "spread_pts": spread_pts,
                "slippage_pts": slippage_pts,
                "now": now_et,
                "earnings_top8": calendar.earnings_within_window,
                "ex_div_today": calendar.ex_div_today,
                "cross_index": calendar.cross_index,
                "event_blackout": calendar.active_blackout,
            },
        )
    except Exception as exc:  # noqa: BLE001
        signal = master.MasterSignal()
        st.error(f"Master signal failed: {exc}")

    # ---- journal ---------------------------------------------------------
    try:
        store = journal_mod.get_store()
        journal_mod.expire_stale(store, now_et)
        logged, log_reason = (False, "auto-log off")
        if auto_log:
            logged, log_reason = journal_mod.log_signal(
                store, signal, now_et, regime=regime, options=options,
                paper=paper_mode, spread_pts=spread_pts, slippage_pts=slippage_pts)
        journal_frame = store.load()
        metrics = journal_mod.compute_metrics(journal_frame)
    except Exception as exc:  # noqa: BLE001
        store = journal_mod.MemoryStore()
        journal_frame = journal_mod.empty_frame()
        metrics = journal_mod.Metrics()
        logged, log_reason = False, f"journal crashed: {exc}"


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

    # The research doc promised realised expectancy would sit on the front
    # page, not buried in a tab. This is that promise.
    st.markdown("---")
    st.markdown("**Realised performance** — what this system has actually done")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Expectancy",
              f"{metrics.expectancy_pts:+.1f} pts" if np.isfinite(metrics.expectancy_pts) else "—",
              f"target ≥ {getattr(config, 'TARGET_EXPECTANCY_PTS', 8.0):.0f}")
    pf = metrics.profit_factor
    e2.metric("Profit factor",
              f"{pf:.2f}" if np.isfinite(pf) and pf != float("inf") else ("∞" if pf == float("inf") else "—"),
              f"target ≥ {getattr(config, 'TARGET_PROFIT_FACTOR', 1.4):.1f}")
    e3.metric("Win rate", metrics.win_rate_text,
              help="Always shown with its 95% Wilson interval. A bare percentage "
                   "from a small sample is how a system talks you into trusting noise.")
    e4.metric("Closed / open", f"{metrics.n_closed} / {metrics.n_open}")
    e5.metric("Verdict", metrics.verdict)

    if metrics.verdict == "FORWARD TEST IN PROGRESS":
        min_t = int(getattr(config, "FORWARD_TEST_MIN_TRADES", 30))
        min_d = int(getattr(config, "FORWARD_TEST_MIN_DAYS", 60))
        p1, p2 = st.columns(2)
        p1.progress(min(metrics.n_closed / max(min_t, 1), 1.0),
                    text=f"Trades {metrics.n_closed}/{min_t}")
        p2.progress(min(metrics.days_elapsed / max(min_d, 1), 1.0),
                    text=f"Days {metrics.days_elapsed}/{min_d}")
    if metrics.warnings:
        st.caption("⚠️ " + metrics.warnings[0])


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


@panel("Options")
def render_options():
    st.subheader("Options & gamma — component-weighted")
    st.caption(
        "Built from the top 8 names by price weight, not from DIA. DIA options "
        "trade ~13.9k contracts/day against QQQ's ~1.53M, so a DIA-based gamma "
        "engine reads a handful of institutional hedges rather than the market. "
        "Those 8 names are roughly 48% of the index."
    )

    if not options.ok:
        st.warning(options.note or "options layer unavailable")
        for f in options.flags:
            st.caption(f"• {f}")
        if not options.table.empty:
            st.dataframe(options.table, width="stretch")
        st.info(
            "L6 reporting unavailable is the designed behaviour when liquidity "
            "is too thin — C6 redistributes its ±10 budget across the live "
            "layers rather than scoring it zero."
        )
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    tone = {"LONG_GAMMA": "🔵", "SHORT_GAMMA": "🔴", "NEUTRAL": "⚪"}.get(
        options.gamma_regime, "⚪")
    c1.metric("Gamma regime", f"{tone} {options.gamma_regime.replace('_', ' ').title()}",
              f"{options.aggregate_gex:+.2f}")
    c2.metric("Weight covered", f"{options.weight_covered * 100:.0f}%",
              help="Share of total DJIA price weight with a usable chain")
    c3.metric("Weighted PCR",
              f"{options.weighted_pcr:.2f}" if np.isfinite(options.weighted_pcr) else "—")
    c4.metric("Weighted skew",
              f"{options.weighted_skew:+.3f}" if np.isfinite(options.weighted_skew) else "—",
              help="10% OTM put IV minus 10% OTM call IV, price-weighted")
    c5.metric("Expected move",
              f"±{options.expected_move_pts:,.0f} pts"
              if np.isfinite(options.expected_move_pts) else "—",
              f"vol scalar {options.vol_scalar:.2f}×")

    if options.gamma_regime == "LONG_GAMMA":
        st.info(
            "**Dealers long gamma.** Hedging sells strength and buys weakness, "
            "which pins price and suppresses range. Breakout and continuation "
            "setups underperform here — C11 downgrades them."
        )
    elif options.gamma_regime == "SHORT_GAMMA":
        st.warning(
            "**Dealers short gamma.** Hedging buys strength and sells weakness, "
            "amplifying moves. Fading extremes is the wrong side of the flow — "
            "C12 blocks mean-reversion entries."
        )

    st.markdown("**Per-name gamma ledger**")
    st.dataframe(options.table, width="stretch")

    x1, x2 = st.columns(2)
    x1.metric("DIA net GEX (cross-check)",
              f"${options.dia_net_gex / 1e6:,.1f}m"
              if np.isfinite(options.dia_net_gex) else "—",
              f"OI {options.dia_total_oi:,}")
    x2.metric("DIA agrees with components",
              {True: "yes", False: "no", None: "not usable"}[options.dia_agrees])

    st.warning(
        "**Two assumptions worth holding onto.** Dealer sign is a convention, "
        "not data — nobody outside the clearing system observes which side "
        "dealers are on, and this uses the standard dealers-long-calls, "
        "short-puts convention. And yfinance ships no greeks, so gamma is "
        "computed here from Black-Scholes using the chain's own implied "
        "volatility, which is unreliable on thin strikes. Strikes outside "
        f"±{float(getattr(config, 'OPTIONS_MONEYNESS_BAND', 0.15)) * 100:.0f}% "
        "moneyness and names under the OI floor are dropped for that reason."
    )
    for f in options.flags:
        st.caption(f"• {f}")


@panel("Calendar")
def render_calendar():
    st.subheader("Event risk, earnings & ex-dividends")
    if not calendar.ok:
        st.warning(calendar.note or "calendar unavailable")
        return

    if calendar.active_blackout is not None:
        e = calendar.active_blackout
        mins = e.minutes_until(calendar.now)
        st.error(
            f"**{e.name} blackout active** — {abs(mins):.0f} minutes "
            f"{'until' if mins >= 0 else 'since'} the release ({e.detail}). "
            f"C14 has blocked new entries."
        )
    elif calendar.next_event is not None:
        e = calendar.next_event
        st.info(
            f"Next high-impact event: **{e.name}** "
            f"{e.when:%a %d %b %H:%M} ET — in "
            f"{_humanise_minutes(e.minutes_until(calendar.now))}. {e.detail}"
        )

    c1, c2, c3, c4 = st.columns(4)
    cross = calendar.cross_index or {}
    corr = cross.get("correlation", float("nan"))
    c1.metric("Events in horizon", len(calendar.upcoming))
    c2.metric("Earnings in window", len(calendar.earnings_within_window),
              ", ".join(calendar.earnings_within_window) or "none")
    c3.metric("Ex-div today", len(calendar.ex_div_today),
              ", ".join(calendar.ex_div_today) or "none")
    c4.metric("DJIA/NDX correlation",
              f"{corr:.2f}" if np.isfinite(corr) else "—",
              cross.get("regime", "UNKNOWN"))

    if cross.get("regime") == "ROTATION":
        st.caption(
            "Low correlation is a **rotation regime** — money moving out of tech "
            "into value is bullish for the Dow specifically, which is why C8 "
            "upgrades rather than downgrades here."
        )

    st.markdown("**Upcoming high-impact events**")
    if calendar.events_frame.empty:
        st.caption("Nothing in the horizon.")
    else:
        st.dataframe(calendar.events_frame, width="stretch", hide_index=True)

    st.markdown("**Earnings — top 8 by price weight**")
    st.dataframe(pd.DataFrame([{
        "ticker": e.ticker,
        "next earnings": e.when.strftime("%a %d %b %H:%M") if e.when else "—",
        "source": e.source,
        "confidence": e.confidence,
        "note": e.note,
    } for e in calendar.earnings]), width="stretch", hide_index=True)

    st.markdown("**Ex-dividend dates**")
    st.dataframe(pd.DataFrame([{
        "ticker": d.ticker,
        "ex-date": d.ex_date.strftime("%a %d %b") if d.ex_date else "—",
        "last amount": round(d.amount, 2) if np.isfinite(d.amount) else None,
        "source": d.source,
        "ESTIMATED": d.estimated,
    } for d in calendar.dividends]), width="stretch", hide_index=True)

    st.warning(
        "**yfinance earnings dates are frequently wrong** — stale by months in "
        "some cases, missing entirely in others. C4 therefore widens stops and "
        "caps size rather than blocking, and any ex-dividend date marked "
        "ESTIMATED was inferred from historical payout cadence, not reported. "
        "Verify both against the company's investor-relations page before "
        "leaning on them."
    )

    age = calendar.calendar_age_days
    st.caption(
        f"FOMC and CPI tables verified {getattr(config, 'CALENDAR_VERIFIED_ON', '?')} "
        f"({age} days ago) against federalreserve.gov and the BLS schedule. "
        f"NFP is computed from the first-Friday rule, so it never goes stale."
    )
    for f in calendar.flags:
        st.warning(f)


def _humanise_minutes(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    if minutes < 60 * 24:
        return f"{minutes / 60:.1f} hours"
    return f"{minutes / 1440:.1f} days"


@panel("Journal")
def render_journal():
    st.subheader("Forward-test journal")

    warning = journal_mod.storage_warning(store)
    if warning:
        st.error(f"**{warning}**")
    else:
        st.success(f"Journal stored at `{store.location}` — filesystem is durable here.")

    st.caption(
        f"Auto-log: {'on' if auto_log else 'off'} · last attempt: {log_reason} · "
        f"mode: {'PAPER' if paper_mode else 'LIVE'}"
    )

    # ---- headline -------------------------------------------------------
    m = metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expectancy",
              f"{m.expectancy_pts:+.1f} pts" if np.isfinite(m.expectancy_pts) else "—",
              f"{m.expectancy_r:+.2f}R" if np.isfinite(m.expectancy_r) else "")
    c2.metric("Total", f"{m.total_pts:+,.0f} pts" if np.isfinite(m.total_pts) else "—")
    c3.metric("Max consecutive losses", m.max_consecutive_losses,
              help="Your bot halts after 2. At a lower win rate this fires often — "
                   "see the R-aware circuit-breaker note in the README.")
    c4.metric("Avg slippage",
              f"{m.avg_slippage_pts:.1f} pts" if np.isfinite(m.avg_slippage_pts) else "—",
              help="Measured from actual fills, not modelled.")

    for w in m.warnings:
        st.warning(w)

    # ---- break-even identity, applied to the system's own numbers -------
    cost = spread_pts + slippage_pts
    grade = journal_mod.grade_against_breakeven(m, cost)
    if np.isfinite(grade["required"]):
        st.markdown("**Against its own break-even**")
        g1, g2, g3 = st.columns(3)
        g1.metric("Win rate required", f"{grade['required'] * 100:.1f}%",
                  help="(1 + cost/risk) / (R + 1) at this system's realised R")
        g2.metric("Win rate achieved", f"{m.win_rate * 100:.1f}%",
                  f"{grade['margin'] * 100:+.1f} pts")
        g3.metric("Verdict", grade["verdict"])
        st.caption(
            "This is the identity from the research doc applied to your own "
            "realised numbers rather than restated as a claim. At 0.5:1 "
            "reward-to-risk with a 40pt stop and 5pts of cost, break-even is "
            "exactly 75% — which is why win rate was never the target."
        )

    # ---- breakdowns ------------------------------------------------------
    if not m.by_regime.empty:
        st.markdown("**By regime** — a system that is +20 in trend and −12 in chop "
                    "is not a 60% system, it is a regime filter waiting to be built")
        st.dataframe(m.by_regime, width="stretch")
    if not m.by_session.empty:
        st.markdown("**By session block**")
        st.dataframe(m.by_session, width="stretch")
    if not m.by_conviction.empty:
        st.markdown("**By conviction tier** — if VERY STRONG does not beat WEAK, "
                    "the scoring is not carrying information")
        st.dataframe(m.by_conviction, width="stretch")
    if not m.layer_correlation.empty:
        st.markdown("**Which layers actually predicted anything**")
        st.caption(
            "Correlation between each layer's score and realised points. This is "
            "the question the per-layer columns exist to answer, and the honest "
            "answer may be that some layers are decoration. Needs n≥10 per layer."
        )
        st.bar_chart(m.layer_correlation)

    # ---- record an outcome -----------------------------------------------
    st.markdown("---")
    st.markdown("**Record an outcome**")
    open_rows = journal_frame[journal_frame["status"].astype(str) == journal_mod.OPEN] \
        if not journal_frame.empty else journal_mod.empty_frame()
    if open_rows.empty:
        st.caption("No open signals to close.")
    else:
        labels = {
            f"{r['signal_id']} · {r['direction']} @ {r['entry']:.0f} "
            f"({str(r['logged_at'])[:16]})": r["signal_id"]
            for _, r in open_rows.iterrows()
        }
        with st.form("record_outcome"):
            chosen = st.selectbox("Signal", list(labels.keys()))
            o1, o2 = st.columns(2)
            exit_price = o1.number_input("Exit price", value=0.0, step=1.0, format="%.1f")
            actual_entry = o2.number_input(
                "Actual fill (0 = use planned)", value=0.0, step=1.0, format="%.1f",
                help="Enter your real MT5 fill. The gap between this and the "
                     "planned entry IS your slippage — the thing backtests lie about.")
            reason = st.selectbox("Exit reason",
                                  ["TP1", "TP2", "Stop", "Manual", "Time", "Event"])
            note = st.text_input("Notes", "")
            if st.form_submit_button("Record"):
                ok, msg = journal_mod.record_outcome(
                    store, labels[chosen], exit_price, now_et, reason,
                    actual_entry if actual_entry > 0 else None, note)
                st.success(f"Recorded: {msg}") if ok else st.error(msg)
                st.rerun()

    # ---- export / import --------------------------------------------------
    st.markdown("---")
    st.markdown("**Export and import** — the only thing standing between you and "
                "losing the forward test to a redeploy")
    x1, x2 = st.columns(2)
    with x1:
        st.download_button(
            "Download journal CSV",
            journal_frame.to_csv(index=False).encode(),
            file_name=f"us30_journal_{now_et:%Y%m%d}.csv",
            mime="text/csv",
            disabled=journal_frame.empty,
        )
    with x2:
        uploaded = st.file_uploader("Restore from CSV", type="csv")
        if uploaded is not None:
            try:
                restored = pd.read_csv(uploaded)
                for col in journal_mod.COLUMNS:
                    if col not in restored.columns:
                        restored[col] = np.nan
                merged = pd.concat([journal_frame, restored[journal_mod.COLUMNS]],
                                   ignore_index=True)
                merged = merged.drop_duplicates(
                    subset=["signal_id", "logged_at"], keep="last")
                store.save(merged)
                st.success(f"Merged — journal now holds {len(merged)} rows.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read that CSV: {exc}")

    if not journal_frame.empty:
        st.markdown("**Full ledger**")
        st.dataframe(journal_frame.sort_values("logged_at", ascending=False),
                     width="stretch", height=360)


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

tabs = st.tabs(["Attribution", "Technicals", "Microstructure", "Options",
                "Macro", "Regime", "Sectors", "Calendar", "Journal",
                "Diagnostics"])
with tabs[0]:
    render_attribution()
with tabs[1]:
    render_technicals()
with tabs[2]:
    render_micro()
with tabs[3]:
    render_options()
with tabs[4]:
    render_macro()
with tabs[5]:
    render_regime()
with tabs[6]:
    render_sectors()
with tabs[7]:
    render_calendar()
with tabs[8]:
    render_journal()
with tabs[9]:
    st.subheader("Diagnostics")
    st.write({
        "attribution": attribution.note,
        "technicals": technicals.note,
        "microstructure": micro.note,
        "options": options.note,
        "calendar": calendar.note,
        "journal": f"{metrics.n_closed} closed / {metrics.n_open} open at {store.location}",
        "journal_durable": getattr(store, "durable", False),
        "macro": macro.note,
        "regime": regime.note,
        "sectors": sectors.note,
        "session_block": f"{block} (scalar {scalar})",
        "coverage": f"{signal.coverage * 100:.0f}%",
        "data_breaker": getattr(dl, "breaker_status", lambda: "unavailable")(),
    })
    if getattr(dl, "breaker_open", lambda: False)():
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
