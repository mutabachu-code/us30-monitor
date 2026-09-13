"""
us30_master_signal.py — US30 Monitor

Seven layers summing directly to ±100. No ±130-to-±100 rescaling step: that
rescale was a recurring source of off-by-a-bit bugs in the NAS100 build and
there is no reason to repeat it.

    L1 attribution      ±25   the largest layer — on a price-weighted index,
                              point attribution is the closest thing to ground
                              truth about what the index is actually doing
    L2 technicals       ±20
    L3 microstructure   ±15
    L4 macro            ±15   weighted up: financials are 27.8% of the Dow
    L5 regime           ±10
    L6 options          ±10   weighted DOWN from the NAS100 build's
                              19% on liquidity evidence: DIA options trade
                              ~13.9k contracts/day against QQQ's ~1.53M
    L7 sectors          ± 5

UNAVAILABLE LAYERS ARE REDISTRIBUTED, NOT ZEROED. A missing layer scoring 0
is not neutral — it silently drags the composite toward the midpoint and makes
every signal look weaker than the evidence supports. Instead its budget is
reallocated across the layers that did report, and the loss is recorded in
`coverage` so the UI can show it honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config


# ==========================================================================
# Layer container
# ==========================================================================
@dataclass
class Layer:
    key: str
    label: str
    budget: int
    score: float = 0.0          # already in ±budget units
    confidence: float = 0.0
    available: bool = False
    note: str = ""

    @property
    def normalised(self) -> float:
        """Score as a fraction of budget, in [-1, 1]."""
        return float(np.clip(self.score / self.budget, -1.0, 1.0)) if self.budget else 0.0


@dataclass
class Conflict:
    code: str
    description: str
    action: str
    severity: str = "warn"      # warn | downgrade | block | upgrade


@dataclass
class TradePlan:
    direction: str = "FLAT"
    entry: float = float("nan")
    entry_low: float = float("nan")
    entry_high: float = float("nan")
    stop: float = float("nan")
    tp1: float = float("nan")
    tp2: float = float("nan")
    risk_pts: float = float("nan")
    reward_pts: float = float("nan")
    rr: float = float("nan")
    lot_multiplier: float = 0.0
    valid: bool = False
    reason: str = ""


@dataclass
class MasterSignal:
    ok: bool = False
    timestamp: str = ""

    raw_score: float = 0.0          # before conflicts, ±100
    final_score: float = 0.0        # after conflicts, ±100
    direction: str = "NEUTRAL"
    conviction: str = "NEUTRAL"

    layers: list[Layer] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)

    coverage: float = 1.0           # fraction of the 100 budget actually reported
    confidence: float = 0.0
    session_block: str = "CLOSED"
    session_scalar: float = 1.0

    blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)

    plan: TradePlan = field(default_factory=TradePlan)
    notes: list[str] = field(default_factory=list)

    @property
    def layer_frame(self) -> pd.DataFrame:
        if not self.layers:
            return pd.DataFrame()
        return pd.DataFrame([{
            "layer": l.label,
            "budget": f"±{l.budget}",
            "score": round(l.score, 1),
            "of_budget": f"{l.normalised * 100:+.0f}%",
            "confidence": f"{l.confidence * 100:.0f}%",
            "status": "live" if l.available else "not built",
            "note": l.note,
        } for l in self.layers])


# ==========================================================================
# Session blocks
# ==========================================================================
def _parse(hhmm: str) -> dtime:
    h, m = hhmm.split(":")
    return dtime(int(h), int(m))


def current_session_block(now: datetime | None = None) -> tuple[str, float]:
    """Which intraday block we are in, and its score scalar."""
    tz = ZoneInfo(config.MARKET_TZ)
    now = now.astimezone(tz) if now is not None else datetime.now(tz)
    if now.weekday() >= 5:
        return "WEEKEND", 0.0
    t = now.time()
    for start, end, name, scalar in config.SESSION_BLOCKS:
        if _parse(start) <= t < _parse(end):
            return name, scalar
    return "CLOSED", 0.0


# ==========================================================================
# Aggregation
# ==========================================================================
def _redistribute(layers: list[Layer]) -> tuple[float, float]:
    """
    Returns (composite score on ±100, coverage fraction).

    Available layers are scaled up so their normalised scores still span the
    full ±100 range. A layer that is merely low-confidence keeps its budget
    but contributes proportionally less.
    """
    live = [l for l in layers if l.available and l.budget > 0]
    if not live:
        return 0.0, 0.0

    live_budget = sum(l.budget for l in live)
    total_budget = sum(l.budget for l in layers) or 100
    coverage = live_budget / total_budget

    # Weight each live layer by budget x confidence.
    num = sum(l.normalised * l.budget * max(l.confidence, 0.05) for l in live)
    den = sum(l.budget * max(l.confidence, 0.05) for l in live)
    composite = (num / den) if den else 0.0
    return float(np.clip(composite, -1.0, 1.0) * 100.0), float(coverage)


def _conviction(score: float) -> str:
    for threshold, label in config.CONVICTION_TIERS:
        if abs(score) >= threshold:
            return label
    return "NEUTRAL"


def _downgrade(score: float, steps: int = 1) -> float:
    """Pull the score toward neutral by one conviction tier per step."""
    return float(score * (0.62 ** steps))


# ==========================================================================
# Conflict resolver
# ==========================================================================
def resolve_conflicts(sig: MasterSignal, ctx: dict) -> MasterSignal:
    """
    C1-C10. `ctx` carries the raw layer reports so conflicts can inspect
    detail the composite score has already thrown away.
    """
    attribution = ctx.get("attribution")
    macro = ctx.get("macro")
    sectors = ctx.get("sectors")
    regime = ctx.get("regime")
    technicals = ctx.get("technicals")
    micro = ctx.get("micro")
    options = ctx.get("options")
    cross = ctx.get("cross_index")
    earnings = ctx.get("earnings_top8") or []
    ex_div = ctx.get("ex_div_today") or []

    score = sig.raw_score
    direction = 1 if score > 0 else (-1 if score < 0 else 0)

    # ---- C1 move-quality gate -------------------------------------------
    # Every attribute is read defensively: a partially-built report must never
    # take the whole master signal down with it.
    if attribution is not None and getattr(attribution, "ok", False):
        net = (getattr(attribution, "pw_advance_pts", 0.0)
               - getattr(attribution, "pw_decline_pts", 0.0))
        top2_share = getattr(attribution, "top2_share", 0.0)
        participation = getattr(attribution, "participation", 0.0)
        if (abs(net) >= config.C1_MIN_MOVE_PTS
                and top2_share >= config.C1_TOP2_SHARE
                and abs(participation) < 0.2):
            table = getattr(attribution, "table", pd.DataFrame())
            top2 = ", ".join(
                table["points"].abs().sort_values(ascending=False).index[:2]
            ) if isinstance(table, pd.DataFrame) and not table.empty else "the top 2"
            sig.conflicts.append(Conflict(
                "C1",
                f"Index moved {net:+.0f}pts but {top2} produced "
                f"{top2_share * 100:.0f}% of it with only "
                f"{getattr(attribution, 'advancers', 0)}A/"
                f"{getattr(attribution, 'decliners', 0)}D participation",
                "Continuation blocked; treat as mean-reversion candidate",
                "block",
            ))
            sig.blocked = True
            sig.block_reasons.append("C1 narrow tape — move is not broad enough to continue")

    # ---- C2 rates conflict ----------------------------------------------
    if macro is not None and sectors is not None and direction > 0:
        yield_bp = getattr(macro, "us10y_change_bp", 0.0)
        xlf_rs = getattr(sectors, "xlf_rs", float("nan"))
        if yield_bp > config.C2_YIELD_BP and getattr(sectors, "xlf_underperforming", False):
            lag = f"{abs(xlf_rs):.2f}%" if np.isfinite(xlf_rs) else "a visible margin"
            sig.conflicts.append(Conflict(
                "C2",
                f"10y +{yield_bp:.1f}bp while XLF lags SPY by {lag} — rates are "
                f"rising and the Dow's 27.8% financials bloc is not benefiting",
                "Downgrade one conviction tier",
                "downgrade",
            ))
            score = _downgrade(score)

    # ---- C3 chop + directional ------------------------------------------
    if regime is not None and getattr(regime, "regime", "") == "CHOP" and direction != 0:
        sig.conflicts.append(Conflict(
            "C3",
            f"Directional signal in CHOP regime "
            f"(ADX {getattr(regime, 'adx_value', float('nan')):.1f}, "
            f"efficiency {getattr(regime, 'efficiency', float('nan')):.2f})",
            "Entry blocked",
            "block",
        ))
        sig.blocked = True
        sig.block_reasons.append("C3 chop regime — no directional entries")

    # ---- C4 earnings blackout --------------------------------------------
    if earnings:
        names = ", ".join(earnings)
        sig.conflicts.append(Conflict(
            "C4",
            f"Top-8 earnings within 24h: {names} — single-name gap risk of "
            f"60-250 index points",
            "Lot multiplier capped at 0.5x, stop widened by 1 ATR",
            "downgrade",
        ))
        sig.notes.append(f"Earnings blackout active: {names}")

    # ---- C5 ex-dividend mechanical drop -----------------------------------
    if ex_div:
        names = ", ".join(ex_div)
        sig.conflicts.append(Conflict(
            "C5",
            f"Ex-dividend today for {names} — the resulting index drop is "
            f"mechanical, not informational",
            "Points neutralised in attribution; gap-based bearish reads suppressed",
            "warn",
        ))

    # ---- C6 thin options liquidity ----------------------------------------
    opt = next((l for l in sig.layers if l.key == "options"), None)
    if opt is not None and not opt.available:
        sig.conflicts.append(Conflict(
            "C6",
            "Options layer unavailable or below liquidity threshold",
            f"±{opt.budget} budget redistributed across live layers, not scored zero",
            "warn",
        ))

    # ---- C7 basis dislocation ---------------------------------------------
    if micro is not None and getattr(micro, "basis_sigma", None) is not None:
        if np.isfinite(micro.basis_sigma) and abs(micro.basis_sigma) > config.C7_BASIS_SIGMA:
            sig.conflicts.append(Conflict(
                "C7",
                f"YM-cash basis {micro.basis_sigma:+.1f}σ from its intraday mean",
                "Hold entry until the basis converges",
                "block",
            ))
            sig.blocked = True
            sig.block_reasons.append("C7 basis dislocation — wait for convergence")

    # ---- C8 cross-index confirmation ---------------------------------------
    if cross is not None and direction != 0:
        corr = cross.get("correlation", float("nan"))
        ndx_dir = cross.get("ndx_direction", 0)
        if np.isfinite(corr) and corr > 0.7 and ndx_dir != 0 and ndx_dir != direction:
            sig.conflicts.append(Conflict(
                "C8",
                f"US30 {'long' if direction > 0 else 'short'} while NDX moves the "
                f"other way at {corr:.2f} correlation",
                "Downgrade one conviction tier",
                "downgrade",
            ))
            score = _downgrade(score)
        elif np.isfinite(corr) and corr < 0.35 and ndx_dir != 0 and ndx_dir != direction:
            sig.conflicts.append(Conflict(
                "C8",
                f"US30 and NDX diverging at low correlation ({corr:.2f}) — "
                f"rotation regime, which favours the Dow specifically",
                "Upgrade one conviction tier",
                "upgrade",
            ))
            score = float(score / 0.62)

    # ---- C9 cost gate -------------------------------------------------------
    spread = ctx.get("spread_pts", config.DEFAULT_SPREAD_PTS)
    slippage = ctx.get("slippage_pts", config.DEFAULT_SLIPPAGE_PTS)
    cost = spread + slippage
    atr_val = getattr(technicals, "atr14", float("nan")) if technicals else float("nan")
    if np.isfinite(atr_val) and atr_val > 0:
        expected_move = atr_val * config.ATR_STOP_MULT * config.TP1_R
        if expected_move < cost * config.C9_COST_MULTIPLE:
            sig.conflicts.append(Conflict(
                "C9",
                f"Expected move {expected_move:.0f}pts is under "
                f"{config.C9_COST_MULTIPLE:.0f}x round-trip cost ({cost:.1f}pts)",
                "Entry blocked regardless of score",
                "block",
            ))
            sig.blocked = True
            sig.block_reasons.append("C9 cost gate — edge does not clear structure")

    # ---- C11 long gamma suppresses breakouts ---------------------------------
    # Dealers long gamma hedge AGAINST the move: they sell into strength and buy
    # weakness. Breakout and continuation setups underperform in that regime.
    if options is not None and getattr(options, "ok", False):
        regime_g = getattr(options, "gamma_regime", "UNKNOWN")
        breakout = False
        if micro is not None:
            levels = getattr(micro, "levels", None)
            breakout = getattr(levels, "state", "") in ("ABOVE_ON", "BELOW_ON")
        if regime_g == "LONG_GAMMA" and breakout and direction != 0:
            sig.conflicts.append(Conflict(
                "C11",
                f"Breakout signal into a long-gamma regime "
                f"(aggregate {getattr(options, 'aggregate_gex', 0):+.2f}) — dealer "
                f"hedging sells strength and buys weakness, pinning price",
                "Downgrade one conviction tier",
                "downgrade",
            ))
            score = _downgrade(score)

        # ---- C12 short gamma punishes fades ----------------------------------
        reversion = getattr(technicals, "mean_reversion", "none") if technicals else "none"
        if regime_g == "SHORT_GAMMA" and reversion != "none":
            sig.conflicts.append(Conflict(
                "C12",
                f"Mean-reversion setup ({reversion}) in a short-gamma regime "
                f"(aggregate {getattr(options, 'aggregate_gex', 0):+.2f}) — dealer "
                f"hedging amplifies moves, so fading is the wrong side of the flow",
                "Entry blocked",
                "block",
            ))
            sig.blocked = True
            sig.block_reasons.append("C12 short gamma — do not fade an amplifying tape")

        # ---- C13 expected-move exhaustion ------------------------------------
        exp_move = getattr(options, "expected_move_pts", float("nan"))
        moved = abs(getattr(attribution, "index_change_pts", float("nan"))) \
            if attribution is not None else float("nan")
        if np.isfinite(exp_move) and exp_move > 0 and np.isfinite(moved):
            used = moved / exp_move
            exhaustion = getattr(config, "EXPECTED_MOVE_EXHAUSTION", 0.95)
            if used >= exhaustion and direction != 0:
                sig.conflicts.append(Conflict(
                    "C13",
                    f"Today's {moved:.0f}pt move is {used * 100:.0f}% of the "
                    f"{exp_move:.0f}pt options-implied expected move",
                    "Continuation blocked — the move is priced out",
                    "block",
                ))
                sig.blocked = True
                sig.block_reasons.append("C13 expected move exhausted")

    # ---- C14 high-impact event blackout --------------------------------------
    # A 15-90 minute scalp has no edge through an FOMC statement or a CPI
    # print. This blocks on the CLOCK, before any layer gets a say, because no
    # amount of confluence survives a number nobody has seen yet.
    blackout = ctx.get("event_blackout")
    if blackout is not None:
        name = getattr(blackout, "name", "event")
        mins = getattr(blackout, "minutes_until", lambda _n: float("nan"))(
            ctx.get("now") or datetime.now(ZoneInfo(config.MARKET_TZ)))
        when = "in" if mins >= 0 else "since"
        sig.conflicts.append(Conflict(
            "C14",
            f"{name} {when} {abs(mins):.0f} minutes "
            f"({getattr(blackout, 'detail', '')})".strip(),
            "Entry blocked through the release window",
            "block",
        ))
        sig.blocked = True
        sig.block_reasons.append(f"C14 {name} blackout")

    # ---- C10 agreement upgrade ----------------------------------------------
    live = [l for l in sig.layers if l.available and abs(l.score) > 0.5]
    if direction != 0 and len(live) >= config.C10_MIN_LAYERS_AGREE:
        agreeing = sum(1 for l in live if np.sign(l.score) == direction)
        if agreeing >= config.C10_MIN_LAYERS_AGREE:
            sig.conflicts.append(Conflict(
                "C10",
                f"{agreeing} of {len(live)} live layers agree in direction",
                "Upgrade one conviction tier",
                "upgrade",
            ))
            score = float(score / 0.62)

    sig.final_score = float(np.clip(score, -100.0, 100.0))
    return sig


# ==========================================================================
# Trade plan
# ==========================================================================
def build_plan(sig: MasterSignal, price: float, atr_val: float,
               earnings_blackout: bool = False) -> TradePlan:
    """
    Risk is defined in ATR multiples, then converted to index points. It is
    NEVER defined in broker points here — see README on the MT5 point trap.
    """
    plan = TradePlan()
    if sig.blocked:
        plan.reason = "; ".join(sig.block_reasons)
        return plan
    if not np.isfinite(price) or not np.isfinite(atr_val) or atr_val <= 0:
        plan.reason = "no price or ATR available"
        return plan
    if abs(sig.final_score) < 25:
        plan.reason = f"score {sig.final_score:+.0f} below MODERATE threshold"
        return plan

    long = sig.final_score > 0
    plan.direction = "LONG" if long else "SHORT"

    stop_distance = atr_val * config.ATR_STOP_MULT
    if earnings_blackout:
        stop_distance += atr_val

    plan.entry = price
    band = atr_val * 0.15
    plan.entry_low = price - band
    plan.entry_high = price + band

    sign = 1.0 if long else -1.0
    plan.stop = price - sign * stop_distance
    plan.tp1 = price + sign * stop_distance * config.TP1_R
    plan.tp2 = price + sign * stop_distance * config.TP2_R

    plan.risk_pts = stop_distance
    plan.reward_pts = stop_distance * config.TP2_R
    plan.rr = config.TP2_R

    if stop_distance * config.TP1_R < config.MIN_TARGET_PTS:
        plan.reason = (
            f"TP1 at {stop_distance * config.TP1_R:.0f}pts is under the "
            f"{config.MIN_TARGET_PTS:.0f}pt structural minimum (YM ticks in whole points)"
        )
        return plan

    # Lot multiplier: conviction x coverage x confidence x session block.
    mult = min(abs(sig.final_score) / 70.0, 1.0)
    mult *= sig.coverage
    mult *= sig.confidence
    mult *= sig.session_scalar
    if earnings_blackout:
        mult *= 0.5
    plan.lot_multiplier = round(float(np.clip(mult, 0.0, 1.0)), 2)
    plan.valid = plan.lot_multiplier > 0.15
    if not plan.valid:
        plan.reason = f"lot multiplier {plan.lot_multiplier:.2f} below 0.15 floor"
    return plan


# ==========================================================================
# Entry point
# ==========================================================================
def build_master_signal(attribution=None, technicals=None, macro=None,
                        regime=None, sectors=None, micro=None, options=None,
                        ctx: dict | None = None) -> MasterSignal:
    """Assemble every layer, aggregate, resolve conflicts, produce a plan."""
    ctx = dict(ctx or {})
    sig = MasterSignal(timestamp=datetime.now(ZoneInfo(config.MARKET_TZ)).isoformat(timespec="seconds"))

    W = config.LAYER_WEIGHTS
    specs = [
        ("attribution", "L1 Attribution & breadth", W["attribution"], attribution),
        ("technicals", "L2 Technicals", W["technicals"], technicals),
        ("microstructure", "L3 Futures & microstructure", W["microstructure"], micro),
        ("macro", "L4 Macro & rates", W["macro"], macro),
        ("regime", "L5 Regime", W["regime"], regime),
        ("options", "L6 Options & gamma", W["options"], options),
        ("sectors", "L7 Sector rotation", W["sectors"], sectors),
    ]

    for key, label, budget, report in specs:
        layer = Layer(key=key, label=label, budget=budget)
        if report is not None and getattr(report, "ok", False):
            layer.score = float(getattr(report, "score", 0.0))
            layer.confidence = float(getattr(report, "confidence", 0.0))
            layer.available = True
            layer.note = str(getattr(report, "note", ""))[:90]
        else:
            layer.note = (str(getattr(report, "note", "")) or "phase not built")[:90]
        sig.layers.append(layer)

    sig.raw_score, sig.coverage = _redistribute(sig.layers)
    sig.session_block, sig.session_scalar = current_session_block(ctx.get("now"))

    live = [l for l in sig.layers if l.available]
    sig.confidence = float(np.mean([l.confidence for l in live])) if live else 0.0

    # Dispersion scales confidence, never direction.
    if regime is not None and getattr(regime, "ok", False):
        sig.confidence = float(np.clip(
            sig.confidence * getattr(regime, "signal_confidence_scalar", 1.0), 0.0, 1.0
        ))
    # Gamma regime is a VOLATILITY read, so it scales conviction the same way —
    # long gamma pins price and makes every directional signal worth less.
    if options is not None and getattr(options, "ok", False):
        sig.confidence = float(np.clip(
            sig.confidence * getattr(options, "vol_scalar", 1.0), 0.0, 1.0
        ))

    ctx.setdefault("attribution", attribution)
    ctx.setdefault("technicals", technicals)
    ctx.setdefault("macro", macro)
    ctx.setdefault("regime", regime)
    ctx.setdefault("sectors", sectors)
    ctx.setdefault("micro", micro)
    ctx.setdefault("options", options)

    sig = resolve_conflicts(sig, ctx)

    # The session block scales conviction, it does not create it.
    sig.final_score *= sig.session_scalar
    if sig.session_scalar == 0.0:
        sig.notes.append(f"{sig.session_block} — cash session closed, signal informational only")

    sig.direction = "LONG" if sig.final_score > 0 else ("SHORT" if sig.final_score < 0 else "NEUTRAL")
    sig.conviction = _conviction(sig.final_score)

    price = getattr(technicals, "price", float("nan")) if technicals else float("nan")
    atr_val = getattr(technicals, "atr14", float("nan")) if technicals else float("nan")
    sig.plan = build_plan(sig, price, atr_val,
                          earnings_blackout=bool(ctx.get("earnings_top8")))
    sig.ok = True
    return sig
