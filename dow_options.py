"""
dow_options.py — US30 Monitor, PHASE 5 (L6, ±10)

Component-weighted gamma exposure for the Dow.

WHY THIS IS NOT THE QQQ ENGINE PORTED TO DIA
--------------------------------------------
DIA options trade roughly 13.9k contracts a day against QQQ's 1.53M, and DJX
index options about 3.0k. Strike-level open interest that thin is dominated by
a handful of institutional hedges, so gamma walls computed from it move around
day to day and describe those hedges rather than the market. Copying
`options_intelligence.py` across and pointing it at DIA would produce a panel
that looks identical to the NAS100 one and means far less.

Instead this reads the top N names by PRICE weight — currently GS, CAT, MSFT,
UNH, AMGN, TRV, V, JPM, all of which have deep chains — computes each one's net
gamma exposure, and weights it by that name's DJIA point contribution. Those
eight names are about 48% of the index, so their aggregated dealer positioning
is a truer read on Dow gamma than the index product itself. DIA is still
fetched, but only as a cross-check, and it is flagged when its OI is too thin
to trust.

THREE THINGS THIS MODULE WILL NOT PRETEND TO KNOW
-------------------------------------------------
1. DEALER SIGN IS AN ASSUMPTION, NOT DATA. Nobody outside the clearing system
   observes which side dealers are on. This uses the standard convention —
   dealers long calls, short puts — so positive GEX means dealers are long
   gamma and hedging suppresses movement. That convention is conventional, not
   measured, and every surface says so.

2. yfinance SHIPS NO GREEKS. Gamma is computed here from Black-Scholes using
   the chain's own implied volatility. Quotes may be delayed, and IV on
   illiquid strikes is frequently nonsense, which is why strikes are restricted
   to a moneyness band and names below an OI floor are dropped entirely.

3. GAMMA IS A VOLATILITY REGIME, NOT A DIRECTION. Long gamma suppresses moves,
   short gamma amplifies them. Neither says which way price goes. The layer's
   small directional score comes from skew and put/call ratio; the gamma read
   is exported as a regime that C11-C13 use to modulate other layers, which is
   the honest use of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import data_layer as dl

# --------------------------------------------------------------------------
# Stale-config tolerance (see us30_micro for why this pattern exists)
# --------------------------------------------------------------------------
_CFG_DEFAULTS: dict[str, object] = {
    "OPTIONS_TOP_N": 8,
    "OPTIONS_MIN_OI_PER_NAME": 2000,
    "OPTIONS_MIN_WEIGHT_COVERED": 0.30,
    "OPTIONS_MAX_DTE": 45,
    "OPTIONS_MIN_DTE_HOURS": 2.0,
    "OPTIONS_MONEYNESS_BAND": 0.15,
    "OPTIONS_SKEW_DELTA": 0.10,
    "DIA_THIN_OI": 50_000,
    "DEFAULT_RISK_FREE": 0.04,
    "GAMMA_LONG_THRESHOLD": 0.15,
    "GAMMA_SHORT_THRESHOLD": -0.15,
    "EXPECTED_MOVE_EXHAUSTION": 0.95,
}


def _cfg(name: str):
    return getattr(config, name, _CFG_DEFAULTS[name])


def config_health() -> list[str]:
    return [n for n in _CFG_DEFAULTS if not hasattr(config, n)]


# ==========================================================================
# Black-Scholes, scipy-free
# ==========================================================================
_erf = np.vectorize(math.erf, otypes=[float])
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def d1_d2(spot, strike, t, vol, rate):
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t = np.asarray(t, dtype=float)
    vol = np.asarray(vol, dtype=float)
    denom = vol * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / denom
    return d1, d1 - denom


def bs_gamma(spot, strike, t, vol, rate=0.0):
    """
    d2(price)/d(spot)2. Identical for calls and puts, which is what makes
    net GEX a simple subtraction.
    """
    spot = np.asarray(spot, dtype=float)
    t = np.asarray(t, dtype=float)
    vol = np.asarray(vol, dtype=float)
    d1, _ = d1_d2(spot, strike, t, vol, rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = norm_pdf(d1) / (spot * vol * np.sqrt(t))
    return np.where(np.isfinite(gamma) & (t > 0) & (vol > 0), gamma, 0.0)


def bs_price(spot, strike, t, vol, rate=0.0, is_call=True):
    """Only used to validate gamma against a numerical second derivative."""
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t = np.asarray(t, dtype=float)
    d1, d2 = d1_d2(spot, strike, t, vol, rate)
    disc = np.exp(-rate * t)
    if is_call:
        return spot * norm_cdf(d1) - strike * disc * norm_cdf(d2)
    return strike * disc * norm_cdf(-d2) - spot * norm_cdf(-d1)


# ==========================================================================
# Results
# ==========================================================================
@dataclass
class NameGamma:
    ticker: str
    spot: float = float("nan")
    weight_pct: float = 0.0
    pts_per_1pct: float = 0.0

    expiry: str = ""
    dte: float = float("nan")

    call_gex: float = 0.0
    put_gex: float = 0.0
    net_gex: float = 0.0
    gex_per_pct: float = 0.0        # normalised by spot notional, comparable across names

    call_oi: int = 0
    put_oi: int = 0
    pcr: float = float("nan")

    atm_iv: float = float("nan")
    iv_skew: float = float("nan")   # 10% OTM put IV minus 10% OTM call IV
    expected_move_pct: float = float("nan")
    expected_move_pts: float = float("nan")   # in DJIA points

    gamma_wall: float = float("nan")
    wall_distance_pct: float = float("nan")

    liquid: bool = False
    note: str = ""


@dataclass
class OptionsReport:
    ok: bool = False
    note: str = ""

    names: list[NameGamma] = field(default_factory=list)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    weight_covered: float = 0.0       # share of DJIA weight with usable chains
    liquidity_ok: bool = False

    aggregate_gex: float = 0.0        # weight-normalised, -1..+1
    gamma_regime: str = "UNKNOWN"     # LONG_GAMMA / SHORT_GAMMA / NEUTRAL
    vol_scalar: float = 1.0           # >1 amplifying, <1 suppressive

    weighted_pcr: float = float("nan")
    weighted_skew: float = float("nan")
    expected_move_pts: float = float("nan")

    dia_net_gex: float = float("nan")
    dia_total_oi: int = 0
    dia_agrees: bool | None = None
    dia_note: str = ""

    score: float = 0.0
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)


# ==========================================================================
# Per-name computation
# ==========================================================================
def _clean_side(df: pd.DataFrame, spot: float, band: float) -> pd.DataFrame:
    """Restrict to a moneyness band and drop rows with unusable IV or OI."""
    if df is None or df.empty or not np.isfinite(spot) or spot <= 0:
        return pd.DataFrame()
    out = pd.DataFrame({
        "strike": pd.to_numeric(df.get("strike"), errors="coerce"),
        "iv": pd.to_numeric(df.get("impliedVolatility"), errors="coerce"),
        "oi": pd.to_numeric(df.get("openInterest"), errors="coerce").fillna(0.0),
        "volume": pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0),
    }).dropna(subset=["strike"])
    if out.empty:
        return out
    out = out[(out["strike"] > spot * (1 - band)) & (out["strike"] < spot * (1 + band))]
    # IV of exactly 0, or absurd values, means the strike is not really quoted.
    out = out[(out["iv"] > 0.01) & (out["iv"] < 5.0) & (out["oi"] > 0)]
    # Reset the index: chains arrive with their own labels, and duplicate labels
    # make idxmin/.loc return a Series instead of a scalar further down.
    return out.reset_index(drop=True)


def compute_name_gamma(ticker: str, chain: dict, spot: float,
                       weight_pct: float, pts_per_1pct: float,
                       rate: float | None = None,
                       now: pd.Timestamp | None = None) -> NameGamma:
    """
    Pure function over one already-fetched chain.

    GEX sign convention: dealers long calls, short puts. Positive net GEX means
    dealers are long gamma and their hedging DAMPENS moves; negative means they
    amplify. This is the standard convention and it is an assumption.
    """
    rate = _cfg("DEFAULT_RISK_FREE") if rate is None else rate
    ng = NameGamma(ticker=ticker, spot=spot, weight_pct=weight_pct,
                   pts_per_1pct=pts_per_1pct)

    if not chain or not chain.get("ok"):
        ng.note = (chain or {}).get("note", "no chain")
        return ng

    ng.expiry = chain.get("expiry", "")
    now = pd.Timestamp.utcnow().tz_localize(None) if now is None else now
    try:
        expiry_ts = pd.Timestamp(ng.expiry) + pd.Timedelta(hours=16)
        hours = max((expiry_ts - now).total_seconds() / 3600.0,
                    float(_cfg("OPTIONS_MIN_DTE_HOURS")))
    except Exception:  # noqa: BLE001
        ng.note = "unparseable expiry"
        return ng
    ng.dte = hours / 24.0
    t = hours / (24.0 * 365.0)

    if not np.isfinite(spot) or spot <= 0:
        ng.note = "no spot price"
        return ng

    band = float(_cfg("OPTIONS_MONEYNESS_BAND"))
    calls = _clean_side(chain.get("calls"), spot, band)
    puts = _clean_side(chain.get("puts"), spot, band)
    if calls.empty and puts.empty:
        ng.note = "no usable strikes in the moneyness band"
        return ng

    ng.call_oi = int(calls["oi"].sum()) if not calls.empty else 0
    ng.put_oi = int(puts["oi"].sum()) if not puts.empty else 0
    total_oi = ng.call_oi + ng.put_oi
    ng.pcr = (ng.put_oi / ng.call_oi) if ng.call_oi > 0 else float("nan")

    if total_oi < int(_cfg("OPTIONS_MIN_OI_PER_NAME")):
        ng.note = f"OI {total_oi:,} below floor — excluded"
        return ng

    # Dollar gamma per 1% move: gamma * OI * 100 * S^2 * 0.01
    def side_gex(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        g = bs_gamma(spot, df["strike"].to_numpy(), t, df["iv"].to_numpy(), rate)
        return float(np.sum(g * df["oi"].to_numpy() * 100.0 * spot * spot * 0.01))

    ng.call_gex = side_gex(calls)
    ng.put_gex = side_gex(puts)
    ng.net_gex = ng.call_gex - ng.put_gex

    # Normalise by the name's own option notional so an 8-name aggregate is not
    # simply a ranking of which stock has the most contracts outstanding.
    notional = max(total_oi * 100.0 * spot, 1.0)
    ng.gex_per_pct = ng.net_gex / notional

    # ATM IV and skew.
    both = pd.concat([calls.assign(side="c"), puts.assign(side="p")],
                     ignore_index=True)
    if not both.empty:
        atm_idx = (both["strike"] - spot).abs().idxmin()
        ng.atm_iv = float(both.at[atm_idx, "iv"])

    skew_d = float(_cfg("OPTIONS_SKEW_DELTA"))
    put_iv = _iv_at(puts, spot * (1 - skew_d))
    call_iv = _iv_at(calls, spot * (1 + skew_d))
    if np.isfinite(put_iv) and np.isfinite(call_iv):
        ng.iv_skew = put_iv - call_iv

    # Expected move to expiry, converted to DJIA points via point sensitivity.
    if np.isfinite(ng.atm_iv) and t > 0:
        ng.expected_move_pct = float(ng.atm_iv * math.sqrt(t)) * 100.0
        ng.expected_move_pts = ng.expected_move_pct * pts_per_1pct

    # Gamma wall: the single strike carrying the most gamma-weighted OI.
    wall = _gamma_wall(calls, puts, spot, t, rate)
    if np.isfinite(wall):
        ng.gamma_wall = wall
        ng.wall_distance_pct = (wall / spot - 1.0) * 100.0

    ng.liquid = True
    ng.note = f"{ng.expiry}, OI {total_oi:,}, {ng.dte:.1f} DTE"
    return ng


def _iv_at(df: pd.DataFrame, target_strike: float) -> float:
    if df is None or df.empty or not np.isfinite(target_strike):
        return float("nan")
    idx = (df["strike"] - target_strike).abs().idxmin()
    return float(df.at[idx, "iv"])


def _gamma_wall(calls: pd.DataFrame, puts: pd.DataFrame, spot: float,
                t: float, rate: float) -> float:
    frames = [d for d in (calls, puts) if d is not None and not d.empty]
    if not frames:
        return float("nan")
    combined = pd.concat(frames, ignore_index=True)
    g = bs_gamma(spot, combined["strike"].to_numpy(), t, combined["iv"].to_numpy(), rate)
    weighted = pd.Series(g * combined["oi"].to_numpy(), index=combined["strike"].to_numpy())
    by_strike = weighted.groupby(level=0).sum()
    return float(by_strike.idxmax()) if not by_strike.empty else float("nan")


# ==========================================================================
# Aggregate
# ==========================================================================
def aggregate(names: list[NameGamma], total_top_weight: float,
              dia_net_gex: float = float("nan"), dia_total_oi: int = 0,
              dia_note: str = "") -> OptionsReport:
    """Pure aggregation of already-computed per-name results."""
    rep = OptionsReport(names=names, dia_net_gex=dia_net_gex,
                        dia_total_oi=dia_total_oi, dia_note=dia_note)

    liquid = [n for n in names if n.liquid]
    if not names:
        rep.note = "no names evaluated"
        return rep

    covered = sum(n.weight_pct for n in liquid)
    rep.weight_covered = covered / 100.0
    rep.liquidity_ok = rep.weight_covered >= float(_cfg("OPTIONS_MIN_WEIGHT_COVERED"))

    rep.table = pd.DataFrame([{
        "ticker": n.ticker, "spot": round(n.spot, 2),
        "weight_%": round(n.weight_pct, 2), "expiry": n.expiry,
        "DTE": round(n.dte, 1) if np.isfinite(n.dte) else None,
        "net_GEX_$m": round(n.net_gex / 1e6, 1),
        "GEX/notional": round(n.gex_per_pct, 5),
        "call_OI": n.call_oi, "put_OI": n.put_oi,
        "PCR": round(n.pcr, 2) if np.isfinite(n.pcr) else None,
        "ATM_IV_%": round(n.atm_iv * 100, 1) if np.isfinite(n.atm_iv) else None,
        "skew": round(n.iv_skew, 4) if np.isfinite(n.iv_skew) else None,
        "exp_move_pts": round(n.expected_move_pts, 0) if np.isfinite(n.expected_move_pts) else None,
        "gamma_wall": round(n.gamma_wall, 1) if np.isfinite(n.gamma_wall) else None,
        "wall_%": round(n.wall_distance_pct, 1) if np.isfinite(n.wall_distance_pct) else None,
        "liquid": n.liquid, "note": n.note,
    } for n in names]).set_index("ticker")

    if not liquid:
        rep.flags.append(
            "No component chain cleared the liquidity floor — layer reports "
            "unavailable so its budget is redistributed rather than scored zero"
        )
        rep.note = "no liquid chains"
        return rep

    # Weight each name's normalised GEX by its DJIA price weight.
    w = np.array([n.weight_pct for n in liquid], dtype=float)
    g = np.array([n.gex_per_pct for n in liquid], dtype=float)
    w_sum = w.sum()
    raw = float(np.sum(g * w) / w_sum) if w_sum > 0 else 0.0
    # Scale into a readable -1..+1 band. The divisor is empirical, not derived:
    # normalised GEX for liquid single names typically lands within +/-0.002.
    rep.aggregate_gex = float(np.clip(raw / 0.002, -1.0, 1.0))

    if rep.aggregate_gex >= float(_cfg("GAMMA_LONG_THRESHOLD")):
        rep.gamma_regime = "LONG_GAMMA"
        rep.vol_scalar = 0.85
    elif rep.aggregate_gex <= float(_cfg("GAMMA_SHORT_THRESHOLD")):
        rep.gamma_regime = "SHORT_GAMMA"
        rep.vol_scalar = 1.15
    else:
        rep.gamma_regime = "NEUTRAL"

    pcrs = np.array([n.pcr for n in liquid], dtype=float)
    skews = np.array([n.iv_skew for n in liquid], dtype=float)
    mask_p, mask_s = np.isfinite(pcrs), np.isfinite(skews)
    if mask_p.any():
        rep.weighted_pcr = float(np.sum(pcrs[mask_p] * w[mask_p]) / w[mask_p].sum())
    if mask_s.any():
        rep.weighted_skew = float(np.sum(skews[mask_s] * w[mask_s]) / w[mask_s].sum())

    moves = np.array([n.expected_move_pts for n in liquid], dtype=float)
    mask_m = np.isfinite(moves)
    if mask_m.any():
        # Component expected moves are independent-ish; sum in quadrature and
        # gross up for the weight the top N does not cover.
        combined = float(np.sqrt(np.sum(moves[mask_m] ** 2)))
        coverage = max(w[mask_m].sum() / max(total_top_weight, 1e-9), 1e-9)
        rep.expected_move_pts = combined / math.sqrt(min(coverage, 1.0))

    if np.isfinite(dia_net_gex) and abs(rep.aggregate_gex) > 1e-9:
        rep.dia_agrees = bool(np.sign(dia_net_gex) == np.sign(rep.aggregate_gex))
        if dia_total_oi < int(_cfg("DIA_THIN_OI")):
            rep.flags.append(
                f"DIA cross-check ignored: total OI {dia_total_oi:,} is below the "
                f"{int(_cfg('DIA_THIN_OI')):,} floor, where strike-level gamma is noise"
            )
            rep.dia_agrees = None
        elif not rep.dia_agrees:
            rep.flags.append(
                "DIA gamma disagrees in sign with the component-weighted read — "
                "treat the gamma regime as unresolved"
            )

    # ---- directional score -------------------------------------------------
    # Gamma itself is a VOLATILITY regime and contributes no direction. The
    # small directional read comes from skew and put/call ratio only.
    parts: list[float] = []
    if np.isfinite(rep.weighted_skew):
        # Rich puts relative to calls = demand for downside protection.
        parts.append(float(np.clip(-rep.weighted_skew / 0.06, -1.0, 1.0)) * 0.7)
    if np.isfinite(rep.weighted_pcr):
        if rep.weighted_pcr > 1.6:
            parts.append(0.4)     # crowded hedging is a contrarian positive
            rep.flags.append(
                f"Weighted PCR {rep.weighted_pcr:.2f} — hedging crowded, read as contrarian"
            )
        elif rep.weighted_pcr < 0.55:
            parts.append(-0.4)
            rep.flags.append(
                f"Weighted PCR {rep.weighted_pcr:.2f} — call-heavy, complacent positioning"
            )

    base = float(np.mean(parts)) if parts else 0.0
    rep.score = float(np.clip(base, -1.0, 1.0) * config.LAYER_WEIGHTS["options"])

    conf = rep.weight_covered / 0.48         # 0.48 = the top 8's full weight
    if rep.dia_agrees is False:
        conf *= 0.7
    if len(liquid) < 4:
        conf *= 0.8
    rep.confidence = float(np.clip(conf, 0.0, 1.0))

    rep.ok = rep.liquidity_ok
    if not rep.ok:
        rep.flags.append(
            f"Only {rep.weight_covered * 100:.0f}% of index weight has usable "
            f"chains, under the {float(_cfg('OPTIONS_MIN_WEIGHT_COVERED')) * 100:.0f}% "
            f"floor — layer reports unavailable and C6 redistributes its budget"
        )
    rep.note = (
        f"{len(liquid)}/{len(names)} chains, {rep.weight_covered * 100:.0f}% "
        f"weight covered, {rep.gamma_regime}"
    )
    return rep


# ==========================================================================
# Fetch + compute
# ==========================================================================
def get_options(attribution=None, rate: float | None = None) -> OptionsReport:
    """
    Live options layer. `attribution` supplies the LIVE price-weight ranking —
    the top 8 is never hard-coded, because a split reorders it overnight.
    """
    try:
        import dow_attribution as attr

        rep_attr = attribution
        if rep_attr is None or not getattr(rep_attr, "ok", False):
            rep_attr = attr.get_attribution()
        if not rep_attr.ok or rep_attr.table.empty:
            return OptionsReport(note="attribution unavailable — cannot rank or weight names")

        top_n = int(_cfg("OPTIONS_TOP_N"))
        tickers = attr.rank_by_weight(rep_attr, top_n)
        table = rep_attr.table

        if rate is None:
            macro = dl.closes(dl.get_macro_daily())
            irx = config.MACRO_TICKERS["us13w"]
            if irx in macro.columns:
                series = pd.to_numeric(macro[irx], errors="coerce").dropna()
                if not series.empty:
                    rate = float(series.iloc[-1]) / 1000.0     # x10 quote, then to decimal
        rate = float(_cfg("DEFAULT_RISK_FREE")) if rate is None else rate

        names: list[NameGamma] = []
        for tk in tickers:
            row = table.loc[tk] if tk in table.index else None
            spot = float(row["price"]) if row is not None else float("nan")
            weight = float(row["weight_pct"]) if row is not None else 0.0
            pts = float(row["pts_per_1pct"]) if row is not None else 0.0
            names.append(compute_name_gamma(
                tk, dl.get_option_chain(tk), spot, weight, pts, rate))

        total_top_weight = sum(n.weight_pct for n in names)

        # DIA cross-check only — never the primary source.
        dia_gex, dia_oi, dia_note = float("nan"), 0, ""
        dia_chain = dl.get_option_chain(config.IDX_ETF)
        if dia_chain.get("ok"):
            dia_spot = float(dia_chain.get("spot", float("nan")))
            dia = compute_name_gamma(config.IDX_ETF, dia_chain, dia_spot, 0.0, 0.0, rate)
            dia_gex = dia.net_gex
            dia_oi = dia.call_oi + dia.put_oi
            dia_note = dia.note
        else:
            dia_note = dia_chain.get("note", "DIA chain unavailable")

        return aggregate(names, total_top_weight, dia_gex, dia_oi, dia_note)
    except Exception as exc:  # noqa: BLE001
        return OptionsReport(note=f"options failed: {exc}")
