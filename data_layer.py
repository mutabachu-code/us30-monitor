"""
data_layer.py — US30 Monitor

All yfinance access goes through here. Nothing else in the repo imports yfinance.

Three rules this module enforces:
  1. ^DJI has NO volume. Any VWAP/RVOL request is routed to YM=F or DIA.
  2. Every fetch returns a normalised single-level-column DataFrame or None.
     Callers never see a MultiIndex and never see an exception.
  3. The 30-component pull is ONE batched download, cached, never per-ticker.

Importable without streamlit (the test suite does exactly that).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

import config

log = logging.getLogger("us30.data")

# --------------------------------------------------------------------------
# Optional streamlit caching — degrades to a no-op outside the app
# --------------------------------------------------------------------------
try:
    import streamlit as st

    def cache(ttl: int):
        return st.cache_data(ttl=ttl, show_spinner=False)

    HAVE_ST = True
except Exception:  # pragma: no cover - exercised only outside streamlit
    def cache(ttl: int):
        def _wrap(fn):
            return fn
        return _wrap

    HAVE_ST = False

try:
    import yfinance as yf
    HAVE_YF = True
except Exception:  # pragma: no cover
    yf = None
    HAVE_YF = False


# --------------------------------------------------------------------------
# Result envelope — every fetch reports its own health
# --------------------------------------------------------------------------
@dataclass
class Fetch:
    """A fetch that always succeeds structurally, and tells you if it failed."""
    data: pd.DataFrame | None = None
    ok: bool = False
    source: str = ""
    note: str = ""
    missing: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok and self.data is not None and not self.data.empty


# --------------------------------------------------------------------------
# Column normalisation
# --------------------------------------------------------------------------
_OHLCV = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def _flatten(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    yfinance returns different column shapes depending on version, ticker count
    and group_by. Normalise all of them to:
      - single ticker  -> columns are plain OHLCV
      - many tickers   -> columns are a MultiIndex (field, ticker) we keep,
                          but guaranteed field-major and sorted.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    lvl0 = set(map(str, df.columns.get_level_values(0)))
    # If level 0 holds tickers instead of fields, swap so it is field-major.
    if not (lvl0 & set(_OHLCV)):
        df = df.swaplevel(axis=1)
    df = df.sort_index(axis=1)

    if len(tickers) == 1:
        try:
            df = df.xs(tickers[0], axis=1, level=1)
        except (KeyError, ValueError):
            df.columns = df.columns.get_level_values(0)
    return df


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------
# When Yahoo is unreachable, every engine retrying independently turns a
# network outage into a multi-minute hang and an app that looks frozen rather
# than degraded. After a run of total failures, stop dialling for a cooldown
# and let every panel fail fast with an honest message instead.
_BREAKER = {"failures": 0, "open_until": 0.0}
_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN = 60.0


def breaker_open() -> bool:
    return time.monotonic() < _BREAKER["open_until"]


def breaker_status() -> str:
    if not breaker_open():
        return "closed"
    return f"open for {_BREAKER['open_until'] - time.monotonic():.0f}s after {_BREAKER['failures']} consecutive failures"


def reset_breaker() -> None:
    _BREAKER["failures"] = 0
    _BREAKER["open_until"] = 0.0


def _record(success: bool) -> None:
    if success:
        reset_breaker()
        return
    _BREAKER["failures"] += 1
    if _BREAKER["failures"] >= _BREAKER_THRESHOLD:
        _BREAKER["open_until"] = time.monotonic() + _BREAKER_COOLDOWN


def _download(tickers: list[str], *, period: str, interval: str,
              retries: int = 1) -> pd.DataFrame:
    """Raw batched download with bounded retry. Never raises, never hangs long."""
    if not HAVE_YF or breaker_open():
        return pd.DataFrame()
    last_err = ""
    for attempt in range(retries + 1):
        try:
            raw = yf.download(
                tickers,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
            if raw is not None and not raw.empty:
                _record(True)
                return _flatten(raw, tickers)
            last_err = "empty frame"
        except Exception as exc:  # noqa: BLE001 - defensive by design
            last_err = str(exc)
        if attempt < retries:
            time.sleep(0.5)
    _record(False)
    log.warning("download failed for %s (%s/%s): %s",
                tickers[:4], period, interval, last_err)
    return pd.DataFrame()


def _field(df: pd.DataFrame, name: str) -> pd.DataFrame | pd.Series | None:
    """Pull one OHLCV field out of a normalised frame."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if name not in df.columns.get_level_values(0):
            return None
        return df[name]
    return df[name] if name in df.columns else None


# --------------------------------------------------------------------------
# Components — the one heavy call
# --------------------------------------------------------------------------
@cache(config.TTL_COMPONENTS)
def get_components_daily(components: tuple[str, ...] | None = None,
                         period: str = "3mo") -> Fetch:
    """
    ONE batched daily download for all 30 names.
    Returns a frame with a MultiIndex column (field, ticker).
    """
    tickers = list(components or config.COMPONENTS)
    df = _download(tickers, period=period, interval="1d")
    if df.empty:
        return Fetch(ok=False, source="yfinance", note="component batch returned empty")

    close = _field(df, "Close")
    missing: list[str] = []
    if isinstance(close, pd.DataFrame):
        missing = [t for t in tickers
                   if t not in close.columns or close[t].dropna().empty]
    return Fetch(
        data=df, ok=True, source="yfinance",
        note=f"{len(tickers) - len(missing)}/{len(tickers)} names resolved",
        missing=missing,
    )


@cache(config.TTL_INTRADAY)
def get_components_intraday(components: tuple[str, ...] | None = None,
                            interval: str = "5m", period: str = "1d") -> Fetch:
    """Intraday batch for live attribution. Heavier — keep the TTL honest."""
    tickers = list(components or config.COMPONENTS)
    df = _download(tickers, period=period, interval=interval)
    if df.empty:
        return Fetch(ok=False, source="yfinance",
                     note=f"component {interval} batch returned empty")
    return Fetch(data=df, ok=True, source="yfinance",
                 note=f"{interval} batch ok")


# --------------------------------------------------------------------------
# Index / instruments
# --------------------------------------------------------------------------
@cache(config.TTL_DAILY)
def get_index_daily(period: str = "1y") -> Fetch:
    """Cash index daily OHLC. Volume is present but ALWAYS 0 — do not use it."""
    df = _download([config.IDX_SPOT], period=period, interval="1d")
    if df.empty:
        return Fetch(ok=False, source=config.IDX_SPOT, note="no daily index data")
    return Fetch(data=df, ok=True, source=config.IDX_SPOT,
                 note="volume field is zero by design")


@cache(config.TTL_INTRADAY)
def get_index_intraday(interval: str = "5m", period: str = "5d") -> Fetch:
    df = _download([config.IDX_SPOT], period=period, interval=interval)
    if df.empty:
        return Fetch(ok=False, source=config.IDX_SPOT, note="no intraday index data")
    return Fetch(data=df, ok=True, source=config.IDX_SPOT, note=interval)


@cache(config.TTL_INTRADAY)
def get_volume_source(interval: str = "5m", period: str = "5d") -> Fetch:
    """
    Returns the first ticker in VOLUME_SOURCES that actually delivers non-zero
    volume. This is the ONLY correct input for VWAP and RVOL on the Dow.
    """
    for tk in config.VOLUME_SOURCES:
        df = _download([tk], period=period, interval=interval)
        if df.empty:
            continue
        vol = _field(df, "Volume")
        if vol is None:
            continue
        v = pd.to_numeric(pd.Series(np.asarray(vol).ravel()), errors="coerce")
        if v.fillna(0).sum() <= 0:
            log.warning("%s returned zero volume, trying next source", tk)
            continue
        return Fetch(data=df, ok=True, source=tk,
                     note=f"volume source = {tk} @ {interval}")
    return Fetch(ok=False, source="none",
                 note="no volume-bearing source available — VWAP/RVOL disabled")


@cache(config.TTL_INTRADAY)
def get_futures_intraday(interval: str = "5m", period: str = "5d") -> Fetch:
    for tk in (config.IDX_FUT, config.IDX_FUT_MICRO):
        df = _download([tk], period=period, interval=interval)
        if not df.empty:
            return Fetch(data=df, ok=True, source=tk, note=f"{tk} @ {interval}")
    return Fetch(ok=False, source="none", note="no Dow futures data")


# --------------------------------------------------------------------------
# Macro
# --------------------------------------------------------------------------
@cache(config.TTL_MACRO)
def get_macro_daily(period: str = "6mo") -> Fetch:
    """
    One batch for every macro ticker. Yahoo quotes ^TNX/^FVX/^TYX/^IRX as
    yield x 10 — the /10 conversion happens in us30_macro, not here.
    """
    wanted = [v for k, v in config.MACRO_TICKERS.items() if k != "dxy_fallback"]
    df = _download(wanted, period=period, interval="1d")
    if df.empty:
        return Fetch(ok=False, source="yfinance", note="macro batch empty")

    close = _field(df, "Close")
    missing = []
    if isinstance(close, pd.DataFrame):
        missing = [t for t in wanted
                   if t not in close.columns or close[t].dropna().empty]

    # DXY is intermittently flaky on Yahoo — swap in UUP if it came back empty.
    if config.MACRO_TICKERS["dxy"] in missing:
        alt = _download([config.MACRO_TICKERS["dxy_fallback"]],
                        period=period, interval="1d")
        if not alt.empty:
            missing.remove(config.MACRO_TICKERS["dxy"])
            return Fetch(data=df, ok=True, source="yfinance",
                         note="DXY unavailable, UUP substituted", missing=missing)

    return Fetch(data=df, ok=True, source="yfinance",
                 note=f"{len(wanted) - len(missing)}/{len(wanted)} macro tickers",
                 missing=missing)


@cache(config.TTL_MACRO)
def get_sector_daily(period: str = "3mo") -> Fetch:
    etfs = sorted(set(config.SECTOR_ETF.values()) | {"SPY"})
    df = _download(etfs, period=period, interval="1d")
    if df.empty:
        return Fetch(ok=False, source="yfinance", note="sector batch empty")
    return Fetch(data=df, ok=True, source="yfinance", note=f"{len(etfs)} ETFs")


# --------------------------------------------------------------------------
# Option chains
# --------------------------------------------------------------------------
@cache(config.TTL_OPTIONS)
def get_option_chain(ticker: str, max_dte: int | None = None) -> dict:
    """
    Nearest usable expiry for one ticker.

    yfinance returns impliedVolatility, openInterest, volume and bid/ask —
    but NO greeks. Gamma is computed in dow_options from Black-Scholes; this
    function only delivers clean inputs.

    Returns a plain dict so it caches cleanly, never raises, and always carries
    an `ok` flag plus a human-readable `note`.
    """
    out = {"ticker": ticker, "ok": False, "note": "", "expiry": "",
           "calls": pd.DataFrame(), "puts": pd.DataFrame(), "spot": float("nan")}
    if not HAVE_YF or breaker_open():
        out["note"] = "yfinance unavailable or circuit breaker open"
        return out

    max_dte = config.OPTIONS_MAX_DTE if max_dte is None else max_dte
    try:
        tk = yf.Ticker(ticker)
        expiries = list(tk.options or [])
        if not expiries:
            out["note"] = "no expiries listed"
            _record(False)
            return out

        today = pd.Timestamp.utcnow().normalize().tz_localize(None)
        chosen = ""
        for exp in expiries:
            dte = (pd.Timestamp(exp) - today).days
            if 0 <= dte <= max_dte:
                chosen = exp
                break
        if not chosen:
            out["note"] = f"no expiry within {max_dte} days"
            return out

        chain = tk.option_chain(chosen)
        calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
        puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()
        if calls.empty and puts.empty:
            out["note"] = f"empty chain for {chosen}"
            _record(False)
            return out

        hist = tk.history(period="1d", interval="1d", auto_adjust=False)
        spot = float(pd.to_numeric(hist["Close"], errors="coerce").dropna().iloc[-1]) \
            if hist is not None and not hist.empty and "Close" in hist else float("nan")

        out.update(ok=True, expiry=chosen, calls=calls, puts=puts, spot=spot,
                   note=f"{chosen}: {len(calls)} calls / {len(puts)} puts")
        _record(True)
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"chain fetch failed: {str(exc)[:70]}"
        _record(False)
    return out


@cache(config.TTL_DAILY)
def get_cross_index(period: str = "3mo") -> Fetch:
    """NAS100 cash for the C8 cross-index conflict check."""
    df = _download([config.NDX_SPOT], period=period, interval="1d")
    if df.empty:
        return Fetch(ok=False, source=config.NDX_SPOT, note="no NDX data")
    return Fetch(data=df, ok=True, source=config.NDX_SPOT, note="ok")


# --------------------------------------------------------------------------
# Extraction helpers used by every engine
# --------------------------------------------------------------------------
def closes(fetch: Fetch, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Close prices as a clean single-level DataFrame (columns = tickers)."""
    if not fetch:
        return pd.DataFrame()
    c = _field(fetch.data, "Close")
    if c is None:
        return pd.DataFrame()
    if isinstance(c, pd.Series):
        c = c.to_frame(name=fetch.source or "value")
    if tickers is not None:
        keep = [t for t in tickers if t in c.columns]
        c = c[keep]
    return c.apply(pd.to_numeric, errors="coerce")


def ohlcv(fetch: Fetch) -> pd.DataFrame:
    """Single-instrument OHLCV as plain columns. Empty frame if unavailable."""
    if not fetch:
        return pd.DataFrame()
    df = fetch.data
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(df.columns.get_level_values(1)[0], axis=1, level=1)
        except Exception:  # noqa: BLE001
            df.columns = df.columns.get_level_values(0)
    out = pd.DataFrame(index=df.index)
    for f in ("Open", "High", "Low", "Close", "Volume"):
        out[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan
    return out.dropna(subset=["Close"])


def module_health(module) -> tuple[list[str], list[str]]:
    """
    Ask one module which config constants it is missing.

    Written defensively on purpose. A module older than app.py does not HAVE
    config_health(), so calling it directly turned the partial-deploy guard
    into the thing that crashed on a partial deploy. A guard that can crash is
    not a guard.

    Returns (missing_constants, stale_module_names).
    """
    name = getattr(module, "__name__", "unknown")
    fn = getattr(module, "config_health", None)
    if not callable(fn):
        return [], [f"{name}.py"]
    try:
        return list(fn()), []
    except Exception:  # noqa: BLE001
        return [], [f"{name}.py"]


def last_two_closes(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Latest close and prior close per column. Returns two aligned Series."""
    clean = frame.dropna(how="all")
    if len(clean) < 2:
        empty = pd.Series(dtype=float)
        return empty, empty
    return clean.iloc[-1], clean.iloc[-2]
