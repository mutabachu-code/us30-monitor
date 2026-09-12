"""
validate_tickers.py — US30 Monitor, PHASE 0

Run this FIRST, and run it on Streamlit Cloud, not on your laptop. Every
assumption in the architecture doc about data availability is documentation-
sourced, not empirically confirmed — this script is what confirms them.

It answers, for every ticker the build depends on:
  * does the symbol resolve at all?
  * does it return 1m / 5m / 1d bars, and how many?
  * does it carry real volume, or is the field zero? (^DJI will be zero — that
    is the expected result, and the reason VWAP is routed to YM=F/DIA)
  * for the top-8 names, is there a usable options chain?
  * does the derived divisor reconcile to the published reference?

Two ways to run it:
    streamlit run validate_tickers.py      # renders a report in the app
    python validate_tickers.py             # prints a table to the console
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    print("yfinance not installed: pip install yfinance")
    sys.exit(1)


INTERVALS = [("1m", "1d"), ("5m", "5d"), ("1d", "1mo")]


def probe(ticker: str, interval: str, period: str) -> dict:
    row = {"ticker": ticker, "interval": interval, "bars": 0,
           "has_volume": False, "last": np.nan, "error": ""}
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            row["error"] = "empty"
            return row
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        row["bars"] = int(len(df))
        if "Volume" in df.columns:
            v = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
            row["has_volume"] = bool(v.sum() > 0)
        if "Close" in df.columns:
            c = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if not c.empty:
                row["last"] = round(float(c.iloc[-1]), 2)
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:80]
    return row


def probe_options(ticker: str) -> dict:
    row = {"ticker": ticker, "expiries": 0, "strikes": 0,
           "total_oi": 0, "error": ""}
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options or []
        row["expiries"] = len(exps)
        if exps:
            chain = tk.option_chain(exps[0])
            calls, puts = chain.calls, chain.puts
            row["strikes"] = int(len(calls) + len(puts))
            oi = pd.concat([
                pd.to_numeric(calls.get("openInterest", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(puts.get("openInterest", pd.Series(dtype=float)), errors="coerce"),
            ])
            row["total_oi"] = int(oi.fillna(0).sum())
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)[:80]
    return row


def all_tickers() -> list[str]:
    macro = [v for v in config.MACRO_TICKERS.values()]
    instruments = [config.IDX_SPOT, config.IDX_FUT, config.IDX_FUT_MICRO,
                   config.IDX_ETF, config.NDX_SPOT]
    sectors = sorted(set(config.SECTOR_ETF.values()))
    return list(dict.fromkeys(instruments + config.COMPONENTS + macro + sectors))


def run_bars() -> pd.DataFrame:
    rows = []
    for tk in all_tickers():
        for interval, period in INTERVALS:
            rows.append(probe(tk, interval, period))
    return pd.DataFrame(rows)


def run_options(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame([probe_options(t) for t in names])


def run_divisor() -> dict:
    out = {"derived": np.nan, "reference": config.DIVISOR_REFERENCE,
           "drift_pct": np.nan, "index": np.nan, "prices_found": 0, "error": ""}
    try:
        df = yf.download(config.COMPONENTS + [config.IDX_SPOT], period="5d",
                         interval="1d", auto_adjust=False, progress=False)
        close = df["Close"] if "Close" in df.columns.get_level_values(0) else None
        if close is None or close.empty:
            out["error"] = "no close data"
            return out
        latest = close.dropna(how="all").iloc[-1]
        comp = pd.to_numeric(latest.reindex(config.COMPONENTS), errors="coerce").dropna()
        idx = float(latest.get(config.IDX_SPOT, np.nan))
        out["prices_found"] = int(len(comp))
        out["index"] = round(idx, 2)
        if len(comp) == 30 and np.isfinite(idx) and idx > 0:
            derived = float(comp.sum()) / idx
            out["derived"] = derived
            out["drift_pct"] = round(abs(derived / config.DIVISOR_REFERENCE - 1) * 100, 4)
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)[:120]
    return out


# --------------------------------------------------------------------------
def _console() -> None:
    print(f"Probing {len(all_tickers())} tickers across {len(INTERVALS)} intervals...\n")
    bars = run_bars()

    failed = bars[(bars["bars"] == 0)]
    print("=" * 72)
    print("FAILURES")
    print("=" * 72)
    print(failed.to_string(index=False) if not failed.empty else "none")

    print("\n" + "=" * 72)
    print("VOLUME CHECK — ^DJI is EXPECTED to be False")
    print("=" * 72)
    vol = bars[bars["ticker"].isin(
        [config.IDX_SPOT, config.IDX_FUT, config.IDX_FUT_MICRO, config.IDX_ETF]
    )][["ticker", "interval", "bars", "has_volume", "last"]]
    print(vol.to_string(index=False))

    print("\n" + "=" * 72)
    print("DIVISOR RECONCILIATION")
    print("=" * 72)
    d = run_divisor()
    for k, v in d.items():
        print(f"  {k:14s} {v}")
    if np.isfinite(d.get("drift_pct", np.nan)):
        verdict = "OK" if d["drift_pct"] < 0.1 else "DRIFT — check constituents/splits"
        print(f"  verdict        {verdict}")

    print("\n" + "=" * 72)
    print("OPTIONS CHAINS — top 8 by price weight")
    print("=" * 72)
    try:
        close = yf.download(config.COMPONENTS, period="5d", interval="1d",
                            auto_adjust=False, progress=False)["Close"]
        top8 = list(close.dropna(how="all").iloc[-1].sort_values(ascending=False).index[:8])
    except Exception:  # noqa: BLE001
        top8 = config.COMPONENTS[:8]
    print(f"  ranked live: {', '.join(top8)}\n")
    print(run_options(top8 + [config.IDX_ETF]).to_string(index=False))


def _streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="US30 Phase 0 — Ticker Validation", layout="wide")
    st.title("Phase 0 — Ticker validation")
    st.caption(
        "Confirms every data assumption in the architecture before any engine "
        "code depends on it. Run this on Streamlit Cloud."
    )

    if st.button("Run validation", type="primary"):
        with st.spinner("Probing tickers..."):
            bars = run_bars()

        failed = bars[bars["bars"] == 0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Probes run", len(bars))
        c2.metric("Failures", len(failed))
        c3.metric("Distinct tickers", bars["ticker"].nunique())

        if not failed.empty:
            st.error("These probes returned nothing — fix before building on them")
            st.dataframe(failed, width="stretch")

        st.subheader("Volume check")
        st.caption("^DJI must show has_volume = False. That is why VWAP is routed to YM=F or DIA.")
        st.dataframe(
            bars[bars["ticker"].isin([config.IDX_SPOT, config.IDX_FUT,
                                      config.IDX_FUT_MICRO, config.IDX_ETF])],
            width="stretch",
        )

        st.subheader("Divisor reconciliation")
        d = run_divisor()
        st.json(d)
        if np.isfinite(d.get("drift_pct", np.nan)):
            if d["drift_pct"] < 0.1:
                st.success(f"Derived divisor within {d['drift_pct']:.4f}% of reference")
            else:
                st.warning(
                    f"Divisor drift {d['drift_pct']:.4f}% — check for a stock split "
                    f"or a constituent change since 29 Jun 2026"
                )

        st.subheader("Full probe results")
        st.dataframe(bars, width="stretch")
    else:
        st.info("Press the button. This makes roughly 150 Yahoo calls and takes a minute or two.")


if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner as _sr
        in_streamlit = _sr.get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001
        in_streamlit = False
    _streamlit() if in_streamlit else _console()
