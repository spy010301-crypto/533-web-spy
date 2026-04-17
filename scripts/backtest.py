"""
Donchian Channel Breakout — full backtest engine.

Pipeline
--------
1. fetch_data()        -> pull GLD daily bars from IBKR TWS, fall back to synthetic GBM.
2. compute_atr()       -> Wilder's smoothing for Average True Range.
3. detect_breakouts()  -> add donchian_upper/lower, atr, breakout_long/short columns.
4. backtest()          -> simulate trades one-at-a-time with 3-exit logic.
5. compute_metrics()   -> 10-metric performance summary dict.
6. main()              -> walk-forward split (train=year 1, test=years 2-3), write artifacts.

Outputs written to data/:
  - price_data.csv       full OHLC + indicator frame
  - trade_blotter.csv    one row per closed trade
  - equity.csv           daily equity curve for the test window
  - metrics.json         performance summary for the test window
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Strategy parameters (all named constants)                                   #
# --------------------------------------------------------------------------- #
LOOKBACK_PERIOD = 20       # Donchian channel bars
ATR_PERIOD = 14            # ATR smoothing bars
ATR_MULT = 2.0             # profit target = entry +/- ATR_MULT * ATR
STOP_MULT = 1.0            # stop-loss     = entry -/+ STOP_MULT * ATR
MAX_HOLDING_DAYS = 20      # time-out in calendar days
POSITION_SIZE = 100        # shares per trade
RISK_FREE_RATE = 0.053     # annualised, for Sharpe / Sortino
COMMISSION = 0.001         # 0.1% per side
SLIPPAGE = 0.001           # 0.1% per side

# IBKR connection settings
IB_HOST = "127.0.0.1"
IB_PORT = 7496
IB_CLIENT_ID = 10
TICKER = "GLD"
DURATION = "3 Y"
BAR_SIZE = "1 day"

# Filesystem
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# 1. Data                                                                     #
# --------------------------------------------------------------------------- #
def fetch_data() -> pd.DataFrame:
    """
    Pull 3 years of daily GLD bars from a local TWS session via ib_insync.
    If the connection fails for any reason, generate a synthetic GBM series
    with comparable statistics so the rest of the pipeline always runs.
    """
    try:
        from ib_insync import IB, Stock, util  # type: ignore

        ib = IB()
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=5)
        contract = Stock(TICKER, "SMART", "USD")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=DURATION,
            barSizeSetting=BAR_SIZE,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        ib.disconnect()
        if not bars:
            raise RuntimeError("IBKR returned empty bar list")
        df = util.df(bars)
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
        df["source"] = "ibkr"
        print(f"[fetch_data] Pulled {len(df)} bars from IBKR.")
        return df
    except Exception as e:  # broad on purpose — any failure falls back
        print(f"[fetch_data] IBKR unreachable ({e}); generating synthetic GBM data.")
        return _synthetic_gld(years=3)


def _synthetic_gld(years: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Geometric Brownian Motion with realistic GLD-like parameters.
    mu ~ 6% annualised, sigma ~ 15% annualised, starting price ~ 170.
    """
    rng = np.random.default_rng(seed)
    n = TRADING_DAYS_PER_YEAR * years
    mu_daily = 0.06 / TRADING_DAYS_PER_YEAR
    sigma_daily = 0.15 / math.sqrt(TRADING_DAYS_PER_YEAR)

    # random walk of log returns
    rets = rng.normal(mu_daily, sigma_daily, size=n)
    closes = 170.0 * np.exp(np.cumsum(rets))

    # build plausible OHLC from the close path
    intraday_vol = sigma_daily * 0.6
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    prev_close = 170.0
    for i in range(n):
        o = prev_close * (1 + rng.normal(0, intraday_vol * 0.25))
        c = closes[i]
        hi = max(o, c) * (1 + abs(rng.normal(0, intraday_vol)))
        lo = min(o, c) * (1 - abs(rng.normal(0, intraday_vol)))
        opens[i] = o
        highs[i] = hi
        lows[i] = lo
        prev_close = c

    end = datetime.today().date()
    dates = pd.bdate_range(end=end, periods=n)
    volume = rng.integers(5_000_000, 12_000_000, size=n)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    df["source"] = "synthetic"
    return df


# --------------------------------------------------------------------------- #
# 2. ATR — Wilder smoothing                                                   #
# --------------------------------------------------------------------------- #
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    Wilder's Average True Range: exponential smoothing of True Range with
    alpha = 1/period (equivalent to RMA). True Range is the greatest of
    (high-low), |high - prev_close|, |low - prev_close|.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing = EMA with alpha = 1/period
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


# --------------------------------------------------------------------------- #
# 3. Breakout detection                                                       #
# --------------------------------------------------------------------------- #
def detect_breakouts(df: pd.DataFrame, lookback: int = LOOKBACK_PERIOD) -> pd.DataFrame:
    """
    Detect Donchian channel breakouts.

    Given a DataFrame of daily OHLC bars, this function walks through every
    bar and asks a simple question: did today's close punch through the
    highest high or the lowest low of the PREVIOUS `lookback` bars?

    To guarantee no peek-ahead, the channel levels are built from prior bars
    only -- we take a rolling window of length `lookback` over the high and
    low columns, then `.shift(1)` that window by one bar. That way the value
    recorded on row `t` is the channel implied by data available at the END
    of bar `t-1`, which is exactly what a trader could have seen before
    placing the order at today's close.

    We also attach the Wilder ATR on the same frame; downstream code uses it
    to size profit targets and stop-losses.

    Output columns added:
      donchian_upper   highest high of prior `lookback` bars
      donchian_lower   lowest low  of prior `lookback` bars
      atr              Wilder Average True Range (`ATR_PERIOD`)
      breakout_long    True when close > donchian_upper (go long signal)
      breakout_short   True when close < donchian_lower (go short signal)

    The first `lookback` rows have NaN channel values, so their breakout
    flags are naturally False.
    """
    out = df.copy()
    # rolling window of the PRIOR `lookback` bars — shift(1) enforces this
    out["donchian_upper"] = out["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    out["donchian_lower"] = out["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    out["atr"] = compute_atr(out, ATR_PERIOD)

    out["breakout_long"] = (out["close"] > out["donchian_upper"]).fillna(False)
    out["breakout_short"] = (out["close"] < out["donchian_lower"]).fillna(False)
    return out


# --------------------------------------------------------------------------- #
# 4. Backtest engine                                                          #
# --------------------------------------------------------------------------- #
def backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk bar-by-bar through the frame.

    Rules
    -----
    - Only one position at a time.
    - Entry: next-day open after a breakout signal (signal observed at today's
      close; we act on tomorrow's open to stay realistic).
    - Three possible exits, checked in this order intraday:
        1. profit target      entry + ATR_MULT * ATR (long)
                              entry - ATR_MULT * ATR (short)
        2. stop-loss          entry - STOP_MULT * ATR (long)
                              entry + STOP_MULT * ATR (short)
        3. time-out           forced exit at close after MAX_HOLDING_DAYS bars.
    - Slippage and commission applied on BOTH entry and exit.

    Returns (trade_blotter, equity_curve).
    """
    d = df.reset_index().copy()
    n = len(d)
    trades = []

    in_position = False
    pos_direction = 0  # +1 long, -1 short
    entry_idx = -1
    entry_price_raw = 0.0
    entry_price_net = 0.0
    entry_atr = 0.0
    profit_target = 0.0
    stop_loss = 0.0

    for i in range(n):
        row = d.iloc[i]

        # Manage open position first
        if in_position:
            held_days = i - entry_idx
            outcome = None
            exit_price_raw = None

            high = row["high"]
            low = row["low"]
            close = row["close"]

            # decide which triggers first; for conservatism, if both hit,
            # assume the stop-loss fills first (worse for the trader).
            if pos_direction == 1:
                hit_stop = low <= stop_loss
                hit_target = high >= profit_target
                if hit_stop and hit_target:
                    outcome, exit_price_raw = "Stop-loss", stop_loss
                elif hit_stop:
                    outcome, exit_price_raw = "Stop-loss", stop_loss
                elif hit_target:
                    outcome, exit_price_raw = "Profit target", profit_target
                elif held_days >= MAX_HOLDING_DAYS:
                    outcome, exit_price_raw = "Time-out", close
            else:  # short
                hit_stop = high >= stop_loss
                hit_target = low <= profit_target
                if hit_stop and hit_target:
                    outcome, exit_price_raw = "Stop-loss", stop_loss
                elif hit_stop:
                    outcome, exit_price_raw = "Stop-loss", stop_loss
                elif hit_target:
                    outcome, exit_price_raw = "Profit target", profit_target
                elif held_days >= MAX_HOLDING_DAYS:
                    outcome, exit_price_raw = "Time-out", close

            if outcome is not None:
                # apply slippage + commission on exit
                if pos_direction == 1:
                    exit_price_net = exit_price_raw * (1 - SLIPPAGE) * (1 - COMMISSION)
                else:
                    exit_price_net = exit_price_raw * (1 + SLIPPAGE) * (1 + COMMISSION)

                pnl = (exit_price_net - entry_price_net) * POSITION_SIZE * pos_direction
                trade_return = (exit_price_net / entry_price_net - 1) * pos_direction

                trades.append(
                    {
                        "entry_date": d.iloc[entry_idx]["date"],
                        "exit_date": row["date"],
                        "entry_price": round(entry_price_net, 4),
                        "exit_price": round(exit_price_net, 4),
                        "quantity": POSITION_SIZE,
                        "direction": "Long" if pos_direction == 1 else "Short",
                        "outcome": outcome,
                        "holding_days": held_days,
                        "pnl": round(pnl, 2),
                        "trade_return": round(trade_return, 5),
                    }
                )
                in_position = False
                pos_direction = 0

        # Look for a new entry only if flat and we have tomorrow's open available
        if not in_position and i + 1 < n:
            sig_long = bool(row.get("breakout_long", False))
            sig_short = bool(row.get("breakout_short", False))
            atr_val = row.get("atr", np.nan)
            if (sig_long or sig_short) and not np.isnan(atr_val) and atr_val > 0:
                next_row = d.iloc[i + 1]
                raw_fill = next_row["open"]
                pos_direction = 1 if sig_long else -1
                # apply slippage + commission on entry
                if pos_direction == 1:
                    entry_price_net = raw_fill * (1 + SLIPPAGE) * (1 + COMMISSION)
                else:
                    entry_price_net = raw_fill * (1 - SLIPPAGE) * (1 - COMMISSION)

                entry_idx = i + 1  # trade starts on the fill bar
                entry_price_raw = raw_fill
                entry_atr = atr_val
                if pos_direction == 1:
                    profit_target = entry_price_raw + ATR_MULT * entry_atr
                    stop_loss = entry_price_raw - STOP_MULT * entry_atr
                else:
                    profit_target = entry_price_raw - ATR_MULT * entry_atr
                    stop_loss = entry_price_raw + STOP_MULT * entry_atr
                in_position = True

    blotter = pd.DataFrame(trades)
    if not blotter.empty:
        blotter["entry_date"] = pd.to_datetime(blotter["entry_date"])
        blotter["exit_date"] = pd.to_datetime(blotter["exit_date"])

    # Daily equity curve: cumulative realised PnL stamped on each exit date,
    # forward-filled so dates without exits carry the last known value.
    equity = pd.DataFrame({"date": d["date"]})
    equity["realised_pnl"] = 0.0
    if not blotter.empty:
        # bucket pnl by exit date
        pnl_by_date = blotter.groupby("exit_date")["pnl"].sum()
        mapped = equity["date"].map(pnl_by_date).fillna(0.0)
        equity["realised_pnl"] = mapped
    equity["equity"] = equity["realised_pnl"].cumsum()
    return blotter, equity


# --------------------------------------------------------------------------- #
# 5. Metrics                                                                  #
# --------------------------------------------------------------------------- #
def compute_metrics(blotter: pd.DataFrame, equity: pd.DataFrame | None = None) -> dict:
    """
    Return the 10 required performance statistics.
    Sharpe and Sortino are annualised using TRADING_DAYS_PER_YEAR.
    Max drawdown is computed on the daily equity curve if available,
    otherwise on the trade-sequence PnL.
    """
    m: dict = {
        "n_trades": 0,
        "win_rate": 0.0,
        "avg_return_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "profit_factor": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "total_pnl": 0.0,
    }
    if blotter is None or blotter.empty:
        return m

    returns = blotter["trade_return"].astype(float)
    pnl = blotter["pnl"].astype(float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    m["n_trades"] = int(len(blotter))
    m["win_rate"] = round(float((returns > 0).mean()), 4)
    m["avg_return_pct"] = round(float(returns.mean() * 100), 4)
    m["avg_win_pct"] = round(float(wins.mean() * 100), 4) if len(wins) else 0.0
    m["avg_loss_pct"] = round(float(losses.mean() * 100), 4) if len(losses) else 0.0

    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    m["profit_factor"] = round(gross_win / gross_loss, 4) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    # Sharpe / Sortino from per-trade returns, scaled to annual using the
    # average number of trades per year (approximation when trades are sparse).
    if equity is not None and not equity.empty:
        eq = equity.set_index("date")["equity"].astype(float)
        # daily returns of the equity curve in dollar space, scaled by
        # nominal capital = entry_price * POSITION_SIZE of first trade (rough)
        base = float(blotter["entry_price"].iloc[0]) * POSITION_SIZE
        daily_ret = eq.diff().fillna(0.0) / base
        excess = daily_ret - (RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
        std = daily_ret.std()
        if std > 0:
            m["sharpe"] = round(float(excess.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR)), 4)
        downside = daily_ret[daily_ret < 0].std()
        if downside and downside > 0:
            m["sortino"] = round(float(excess.mean() / downside * math.sqrt(TRADING_DAYS_PER_YEAR)), 4)

        running_max = eq.cummax()
        drawdown = (eq - running_max)
        m["max_drawdown"] = round(float(drawdown.min()), 2)
    else:
        excess = returns - (RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
        if returns.std() > 0:
            m["sharpe"] = round(float(excess.mean() / returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)), 4)
        dn = returns[returns < 0].std()
        if dn and dn > 0:
            m["sortino"] = round(float(excess.mean() / dn * math.sqrt(TRADING_DAYS_PER_YEAR)), 4)
        cum = pnl.cumsum()
        m["max_drawdown"] = round(float((cum - cum.cummax()).min()), 2)

    m["total_pnl"] = round(float(pnl.sum()), 2)
    return m


# --------------------------------------------------------------------------- #
# 6. Main                                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    print("[main] Starting Donchian breakout backtest.")
    raw = fetch_data()
    enriched = detect_breakouts(raw, LOOKBACK_PERIOD)

    # Walk-forward split: first year -> train (parameters already fixed here,
    # so this window is used for indicator warm-up). Years 2 and 3 -> test.
    enriched = enriched.sort_index()
    split_date = enriched.index.min() + timedelta(days=365)
    train = enriched.loc[: split_date].copy()
    test = enriched.loc[split_date:].copy()
    print(f"[main] Train rows: {len(train)}  Test rows: {len(test)}  Split at {split_date.date()}")

    # Run backtest on the test window only (out-of-sample)
    blotter, equity = backtest(test)
    metrics = compute_metrics(blotter, equity)
    metrics["split_date"] = str(split_date.date())
    metrics["ticker"] = TICKER
    metrics["data_source"] = str(raw["source"].iloc[0]) if "source" in raw.columns else "unknown"
    metrics["start_date"] = str(test.index.min().date()) if len(test) else ""
    metrics["end_date"] = str(test.index.max().date()) if len(test) else ""

    # Persist artifacts
    price_out = enriched.reset_index()
    price_out.to_csv(DATA_DIR / "price_data.csv", index=False)
    blotter.to_csv(DATA_DIR / "trade_blotter.csv", index=False)
    equity.to_csv(DATA_DIR / "equity.csv", index=False)
    with open(DATA_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"[main] Wrote {len(blotter)} trades. Total PnL ${metrics['total_pnl']:.2f}")
    print(f"[main] Artifacts in {DATA_DIR}")


if __name__ == "__main__":
    main()
