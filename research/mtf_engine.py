"""Causal 1D/4H/1H rule learning, backtest and live call selection.

Method:
- 1h bars are the execution frame.
- Higher-timeframe indicators join only AFTER their candle closes.
- Configurations are selected on TRAIN + VALIDATION, not TEST.
- Entry at NEXT 1h candle open, not the setup candle close.
- Stops take priority when TP and SL both touch in the same hourly candle.
- Trading fees/slippage modeled; perpetual funding/liquidation NOT modeled.
- Setups are mechanical ICT/SMC-inspired proxies, not the upstream LLM's opinion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from math import sqrt
from typing import Any
import numpy as np
import pandas as pd

TIMEFRAMES = {"4h": 4, "1d": 24}
FEE_SLIP_PER_SIDE = 0.00075  # 0.05% taker + 0.025% assumed slippage
MIN_HIST_DAYS = 540


@dataclass(frozen=True)
class Strategy:
    setup: str
    trend: str
    stop_atr: float
    rr: float
    max_hold: int


def strategies() -> list[Strategy]:
    # Simple, auditable search space; no black-box unexplainable learning.
    return [
        Strategy(setup, trend, stop_atr, rr, hold)
        for setup, trend, stop_atr, rr, hold in product(
            ("fvg", "sweep", "pullback"),
            ("fast", "steady"),
            (0.8, 1.3),
            (3.0, 4.0),
            (48, 96),
        )
    ]


def _ohlc_agg(candles: pd.DataFrame, frequency: str, expected: int):
    agg = candles.resample(frequency, label="left", closed="left", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    count = candles["close"].resample(frequency, label="left", closed="left", origin="epoch").count()
    return agg.loc[count == expected].dropna()


def prepared(candles: pd.DataFrame) -> pd.DataFrame:
    """Align only finalized higher timeframe candles with 1h close events."""
    df = candles.sort_index().copy()
    if df.empty or df.index.tz is None:
        raise ValueError("Expected timezone-aware hourly OHLCV")
    df["close_time"] = df.index + pd.Timedelta(hours=1)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [(df["high"]-df["low"]),
         (df["high"]-prev_close).abs(),
         (df["low"]-prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=14).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["sweep_low"] = df["low"].shift(1).rolling(12).min()
    df["sweep_high"] = df["high"].shift(1).rolling(12).max()

    # Each 4H/daily bar becomes visible at its end, never its open.
    for frequency, amount in TIMEFRAMES.items():
        high_frame = _ohlc_agg(candles, frequency, amount)
        high_frame["available_at"] = high_frame.index + pd.Timedelta(hours=amount)
        for trend, (fast, slow) in {"fast": (10, 30), "steady": (20, 50)}.items():
            fast_ma = high_frame["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
            slow_ma = high_frame["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
            # 1: bullish, -1: bearish, 0: mixed/insufficient.
            bias = np.select(
                [(fast_ma > slow_ma) & (high_frame["close"] > slow_ma),
                 (fast_ma < slow_ma) & (high_frame["close"] < slow_ma)],
                [1, -1], default=0
            )
            high_frame[trend] = np.where(slow_ma.notna(), bias, 0)
        right = high_frame[["available_at", "fast", "steady"]].sort_values("available_at").copy()
        right = right.rename(columns={t: f"{frequency}_{t}" for t in ("fast", "steady")})
        df = pd.merge_asof(
            df.reset_index().sort_values("close_time"),
            right, left_on="close_time", right_on="available_at", direction="backward"
        ).drop(columns=["available_at"]).set_index("open_time")
        for t in ("fast", "steady"):
            df[f"{frequency}_{t}"] = df[f"{frequency}_{t}"].fillna(0).astype("int8")
    return df


def signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Three causal setups: FVG mitigation, liquidity sweep, EMA pullback confirmation."""
    n = len(df)
    op = df["open"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    ema = df["ema20"].to_numpy(dtype=float)
    p_hi = df["prev_high"].to_numpy(dtype=float)
    p_lo = df["prev_low"].to_numpy(dtype=float)
    sw_lo = df["sweep_low"].to_numpy(dtype=float)
    sw_hi = df["sweep_high"].to_numpy(dtype=float)
    output = {key: np.zeros(n, dtype=np.int8) for key in ("fvg", "sweep", "pullback")}

    # All conditions depend on OHLC at or before the current *closed* 1h bar.
    is_bull_sweep = (lo < sw_lo) & (cl > sw_lo) & (cl > op)
    is_bear_sweep = (hi > sw_hi) & (cl < sw_hi) & (cl < op)
    output["sweep"][is_bull_sweep] = 1
    output["sweep"][is_bear_sweep] = -1

    is_bull_pull = (lo <= ema) & (cl > ema) & (cl > p_hi) & (cl > op)
    is_bear_pull = (hi >= ema) & (cl < ema) & (cl < p_lo) & (cl < op)
    output["pullback"][is_bull_pull] = 1
    output["pullback"][is_bear_pull] = -1

    # Single most recent not-yet-invalidated 1H fair-value gap per side.
    active_bull = None  # (lower, upper, creation index)
    active_bear = None
    for i in range(2, n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if active_bull:
            low_b, high_b, created = active_bull
            if i > created and i - created <= 72:
                retest = (lo[i] <= high_b) and (hi[i] >= low_b)
                bullish_confirmation = cl[i] > 0.5 * (high_b + low_b) and cl[i] > op[i]
                if retest and bullish_confirmation and lo[i] >= low_b - 0.2 * atr[i]:
                    output["fvg"][i] = 1
            if lo[i] < low_b - 0.2 * atr[i] or i - created > 72:
                active_bull = None
        if active_bear:
            low_b, high_b, created = active_bear
            if i > created and i - created <= 72:
                retest = hi[i] >= low_b and lo[i] <= high_b
                bearish_confirmation = cl[i] < 0.5 * (high_b + low_b) and cl[i] < op[i]
                if retest and bearish_confirmation and hi[i] <= high_b + 0.2 * atr[i]:
                    output["fvg"][i] = -1
            if hi[i] > high_b + 0.2 * atr[i] or i - created > 72:
                active_bear = None
        # New gap is not tradable until a subsequent bar.
        if lo[i] - hi[i-2] > 0.12 * atr[i]:
            active_bull = (hi[i-2], lo[i], i)
        if lo[i-2] - hi[i] > 0.12 * atr[i]:
            active_bear = (hi[i], lo[i-2], i)
    return output


def aligned_signals(df: pd.DataFrame, setup: str, trend: str, setup_signals: dict) -> np.ndarray:
    direction = setup_signals[setup].copy()
    d = df[f"1d_{trend}"].to_numpy(dtype=np.int8)
    h = df[f"4h_{trend}"].to_numpy(dtype=np.int8)
    # Both *closed* higher frames must agree; 1h trigger must match the bias.
    return np.where((direction == d) & (direction == h) & (direction != 0),
                    direction, 0).astype(np.int8)


def simulate(df: pd.DataFrame, trade_signals: np.ndarray, params: Strategy,
             fee_slip: float = FEE_SLIP_PER_SIDE) -> list[dict]:
    """Conservative non-overlapping 1h order simulation. Mark R net of cost."""
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    times = df.index
    trades = []
    i = 0
    end = len(df)
    while i < end - 2:
        direction = int(trade_signals[i])
        if direction == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        # Decision is at close of hour i; the next hour's opening price is entry.
        entry_index = i + 1
        entry = float(opens[entry_index])
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        if direction == 1:
            risk = max(params.stop_atr * atr[i], entry - lows[i] + 0.15 * atr[i])
        else:
            risk = max(params.stop_atr * atr[i], highs[i] - entry + 0.15 * atr[i])
        if risk <= 0 or risk / entry > 0.12 or risk / entry < 0.001:
            i += 1
            continue
        stop = entry - direction * risk
        target = entry + direction * params.rr * risk
        max_exit = min(end - 1, entry_index + params.max_hold - 1)
        exit_price = float(closes[max_exit])
        exit_i = max_exit
        result = "TIME"
        for k in range(entry_index, max_exit + 1):
            # adverse gap through the SL executes at the next open, not SL.
            if (direction == 1 and opens[k] <= stop) or (direction == -1 and opens[k] >= stop):
                exit_i, exit_price, result = k, float(opens[k]), "SL_GAP"
                break
            hit_sl = (lows[k] <= stop) if direction == 1 else (highs[k] >= stop)
            hit_tp = (highs[k] >= target) if direction == 1 else (lows[k] <= target)
            if hit_sl:
                exit_i, exit_price, result = k, float(stop), "SL"
                break
            if hit_tp:
                exit_i, exit_price, result = k, float(target), "TP"
                break
        gross_r = direction * (exit_price - entry) / risk
        cost_r = fee_slip * (entry + exit_price) / risk
        trades.append({
            "time": times[i].isoformat(), "exit": times[exit_i].isoformat(),
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": round(entry, 8), "sl": round(stop, 8),
            "tp": round(target, 8), "exit_price": round(exit_price, 8),
            "exit_reason": result, "net_r": round(gross_r - cost_r, 6),
        })
        i = exit_i + 1  # no simultaneous exposure
    return trades


def stats(trades: list[dict]) -> dict:
    r = np.array([t["net_r"] for t in trades], dtype=float)
    if len(r) == 0:
        return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0,
                "net_r": 0.0, "profit_factor": 0.0, "max_dd_r": 0.0}
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    equity = np.concatenate(([0.0], np.cumsum(r)))
    drawdown = np.maximum.accumulate(equity) - equity
    return {
        "trades": len(r), "win_rate": round(100 * np.mean(r > 0), 2),
        "expectancy_r": round(float(r.mean()), 4),
        "net_r": round(float(r.sum()), 3),
        "profit_factor": round(float(gains / losses), 3) if losses else (99.0 if gains else 0.0),
        "max_dd_r": round(float(drawdown.max()), 3),
    }


def segments(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Strict chronological train -> validation -> untouched diagnostic holdout."""
    start = df.index.min()
    end = df.index.max() + pd.Timedelta(hours=1)
    days = (end-start).total_seconds() / 86400
    if days < MIN_HIST_DAYS:
        raise ValueError(f"Insufficient history ({days:.0f} days; at least {MIN_HIST_DAYS})")
    if days >= 5*365 - 15:
        train_end = end - pd.Timedelta(days=730)
        val_end = end - pd.Timedelta(days=365)
    else:
        train_end = start + (end-start)*0.60
        val_end = start + (end-start)*0.80
    return {
        "train": (start.isoformat(), train_end.isoformat()),
        "validation": (train_end.isoformat(), val_end.isoformat()),
        "test": (val_end.isoformat(), end.isoformat()),
    }


def _trades_in_period(trades: list[dict], interval: tuple[str, str]):
    begin, end = pd.Timestamp(interval[0]), pd.Timestamp(interval[1])
    return [t for t in trades if begin <= pd.Timestamp(t["time"]) < end]


def fit_pair(symbol: str, candles: pd.DataFrame, metadata: dict) -> dict:
    data = prepared(candles)
    split = segments(data)
    all_signals = signals(data)
    candidates = []
    # Exclude the first 75 days from trading to allow EMA50 daily to stabilize.
    ready_time = data.index.min() + pd.Timedelta(days=75)
    min_train = 18 if metadata["years_available"] >= 3 else 10
    min_val = 6 if metadata["years_available"] >= 3 else 4
    for params in strategies():
        decision = aligned_signals(data, params.setup, params.trend, all_signals)
        decision[data.index < ready_time] = 0
        transactions = simulate(data, decision, params)
        train = stats(_trades_in_period(transactions, split["train"]))
        val = stats(_trades_in_period(transactions, split["validation"]))
        if (train["trades"] < min_train or val["trades"] < min_val
            or train["expectancy_r"] <= 0 or val["expectancy_r"] <= 0
            or train["profit_factor"] < 1.05 or val["profit_factor"] < 1.05):
            continue
        # Penalize unstable equity rather than just maximizing spectacular RR.
        score = (val["expectancy_r"] * sqrt(min(val["trades"], 40))
                 - 0.025 * val["max_dd_r"])
        candidates.append((score, params, train, val, transactions))
    result = {
        "symbol": symbol, "data": metadata, "splits": split,
        "tested_variants": len(strategies()), "eligible_variants": len(candidates),
        "selection_policy": "train+validation only; TEST not used for hyperparameter selection",
        "cost_model": {"taker_plus_slippage_each_side": FEE_SLIP_PER_SIDE,
                       "funding_included": False,
                       "liquidation_included": False},
    }
    if not candidates:
        result.update({"status": "NO_VALIDATED_EDGE", "deployment_eligible": False})
        return result
    candidates.sort(key=lambda item: (item[0], item[3]["trades"], -item[3]["max_dd_r"]), reverse=True)
    score, params, train, val, trades = candidates[0]
    test_trades = _trades_in_period(trades, split["test"])
    holdout = stats(test_trades)
    # Holdout is a DIAGNOSTIC only; cannot claim it was part of training or guaranteed returns.
    result.update({
        "status": "SELECTED", "deployment_eligible": bool(
            holdout["trades"] >= min_val and holdout["expectancy_r"] > 0
            and holdout["profit_factor"] > 1.0),
        "deployment_gate_note": "Holdout is also used as conservative deployment gate; this makes deployed-model selection subject to testing bias.",
        "selected": asdict(params), "selection_score": round(float(score), 4),
        "train": train, "validation": val, "test": holdout,
        "recent_test_trades": test_trades[-10:]
    })
    return result


def call_pair(model: dict, candles: pd.DataFrame, metadata: dict) -> dict:
    symbol = model["symbol"]
    answer = {"symbol": symbol, "status": "NO CALL", "market": metadata}
    if not model.get("deployment_eligible") or "selected" not in model:
        answer["reason"] = "No validated/deployed strategy"
        return answer
    params = Strategy(**model["selected"])
    df = prepared(candles)
    if df.empty:
        answer["reason"] = "No recent data"
        return answer
    last_close = df.index[-1] + pd.Timedelta(hours=1)
    age = (pd.Timestamp.now(tz="UTC") - last_close).total_seconds() / 60
    if age < -1 or age > 75:
        answer["reason"] = f"Latest finished 1h candle stale (age {age:.1f} minutes)"
        return answer
    sig = signals(df)
    direction = int(aligned_signals(df, params.setup, params.trend, sig)[-1])
    if not direction:
        answer["reason"] = "Selected rule not triggered at latest closed 1h candle"
        return answer
    atr = float(df["atr"].iloc[-1])
    close = float(df["close"].iloc[-1])
    hi, lo = float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        answer["reason"] = "ATR unavailable"
        return answer
    risk = (max(params.stop_atr * atr, close - lo + 0.15 * atr)
            if direction == 1 else
            max(params.stop_atr * atr, hi - close + 0.15 * atr))
    if risk / close > 0.12 or risk / close < 0.001:
        answer["reason"] = "Risk distance outside safe model range"
        return answer
    side = "LONG" if direction == 1 else "SHORT"
    sl, tp = close - direction*risk, close + direction*params.rr*risk
    # These are levels from the last closed 1h candle, not guaranteed fills.
    value = 2.50 / (risk / close)  # nominal trading exposure to risk about USD 2.50 pre-cost
    answer.update({
        "status": "CONDITIONAL CALL",
        "direction": side, "setup": params.setup,
        "last_closed_1h": last_close.isoformat(),
        "entry_reference": round(close, 8), "sl_reference": round(sl, 8),
        "tp_reference": round(tp, 8), "rr_before_cost": params.rr,
        "risk_pct": round(100 * risk / close, 3),
        "position_notional_for_2p5_usd_risk": round(value, 2),
        "data_warning": "TRAIN: Binance Futures archive; CALL: Bitunix; cross-exchange price/basis may differ. Recheck the price before any action.",
        "fill_warning": "Backtest entered at next 1h open; live entry may differ. No order placed.",
        "model_test": model.get("test"),
        "model_validation": model.get("validation")
    })
    return answer
