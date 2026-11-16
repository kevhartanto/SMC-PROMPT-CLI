#!/usr/bin/env python3
"""Train per-symbol 1D/4H/1H setups, then issue fresh conditional calls.

TRAIN: 5y (or actual exchange listing history), walk-forward split, one
mechanical strategy selected per symbol, model JSON + markdown research report.
CALL: latest complete Bitunix 1h candles, only when the selected rule fires.

Not financial advice. No automatic orders or calibrated success probabilities.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import traceback

from mtf_data import download_history, bitunix_latest
from mtf_engine import fit_pair, call_pair

PAIRS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "XRPUSDT",
    "BNBUSDT", "SUIUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _train_one(symbol: str, years: int, cache_dir: str):
    try:
        candles, meta = download_history(symbol, years=years, cache_dir=cache_dir)
        if meta["large_gaps_over_3h"] > 10:
            return symbol, {"symbol": symbol, "status": "DATA_GAPS", "deployment_eligible": False,
                            "data": meta, "note": "Data gaps >3h are too frequent for valid simulation"}
        model = fit_pair(symbol, candles, meta)
        return symbol, model
    except Exception as exc:
        return symbol, {"symbol": symbol, "status": "ERROR", "deployment_eligible": False,
                        "error": f"{type(exc).__name__}: {exc}"}


def do_train(pairs: list[str], years: int, root: Path, cache_dir: str):
    models = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = {executor.submit(_train_one, symbol, years, cache_dir): symbol for symbol in pairs}
        for job in as_completed(jobs):
            sym, outcome = job.result()
            models[sym] = outcome
            print(f"[RESEARCH] {sym} -> {outcome['status']}", flush=True)
    models = {symbol: models[symbol] for symbol in pairs}
    payload = {
        "generated_at_utc": _timestamp(),
        "requested_history_years": years,
        "timeframes": ["1D", "4H", "1H"],
        "markets": {
            "training": "Binance USD-M futures public archived OHLCV",
            "calling": "Bitunix USD-M futures public hourly OHLCV"
        },
        "methods": {
            "selection": "TRAIN > VALIDATION > holdout TEST (test not used in hyperparameter ranking)",
            "signals": ["mechanical FVG retest", "sweep and reclaim", "EMA pullback with close confirmation"],
            "execution": "signal on closed 1H; next 1H open; conservative SL-first; one position at a time",
            "roundtrip_fee_and_slippage_pct": 0.15,
            "funding_and_liquidation": "not modeled",
            "risk_guard": "NOT auto trade; calls conditional"
        },
        "pairs": models
    }
    _write_json(root / "models.json", payload)
    lines = [
        "# MTF 1D / 4H / 1H walk-forward research",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        f"Requested history: {years} years, limited by actual symbol listing and archive availability.",
        "",
        "Historical Binance USD-M Futures 1H data, resampled to completed 4H and 1D.",
        "Trigger: 1H FVG mitigation, sweep-reclaim or EMA pullback with HTF alignment.",
        "Selection uses training+validation only. OOS holdout is a diagnostic and a conservative deployment gate (this introduces deployment selection bias).",
        "Simulated entry next 1H open; stop prioritized for same-bar stop/target; modeled 0.15% roundtrip fees+slippage.",
        "**Not modeled:** funding, liquidation, order-book fills, market impact, exchange basis; no guaranteed winning rate.",
        "",
        "| Pair | Available years | Status | Setup | Train n / Exp(R) | Validation n / Exp(R) | Test n / Exp(R) | Test PF | Test max DD (R) | Eligible for calls |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    eligible = 0
    for sym, m in models.items():
        hist = m.get("data", {})
        params = m.get("selected") or {}
        tr, val, ts = [m.get(z, {}) for z in ("train", "validation", "test")]
        selected = f"{params.get('setup','—')} / {params.get('trend','—')} / {params.get('rr','—')}R" if params else "—"
        good = m.get("deployment_eligible", False)
        eligible += int(good)
        fmt = lambda d: f"{d.get('trades','—')} / {d.get('expectancy_r','—')}"
        lines.append(f"| {sym} | {hist.get('years_available','—')} | {m['status']} | {selected} | {fmt(tr)} | {fmt(val)} | {fmt(ts)} | {ts.get('profit_factor','—')} | {ts.get('max_dd_r','—')} | {'YES' if good else 'NO'} |")
    lines += ["", f"Eligible models: {eligible}/{len(pairs)}.", "",
              "A model with NO_VALIDATED_EDGE or weak holdout deliberately makes NO CALL.",
              "No cross-market/5-year claim is made for pairs with shorter listing histories."]
    for sym, m in models.items():
        if m["status"] in ("DATA_GAPS", "ERROR"):
            lines.append(f"- {sym}: {m.get('error', m.get('note', 'dataset unavailable'))}")
    (root / "train-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if len(pairs) == 0 or all(m["status"] == "ERROR" for m in models.values()):
        raise SystemExit("No symbol could be researched")


def _call_one(sym: str, model: dict):
    try:
        candles, meta = bitunix_latest(sym, hours=24*100)
        return call_pair(model, candles, meta)
    except Exception as exc:
        return {"symbol": sym, "status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"}


def do_call(pairs: list[str], root: Path, model_path: Path):
    if not model_path.is_file():
        raise SystemExit(f"Missing models file: {model_path}; run research training first")
    models = json.loads(model_path.read_text(encoding="utf-8"))
    known = models.get("pairs", {})
    results = []
    # Serial Bitunix pagination to honor public per-IP rate limiting.
    for sym in pairs:
        model = known.get(sym)
        if model is None:
            results.append({"symbol": sym, "status": "NO CALL", "reason": "No trained model for pair"})
            continue
        outcome = _call_one(sym, model)
        results.append(outcome)
        print(f"[CALL] {sym} -> {outcome['status']}", flush=True)
    payload = {
        "generated_at_utc": _timestamp(),
        "trained_at": models.get("generated_at_utc"),
        "results": results
    }
    _write_json(root / "latest-calls.json", payload)
    lines = ["# MTF conditional calls — 1D / 4H / 1H", "",
             f"Generated UTC: {payload['generated_at_utc']}",
             f"Trained UTC: {payload['trained_at']}", "",
             "Only the latest CLOSED hourly candle is eligible. Entries are reference prices, not guaranteed fills.",
             "**No trades are placed, and no measured winning probability is supplied.**", "",
             "| Pair | Status | Direction | Setup | Entry ref | Structural SL | TP (RR gross) | Risk % | Model OOS expectancy R |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for result in results:
        test = result.get("model_test") or {}
        f = lambda key: str(result.get(key, "—"))
        lines.append(
            f"| {result['symbol']} | {result['status']} | {f('direction')} | {f('setup')} | {f('entry_reference')} | "
            f"{f('sl_reference')} | {f('tp_reference')} ({f('rr_before_cost')}) | {f('risk_pct')} | "
            f"{test.get('expectancy_r','—')} |"
        )
    lines += ["", "## Detail / no-call reasons"]
    for result in results:
        lines.append(f"- **{result['symbol']}**: {result.get('reason', result.get('data_warning', 'Conditional setup, confirm current market price'))}")
    (root / "latest-calls.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if results and all(x["status"] == "ERROR" for x in results):
        raise SystemExit("All live calls failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("train", "call"))
    ap.add_argument("--pairs", default=" ".join(PAIRS))
    ap.add_argument("--years", default=5, type=int)
    ap.add_argument("--cache-dir", default=".cache/mtf")
    ap.add_argument("--out", default="research-results")
    ap.add_argument("--models", default="research/models.json")
    args = ap.parse_args()
    if args.years < 1 or args.years > 6:
        ap.error("--years must be 1..6")
    pairs = list(dict.fromkeys(args.pairs.upper().replace(",", " ").split()))
    if not pairs or len(pairs) > 30 or any(not x.endswith("USDT") or not x[:-4].isalnum() for x in pairs):
        ap.error("Expected 1–30 crypto USDT futures symbols")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        do_train(pairs, args.years, root, args.cache_dir)
    else:
        do_call(pairs, root, Path(args.models))


if __name__ == "__main__":
    main()
