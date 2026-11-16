#!/usr/bin/env python3
"""Read-only multi-pair SMC pre-close screener, using the upstream CLI facts.

This script ranks mechanical *candidates*, not statistically verified trade
signals. Do not place orders based only on this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import re
import subprocess
import sys

DEFAULT_PAIRS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT", "XRPUSDT",
                 "BNBUSDT", "SUIUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT")
PAIRS_RE = re.compile(r"^[A-Z0-9]{3,25}$")
NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def section(text: str, name: str) -> str:
    match = re.search(r"^### " + re.escape(name) + r"\s*$", text, re.M)
    if not match:
        return ""
    rest = text[match.end():]
    stop = re.search(r"^#{2,3} ", rest, re.M)
    return rest[:stop.start()] if stop else rest


def parse_poi(text: str, direction: str, kind: str):
    if kind == "OB":
        pattern = r"^\d\d-\d\d \d\d:\d\d,(BULL_OB|BEAR_OB),([\d.]+),([\d.]+),([\d.]+),(unmitigated|tapped)$"
        accepted = {"unmitigated"}
    else:
        pattern = r"^\d\d-\d\d \d\d:\d\d,(BULL|BEAR),([\d.]+),([\d.]+),([\d.]+),(unfilled|filled)$"
        accepted = {"unfilled"}
    side = ("BULL" if direction == "LONG" else "BEAR") + ("_OB" if kind == "OB" else "")
    found = []
    for m in re.finditer(pattern, text, re.M):
        if m.group(1) == side and m.group(5) in accepted:
            found.append({"kind": kind, "lower": float(m.group(2)), "upper": float(m.group(3)),
                          "mid": float(m.group(4)), "status": m.group(5)})
    return found


def get_swings(text: str, kind: str):
    return [float(v) for v in re.findall(r"^\d\d-\d\d \d\d:\d\d," + kind +
                                         r",([\d.]+),", text, re.M)]


def extract(prompt: str, symbol: str) -> dict:
    result = {"symbol": symbol, "status": "NO SETUP", "score": 0}
    pm = re.search(r"- \*\*Current price:\*\*\s*" + NUMBER, prompt)
    if not pm:
        result["reason"] = "Current price not found"
        return result
    price = float(pm.group(1))
    frames = {}
    for frame in ("4H", "1H", "15m"):
        body = section(prompt, f"Ringkasan Data {'HTF' if frame == '4H' else ('MTF' if frame == '1H' else 'LTF')} ({frame})")
        mm = re.search(r"Klasifikasi struktur \(mekanis\):\s*(\w+)", body)
        atr = re.search(r"ATR\(14\):\s*" + NUMBER, body)
        vol = re.search(r"spike:\s*(yes|no)", body)
        sw = section(prompt, f"Sequence Swing Terdeteksi {'HTF' if frame == '4H' else ('MTF' if frame == '1H' else 'LTF')} ({frame}, oldest→newest)")
        frames[frame] = {
            "bias": mm.group(1).upper() if mm else "UNKNOWN",
            "atr": float(atr.group(1)) if atr else None,
            "volume_spike": bool(vol and vol.group(1) == "yes"),
            "highs": get_swings(sw, "SH"),
            "lows": get_swings(sw, "SL")
        }
    result.update({"price": price, "frames": frames})
    htf, mtf, ltf = (frames[k] for k in ("4H", "1H", "15m"))
    if htf["bias"] not in ("BULLISH", "BEARISH") or not ltf["atr"]:
        result["reason"] = "Missing structure or ATR"
        return result
    direction = "LONG" if htf["bias"] == "BULLISH" else "SHORT"
    result["direction"] = direction
    ltf_label = "LTF (15m, oldest→newest, terbaru maks. "
    pois = (
        parse_poi(section(prompt, "FVG Mekanis (Unfilled) " + ltf_label + "8)"), direction, "FVG") +
        parse_poi(section(prompt, "Order Block Aktif " + ltf_label + "5)"), direction, "OB")
    )
    if not pois:
        result["reason"] = "No unfilled FVG or unmitigated OB in LTF"
        return result
    atr = ltf["atr"]
    candidates = []
    for p in pois:
        entry = p["mid"]
        # For pending limit orders, entry must be on the pullback side of the current price.
        if (direction == "SHORT" and entry <= price) or (direction == "LONG" and entry >= price):
            continue
        distance = (p["lower"] - price) if direction == "SHORT" else (price - p["upper"])
        distance = max(distance, 0.0)
        # Only nearby points of interest are relevant to a pre-close setup.
        if distance > 1.5 * atr:
            continue
        if direction == "SHORT":
            recent_high = ltf["highs"][-1] if ltf["highs"] else p["upper"]
            sl = max(p["upper"], recent_high) + 0.15 * atr
            below = sorted((x for x in htf["lows"] if x < min(entry, price)), reverse=True)
            if not below:
                continue
            tp = below[0]
            reward, risk = entry - tp, sl - entry
        else:
            recent_low = ltf["lows"][-1] if ltf["lows"] else p["lower"]
            sl = min(p["lower"], recent_low) - 0.15 * atr
            above = sorted(x for x in htf["highs"] if x > max(entry, price))
            if not above:
                continue
            tp = above[0]
            reward, risk = tp - entry, entry - sl
        if risk <= 0 or reward <= 0:
            continue
        rr = reward / risk
        score = sum([
            htf["bias"] == mtf["bias"],
            htf["bias"] == ltf["bias"],
            distance <= 1.5 * atr,
            rr >= 3,
            ltf["volume_spike"] or mtf["volume_spike"]
        ])
        consistent = (htf["bias"] == mtf["bias"] == ltf["bias"])
        status = "CANDIDATE" if consistent and rr >= 3 else "WATCH"
        candidates.append({
            "status": status, "score": score, "poi": p, "entry": round(entry, 7),
            "sl": round(sl, 7), "tp": round(tp, 7), "rr": round(rr, 2),
            "distance_atr": round(distance / atr, 2)
        })
    if not candidates:
        result["reason"] = "No nearby unmitigated POI with valid structural SL/TP"
        return result
    candidates.sort(key=lambda c: (c["status"] == "CANDIDATE", c["score"], c["rr"]), reverse=True)
    result.update(candidates[0])
    result["reason"] = "Pre-close watchlist only: no current 4H candle confirmation"
    return result


def run_pair(symbol: str, root: Path) -> dict:
    directory = root / "prompts" / symbol
    directory.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "smc_prompt", symbol, "--preset", "balanced",
           "--provider", "binance", "--output-dir", str(directory)]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=100)
        (directory / "execution.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        if proc.returncode:
            return {"symbol": symbol, "status": "ERROR", "reason": proc.stderr[-500:]}
        paths = sorted(directory.glob("*.md"), key=lambda p: p.stat().st_mtime)
        if not paths:
            return {"symbol": symbol, "status": "ERROR", "reason": "No prompt produced"}
        result = extract(paths[-1].read_text(encoding="utf-8"), symbol)
        result["prompt_file"] = str(paths[-1])
        return result
    except Exception as exc:
        return {"symbol": symbol, "status": "ERROR", "reason": str(exc)}


def clock_note(now: datetime) -> str:
    hour = (now.hour // 4 + 1) * 4
    next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hour - now.hour)
    seconds = (next_close - now).total_seconds()
    if 0 < seconds <= 120:
        status = "WITHIN 120s BEFORE 4H CLOSE"
    elif 120 < seconds <= 240:
        status = "EARLY (runner started ahead of pre-close window)"
    else:
        status = "OFF-WINDOW / LATE (do not interpret as pre-close)"
    return f"{status}; next nominal 4H boundary {next_close.isoformat()} ({seconds:.0f}s remaining)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=" ".join(DEFAULT_PAIRS))
    ap.add_argument("--out", default="scan-output")
    args = ap.parse_args()
    pairs = list(dict.fromkeys(args.pairs.upper().replace(",", " ").split()))
    if not pairs or len(pairs) > 30 or not all(PAIRS_RE.fullmatch(p) for p in pairs):
        ap.error("Supply 1-30 valid futures symbols like HYPEUSDT")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    timing = clock_note(now)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_pair, pair, out): pair for pair in pairs}
        results = [f.result() for f in as_completed(futures)]
    order = {"CANDIDATE": 0, "WATCH": 1, "NO SETUP": 2, "ERROR": 3}
    results.sort(key=lambda r: (order.get(r["status"], 9), -r.get("score", 0),
                                -r.get("rr", 0), r["symbol"]))
    doc = [
        "# Multi-pair SMC / ICT 4H Pre-close Watchlist",
        f"Generated (UTC): {now.isoformat()}",
        f"Timing: {timing}",
        f"Symbols scanned: {len(pairs)}. Preset: Balanced (4H / 1H / 15m).",
        "",
        "**IMPORTANT:** This is a mechanical candidate screen, NOT a verified "
        "entry signal, not a calibrated winning probability, and not an "
        "automatic trade. It uses CLOSED 4H candles, so the candle about to "
        "finish is NOT confirmed. API values can be delayed or exchange-specific. "
        "Confirm on fresh closed candles and apply fees/slippage before placing "
        "any order.",
        "",
        "| Pair | Result | 4H/1H/15m | Side | Snapshot | POI entry | SL | Target 4H | Indicative RR | Score |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        frames = r.get("frames", {})
        biases = "/".join(frames.get(k, {}).get("bias", "?")[:1] for k in ("4H", "1H", "15m"))
        fmt = lambda value: f"{value:.6g}" if isinstance(value, (int, float)) else "—"
        doc.append("| {symbol} | {status} | {bias} | {direction} | {price} | {entry} | {sl} | {tp} | {rr} | {score} |".format(
            symbol=r["symbol"], status=r["status"], bias=biases,
            direction=r.get("direction", "—"), price=fmt(r.get("price")),
            entry=fmt(r.get("entry")), sl=fmt(r.get("sl")), tp=fmt(r.get("tp")),
            rr=fmt(r.get("rr")), score=r.get("score", "—")
        ))
    doc += ["", "## Notes / errors"]
    for r in results:
        doc.append(f"- **{r['symbol']}** [{r['status']}]: {r.get('reason', '—')}")
    (out / "scan-summary.md").write_text("\n".join(doc) + "\n", encoding="utf-8")
    (out / "scan-results.json").write_text(json.dumps(
        {"generated_at": now.isoformat(), "timing": timing, "results": results}, indent=2
    ) + "\n", encoding="utf-8")
    print("\n".join(doc))
    if all(r["status"] == "ERROR" for r in results):
        sys.exit("All pair analyses failed")


if __name__ == "__main__":
    main()
