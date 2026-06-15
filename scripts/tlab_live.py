from __future__ import annotations

import argparse
import csv
import curses
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "reports"
LIVE_DATA_DIR = ROOT / "data"

VALID_MODES = ("live", "snapshot", "minimal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TLAB Live HUD")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="live",
        help="Display mode: live (default), snapshot, minimal",
    )
    return parser.parse_args()

REFRESH_SECONDS = 3
MAX_RECENT_TRADES = 4
MAX_ALERTS = 3


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def first_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def last_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[-1] if rows else {}


def pick(d: dict[str, Any], *keys: str, default: str = "n/a") -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def fnum(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def ffloat(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def build_price_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(pick(row, "ticker", default="")).strip().upper()
        if ticker:
            out[ticker] = row
    return out


def calc_unrealized_pct(direction: str, entry: float, current: float) -> float | None:
    if entry <= 0:
        return None
    if direction == "long":
        return ((current - entry) / entry) * 100.0
    if direction == "short":
        return ((entry - current) / entry) * 100.0
    return None


def calc_stop_distance_pct(direction: str, current: float, stop: float) -> float | None:
    if current <= 0:
        return None
    if direction == "long":
        return ((current - stop) / current) * 100.0
    if direction == "short":
        return ((stop - current) / current) * 100.0
    return None


def safe_add(stdscr, y: int, x: int, text: str, attr: int = 0) -> None:
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    clipped = text[: max(0, w - x - 1)]
    if not clipped:
        return
    try:
        stdscr.addstr(y, x, clipped, attr)
    except curses.error:
        pass


def hline(stdscr, y: int, x: int, width: int) -> None:
    if width <= 1:
        return
    safe_add(stdscr, y, x, "─" * max(0, width), curses.A_DIM)


def draw_box(stdscr, y: int, x: int, w: int, h: int, title: str) -> None:
    if w < 8 or h < 3:
        return
    safe_add(stdscr, y, x, "┌" + "─" * (w - 2) + "┐")
    safe_add(stdscr, y + h - 1, x, "└" + "─" * (w - 2) + "┘")
    for row in range(y + 1, y + h - 1):
        safe_add(stdscr, row, x, "│")
        safe_add(stdscr, row, x + w - 1, "│")
    label = f" {title} "
    if len(label) < w - 2:
        safe_add(stdscr, y, x + 2, label, curses.A_BOLD)


def build_alerts(
    metrics: dict[str, Any],
    trades: list[dict[str, Any]],
    open_trades: list[dict[str, Any]],
    price_map: dict[str, dict[str, Any]],
) -> list[str]:
    alerts = []
    try:
        if float(pick(metrics, "sharpe", default="0")) <= 0:
            alerts.append(f"Sharpe weak: {fnum(metrics.get('sharpe'))}")
    except Exception:
        pass
    try:
        if float(pick(metrics, "max_drawdown", default="0")) <= -0.10:
            alerts.append(f"Drawdown high: {fnum(metrics.get('max_drawdown'))}")
    except Exception:
        pass
    if not trades:
        alerts.append("Trade log empty")
    if not open_trades:
        alerts.append("Open trades empty")

    for row in open_trades[:MAX_ALERTS]:
        try:
            ticker = str(pick(row, "ticker", default="n/a")).upper()
            direction = str(pick(row, "side", "direction", default="n/a")).lower()
            stop = ffloat(pick(row, "stop_price"))
            price_row = price_map.get(ticker, {})
            current = ffloat(pick(price_row, "current_price"))
            if stop is None or current is None:
                continue
            dist = calc_stop_distance_pct(direction, current, stop)
            if dist is not None and dist <= 2.0:
                alerts.append(f"Stop near: {ticker} {dist:.1f}%")
        except Exception:
            pass

    if not alerts:
        alerts.append("No active warnings")
    return alerts[:MAX_ALERTS]



def build_symbol_summary(open_trades, price_map):
    grouped = {}

    for r in open_trades:
        symbol = (
            pick(r, "ticker")
            or pick(r, "symbol")
            or pick(r, "underlying")
            or pick(r, "asset")
            or pick(r, "instrument")
            or pick(r, "pair")
            or "n/a"
        )
        symbol = str(symbol).strip() or "n/a"
        direction = str(pick(r, "side", "direction", default="")).lower()
        qty = pick(r, "qty", pick(r, "quantity", 1))
        entry = pick(r, "entry", pick(r, "entry_price", 0))
        curr = price_map.get(symbol, entry)

        try:
            qty = float(qty or 0)
        except Exception:
            qty = 0.0

        try:
            entry = float(entry or 0)
        except Exception:
            entry = 0.0

        try:
            curr = float(curr or 0)
        except Exception:
            curr = entry

        side = 1.0 if direction == "long" else -1.0
        upnl_pct = ((curr - entry) / entry * 100.0 * side) if entry else 0.0

        bucket = grouped.setdefault(symbol, {"count": 0, "sum_upnl": 0.0, "long": 0, "short": 0})
        bucket["count"] += 1
        bucket["sum_upnl"] += upnl_pct
        if direction == "long":
            bucket["long"] += 1
        elif direction == "short":
            bucket["short"] += 1

    out = []
    for symbol, v in grouped.items():
        bias = "L" if v["long"] >= v["short"] else "S"
        avg_upnl = (v["sum_upnl"] / v["count"]) if v["count"] else 0.0
        out.append(f"{symbol} x{v['count']} {bias} avgUPNL {avg_upnl:+.1f}%")

    return out

def draw(stdscr, mode: str) -> None:
    metrics = first_row(load_csv(DATA_DIR / "metrics.csv"))
    trades = load_csv(LIVE_DATA_DIR / "backtests" / "trade_log.csv")
    open_trades = load_csv(LIVE_DATA_DIR / "trades" / "open_trades.csv")
    current_prices = load_csv(LIVE_DATA_DIR / "live" / "current_prices.csv")
    price_map = build_price_map(current_prices)
    macro = last_row(load_csv(DATA_DIR / "macro_snapshot.csv"))
    alerts = build_alerts(metrics, trades, open_trades, price_map)
    recent = trades[-MAX_RECENT_TRADES:]
    open_recent = open_trades[:2]

    open_count = len(open_trades)
    long_count = sum(1 for r in open_trades if str(pick(r, "side", "direction", default="")).lower() == "long")
    short_count = sum(1 for r in open_trades if str(pick(r, "side", "direction", default="")).lower() == "short")
    symbol_summary = build_symbol_summary(open_trades, price_map)

    stdscr.erase()
    h, w = stdscr.getmaxyx()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    refresh_label = "static" if mode == "snapshot" else f"{REFRESH_SECONDS}s"

    safe_add(stdscr, 0, 1, "TLAB LIVE HUD", curses.A_BOLD)
    safe_add(stdscr, 1, 1, f"{now}   refresh {refresh_label}", curses.A_DIM)
    hline(stdscr, 2, 1, max(10, w - 2))

    full_w = max(40, w - 2)
    left_w = max(24, (w - 3) // 2)
    right_w = max(24, w - left_w - 3)

    y = 3
    draw_box(stdscr, y, 1, full_w, 3, "SYSTEM")
    safe_add(stdscr, y + 1, 3, f"mode {mode:<8} status running    heartbeat {now}")

    y = 7
    draw_box(stdscr, y, 1, full_w, 4, "KPI")
    safe_add(
        stdscr,
        y + 1,
        3,
        f"trades   {pick(metrics, 'n_trades'):<8}   return   {fnum(pick(metrics, 'total_return')):<8}   cagr   {fnum(pick(metrics, 'cagr'))}",
    )
    safe_add(
        stdscr,
        y + 2,
        3,
        f"sharpe   {fnum(pick(metrics, 'sharpe')):<8}   dd       {fnum(pick(metrics, 'max_drawdown')):<8}   win    {fnum(pick(metrics, 'win_rate'))}",
    )

    y = 12
    draw_box(stdscr, y, 1, left_w, 4, "ALERTS")
    for i, alert in enumerate(alerts[:2]):
        attr = curses.A_DIM if alert == "No active warnings" else curses.A_BOLD
        safe_add(stdscr, y + 1 + i, 3, f"! {alert}", attr)

    draw_box(stdscr, y, left_w + 2, right_w, 4, "MACRO")
    safe_add(stdscr, y + 1, left_w + 4, f"date   {pick(macro, 'date')}")
    safe_add(
        stdscr,
        y + 2,
        left_w + 4,
        f"rate   {fnum(pick(macro, 'rate'))}   infl {fnum(pick(macro, 'inflation'))}   gdp {fnum(pick(macro, 'gdp'), 0)}",
    )

    y = 17
    draw_box(stdscr, y, 1, full_w, 6, "OPEN TRADES")
    summary_text = " | ".join(symbol_summary[:1]) if symbol_summary else "n/a"
    safe_add(stdscr, y + 1, 3, f"open {open_count}   long {long_count}   short {short_count}   {summary_text}")

    if open_recent:
        for i, row in enumerate(open_recent[:1]):
            ticker = str(pick(row, 'ticker')).upper()
            direction = str(pick(row, 'side', 'direction')).lower()
            entry = ffloat(pick(row, 'entry_price'))
            stop = ffloat(pick(row, 'stop_price'))
            price_row = price_map.get(ticker, {})
            current = ffloat(pick(price_row, 'current_price'))

            upnl = calc_unrealized_pct(direction, entry, current) if entry is not None and current is not None else None
            stop_dist = calc_stop_distance_pct(direction, current, stop) if stop is not None and current is not None else None

            if mode == "minimal":
                safe_add(
                    stdscr,
                    y + 2 + i,
                    3,
                    f"{ticker} {direction}  entry {fnum(entry)}  curr {fnum(current)}  upnl {fnum(upnl)}  stop {fnum(stop_dist)}",
                )
            else:
                safe_add(stdscr, y + 2, 3, f"{'TICKER':<6} {'DIR':<5} {'ENTRY':>8} {'CURR':>8} {'UPNL%':>7} {'STOP%':>7}", curses.A_UNDERLINE)
                safe_add(
                    stdscr,
                    y + 3 + i,
                    3,
                    f"{ticker:<6} "
                    f"{direction:<5} "
                    f"{fnum(entry):>8} "
                    f"{fnum(current):>8} "
                    f"{fnum(upnl):>7} "
                    f"{fnum(stop_dist):>7}",
                )
                safe_add(
                    stdscr,
                    y + 4 + i,
                    3,
                    f"stop {fnum(stop)}   target {fnum(pick(row, 'target_price'))}   asof {str(pick(price_row, 'asof'))[:19]}",
                    curses.A_DIM,
                )
    else:
        safe_add(stdscr, y + 3, 3, "No open trades available.", curses.A_DIM)

    footer_y = 24

    if mode == "minimal":
        y = 24
        draw_box(stdscr, y, 1, full_w, 3, "RECENT ACTIVITY")
        if recent:
            row = recent[-1]
            safe_add(
                stdscr,
                y + 1,
                3,
                f"last trade {str(pick(row, 'entry_date'))[:10]} -> {str(pick(row, 'exit_date'))[:10]}  "
                f"{str(pick(row, 'side', 'direction'))}  pnl {str(pick(row, 'pnl'))}",
            )
        else:
            safe_add(stdscr, y + 1, 3, "No recent activity.", curses.A_DIM)
        footer_y = y + 4
    else:
        y = 24
        draw_box(stdscr, y, 1, full_w, 5, "RECENT ACTIVITY")
        safe_add(stdscr, y + 1, 3, f"{'ENTRY':<12} {'EXIT':<12} {'SIDE':<6} {'PNL':>8} {'RET%':>8} {'BARS':>6}", curses.A_UNDERLINE)

        if recent:
            for i, row in enumerate(recent[:2]):
                safe_add(
                    stdscr,
                    y + 2 + i,
                    3,
                    f"{str(pick(row, 'entry_date')):<12} "
                    f"{str(pick(row, 'exit_date')):<12} "
                    f"{str(pick(row, 'side', 'direction')):<6} "
                    f"{str(pick(row, 'pnl')):>8} "
                    f"{str(pick(row, 'return_pct')):>8} "
                    f"{str(pick(row, 'bars', 'duration_bars')):>6}",
                )
        else:
            safe_add(stdscr, y + 2, 3, "No trades available.", curses.A_DIM)

        footer_y = y + 6

    safe_add(stdscr, min(h - 1, footer_y), 1, "[q] quit", curses.A_DIM)
    stdscr.refresh()


def main(stdscr) -> None:
    curses.curs_set(0)

    if args.mode == "snapshot":
        stdscr.nodelay(False)
        draw(stdscr, args.mode)
        while True:
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                return
    else:
        stdscr.nodelay(True)
        while True:
            draw(stdscr, args.mode)
            for _ in range(REFRESH_SECONDS * 10):
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    return
                time.sleep(0.1)


if __name__ == "__main__":
    args = parse_args()
    curses.wrapper(main)
