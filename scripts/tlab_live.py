from __future__ import annotations

import csv
import curses
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "reports"
LIVE_DATA_DIR = ROOT / "data"

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


def build_alerts(metrics: dict[str, Any], trades: list[dict[str, Any]], open_trades: list[dict[str, Any]]) -> list[str]:
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
            entry = float(pick(row, "entry_price", default="nan"))
            stop = float(pick(row, "stop_price", default="nan"))
            direction = str(pick(row, "direction", default="n/a")).lower()
            ticker = str(pick(row, "ticker", default="n/a"))
            if entry > 0:
                if direction == "long":
                    dist = (entry - stop) / entry
                elif direction == "short":
                    dist = (stop - entry) / entry
                else:
                    dist = 9.0
                if dist <= 0.02:
                    alerts.append(f"Stop near: {ticker} {dist*100:.1f}%")
        except Exception:
            pass
    if not alerts:
        alerts.append("No active warnings")
    return alerts[:MAX_ALERTS]


def draw(stdscr) -> None:
    metrics = first_row(load_csv(DATA_DIR / "metrics.csv"))
    trades = load_csv(LIVE_DATA_DIR / "backtests" / "trade_log.csv")
    open_trades = load_csv(LIVE_DATA_DIR / "trades" / "open_trades.csv")
    macro = last_row(load_csv(DATA_DIR / "macro_snapshot.csv"))
    alerts = build_alerts(metrics, trades, open_trades)
    recent = trades[-MAX_RECENT_TRADES:]
    open_recent = open_trades[:2]

    open_count = len(open_trades)
    long_count = sum(1 for r in open_trades if str(pick(r, "direction", default="")).lower() == "long")
    short_count = sum(1 for r in open_trades if str(pick(r, "direction", default="")).lower() == "short")

    stdscr.erase()
    h, w = stdscr.getmaxyx()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    safe_add(stdscr, 0, 1, "TLAB LIVE HUD", curses.A_BOLD)
    safe_add(stdscr, 1, 1, f"{now}   refresh {REFRESH_SECONDS}s", curses.A_DIM)
    hline(stdscr, 2, 1, max(10, w - 2))

    full_w = max(40, w - 2)
    left_w = max(24, (w - 3) // 2)
    right_w = max(24, w - left_w - 3)

    y = 3
    draw_box(stdscr, y, 1, full_w, 3, "SYSTEM")
    safe_add(stdscr, y + 1, 3, f"mode monitor   status snapshot   heartbeat {now}")

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
    draw_box(stdscr, y, 1, full_w, 5, "OPEN TRADES")
    safe_add(stdscr, y + 1, 3, f"open {open_count}   long {long_count}   short {short_count}")
    safe_add(stdscr, y + 2, 3, f"{'TICKER':<8} {'DIR':<6} {'ENTRY':>8} {'STOP':>8} {'TARGET':>8} {'DATE':<12}", curses.A_UNDERLINE)

    if open_recent:
        for i, row in enumerate(open_recent[:1]):
            safe_add(
                stdscr,
                y + 3 + i,
                3,
                f"{str(pick(row, 'ticker')):<8} "
                f"{str(pick(row, 'direction')):<6} "
                f"{fnum(pick(row, 'entry_price')):>8} "
                f"{fnum(pick(row, 'stop_price')):>8} "
                f"{fnum(pick(row, 'target_price')):>8} "
                f"{str(pick(row, 'entry_date')):<12}",
            )
    else:
        safe_add(stdscr, y + 3, 3, "No open trades available.", curses.A_DIM)

    y = 23
    draw_box(stdscr, y, 1, full_w, 5, "TRADE LOG")
    safe_add(stdscr, y + 1, 3, f"{'ENTRY':<12} {'EXIT':<12} {'SIDE':<6} {'PNL':>8} {'RET%':>8} {'BARS':>6}", curses.A_UNDERLINE)

    if recent:
        for i, row in enumerate(recent[:2]):
            safe_add(
                stdscr,
                y + 2 + i,
                3,
                f"{str(pick(row, 'entry_date')):<12} "
                f"{str(pick(row, 'exit_date')):<12} "
                f"{str(pick(row, 'direction')):<6} "
                f"{str(pick(row, 'pnl')):>8} "
                f"{str(pick(row, 'return_pct')):>8} "
                f"{str(pick(row, 'duration_bars')):>6}",
            )
    else:
        safe_add(stdscr, y + 2, 3, "No trades available.", curses.A_DIM)

    safe_add(stdscr, min(h - 1, y + 6), 1, "[q] quit", curses.A_DIM)
    stdscr.refresh()


def main(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)

    while True:
        draw(stdscr)
        for _ in range(REFRESH_SECONDS * 10):
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                return
            time.sleep(0.1)


if __name__ == "__main__":
    curses.wrapper(main)
