#!/usr/bin/env python3
"""
Polymarket Weather Paper Bot
- Scans top ~100 temperature markets daily at 10:00 AM ET
- Dutches the top 3 highest-priced "Yes" outcomes (flat shares across legs,
  so payout is fixed if any one hits) ONLY when the top-3 sum falls in a
  band [MIN_TOP3_SUM, MAX_TOP3_SUM) that guarantees positive ROI on a hit
- Position size scales with margin: bigger stakes near the floor (more
  margin), smaller stakes near the ceiling (thin margin)
- Optionally widens with a couple of cheap extra legs to cut "total miss"
  risk (the true outcome landing outside every bought bucket)
- Tracks resolution → daily P&L + win rate
- Starting balance $3000, realized P&L is added to balance
- Designed for Railway

Why the band, not just a floor:
Backtesting one month of paper trades under the old floor-only rule (buy
whenever top3 >= 80%, flat 10 shares) showed:
  - Every group with top3 >= 100% LOST money on every single trade, despite
    hitting the correct bucket 100% of the time -- cost (top3% * shares)
    exceeded the fixed payout (shares) by construction. Guaranteed loss.
  - The two biggest single losses of the month were "total misses" (true
    outcome outside all 3 bought buckets), not thin-margin trades -- one of
    them had a top3 of 99.5%, i.e. the market itself thought a miss was
    almost impossible, and it still happened.
  - The lowest top3 buckets tested (80-90%) were the only clearly profitable
    ones on a per-group basis.
This version removes the guaranteed-loss trades with a hard ceiling, sizes
positions by how much margin a group actually offers, and widens cheaply
to blunt the "total miss" tail risk that hurt more than thin margins did.
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import requests
import pytz
import schedule

# ====================== CONFIG ======================
GAMMA_API = "https://gamma-api.polymarket.com"

TOP_OUTCOMES_PER_MARKET = 3
MAX_MARKETS = 100
MIN_YES_PRICE = 0.05
MAX_YES_PRICE = 0.95
POLL_RESOLVED_EVERY_MIN = 30
STARTING_BALANCE = 3000.0

# ---- Dutching entry band ----
# Since we buy flat shares across the top-N legs, payout on a hit is fixed
# (= shares) and cost = shares * top_sum. Margin = shares - cost, so the
# entry band directly controls guaranteed ROI-if-hit: ROI = 1/top_sum - 1.
MIN_TOP3_SUM = 0.75                  # floor: skip low-coverage groups (miss risk)
MAX_TOP3_SUM = 0.90                  # ceiling: skip thin/negative-margin groups
                                      # (0.90 -> ~11% guaranteed ROI on a hit;
                                      #  anything >= 1.00 was a GUARANTEED LOSS
                                      #  in backtesting, every single time)

# ---- Margin-scaled position sizing ----
# Instead of a flat share count for every trade, size up when the group has
# more margin (lower top_sum, cheap insurance) and size down as top_sum
# approaches the ceiling (thin margin, not worth as much capital).
BASE_SHARES = 10                     # size used at/near MIN_TOP3_SUM
MIN_SHARES = 2                       # floor size used near MAX_TOP3_SUM

# ---- Optional widening to reduce "total miss" risk ----
# Backtesting showed the biggest single losses weren't thin-margin trades --
# they were total misses where the true outcome fell outside the bought
# legs entirely (e.g. Munich @ 99.5% top3, still missed all 3 buckets).
# Adding cheap extra legs (outside the core top-3) barely dents margin but
# reduces that tail risk.
ENABLE_WIDENING = True
WIDEN_MAX_EXTRA_LEGS = 2             # up to N extra legs beyond the top 3
WIDEN_MAX_PRICE = 0.10               # only widen with legs this cheap or cheaper

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
POSITIONS_FILE = DATA_DIR / "positions.json"
HISTORY_FILE = DATA_DIR / "daily_history.json"
STATE_FILE = DATA_DIR / "state.json"
LOG_FILE = DATA_DIR / "bot.log"

ET = pytz.timezone("America/New_York")

# ====================== LOGGING ======================
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("weather-paperbot")

# ====================== STORAGE ======================
def load_json(path: Path, default):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load {path}: {e}")
    return default

def save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_state() -> dict:
    default = {
        "starting_balance": STARTING_BALANCE,
        "current_balance": STARTING_BALANCE,
        "total_realized_pnl": 0.0,
        "all_time_wins": 0,
        "all_time_losses": 0,
        "all_time_even": 0,
        "all_time_trades": 0,
    }
    state = load_json(STATE_FILE, default)
    if "current_balance" not in state:
        state = default
    return state

def save_state(state: dict):
    save_json(STATE_FILE, state)

# ====================== API HELPERS ======================
def safe_get(url: str, params: dict = None, retries: int = 3) -> Optional[Any]:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"GET {url} failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None

def is_temperature_market(text: str) -> bool:
    """Strict filter: only real temperature markets."""
    q = (text or "").lower()
    if any(bad in q for bad in ("earthquake", "rhine", "hurricane", "tornado", "flood")):
        return False
    return any(k in q for k in (
        "temperature", "highest temperature", "lowest temperature",
        "°c", "°f", "degrees"
    ))

def fetch_weather_events(limit: int = 100) -> List[dict]:
    """Fetch active weather-tagged events and keep only temperature ones."""
    events = []
    offset = 0
    page_size = 50

    while len(events) < limit:
        data = safe_get(
            f"{GAMMA_API}/events",
            params={
                "tag_slug": "weather",
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        if not data:
            data = safe_get(
                f"{GAMMA_API}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )
            if data:
                data = [e for e in data if is_temperature_market(e.get("title", ""))]

        if not data:
            break

        data = [e for e in data if is_temperature_market(e.get("title", ""))]
        events.extend(data)

        if len(data) < page_size:
            break
        offset += page_size

    return events[:limit]

def parse_outcomes(market: dict) -> List[dict]:
    """Return list of Yes-side outcomes with price + token_id."""
    try:
        outcomes = json.loads(market.get("outcomes") or "[]")
        prices = json.loads(market.get("outcomePrices") or "[]")
        token_ids = json.loads(market.get("clobTokenIds") or "[]")
    except Exception:
        return []

    results = []
    for i, name in enumerate(outcomes):
        if i >= len(prices) or i >= len(token_ids):
            continue
        if name.strip().lower() in ("yes", "y"):
            try:
                price = float(prices[i])
            except Exception:
                continue
            if MIN_YES_PRICE <= price <= MAX_YES_PRICE:
                results.append({
                    "outcome": name,
                    "price": price,
                    "token_id": token_ids[i],
                    "market_id": market.get("id"),
                    "condition_id": market.get("conditionId"),
                    "question": market.get("question") or market.get("groupItemTitle") or "",
                    "end_date": market.get("endDate"),
                })
    return results

def get_market_status(market_id: str) -> Optional[dict]:
    return safe_get(f"{GAMMA_API}/markets/{market_id}")

# ====================== PAPER TRADING LOGIC ======================
def shares_for_group(top_sum: float) -> int:
    """
    Margin-scaled position size: BASE_SHARES near the floor (most margin),
    tapering linearly down to MIN_SHARES near the ceiling (least margin).
    """
    span = MAX_TOP3_SUM - MIN_TOP3_SUM
    if span <= 0:
        return BASE_SHARES
    frac = (MAX_TOP3_SUM - top_sum) / span   # 1.0 at floor, 0.0 at ceiling
    frac = max(0.0, min(1.0, frac))
    shares = MIN_SHARES + frac * (BASE_SHARES - MIN_SHARES)
    return max(1, round(shares))

def paper_buy(positions: List[dict], event: dict, market: dict, outcome: dict, shares: int):
    cost = outcome["price"] * shares
    pos = {
        "id": f"{market.get('id')}_{outcome['token_id']}_{int(time.time())}",
        "event_title": event.get("title"),
        "market_id": market.get("id"),
        "condition_id": market.get("conditionId"),
        "question": outcome["question"],
        "token_id": outcome["token_id"],
        "outcome": outcome["outcome"],
        "entry_price": outcome["price"],
        "shares": shares,
        "cost": round(cost, 4),
        "entry_time": datetime.now(ET).isoformat(),
        "end_date": outcome.get("end_date"),
        "status": "open",
        "exit_price": None,
        "pnl": None,
        "resolved_at": None,
    }
    positions.append(pos)
    log.info(
        f"PAPER BUY | {outcome['question'][:70]}... | "
        f"Yes @ {outcome['price']:.3f} × {shares} = ${cost:.2f}"
    )
    return pos

def check_resolutions(positions: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Returns (all_positions, newly_resolved)"""
    still_open = []
    newly_resolved = []

    for pos in positions:
        if pos["status"] != "open":
            still_open.append(pos)
            continue

        m = get_market_status(pos["market_id"])
        if not m:
            still_open.append(pos)
            continue

        closed = m.get("closed") or False
        try:
            prices = json.loads(m.get("outcomePrices") or "[]")
            token_ids = json.loads(m.get("clobTokenIds") or "[]")
        except Exception:
            still_open.append(pos)
            continue

        if not closed:
            still_open.append(pos)
            continue

        try:
            idx = token_ids.index(pos["token_id"])
            final_price = float(prices[idx])
        except (ValueError, IndexError, TypeError):
            final_price = 1.0 if any(float(p) > 0.99 for p in prices) else 0.0

        pnl = (final_price - pos["entry_price"]) * pos["shares"]
        pos["status"] = "resolved"
        pos["exit_price"] = final_price
        pos["pnl"] = round(pnl, 4)
        pos["resolved_at"] = datetime.now(ET).isoformat()
        newly_resolved.append(pos)

        result = "WIN " if pnl > 0 else "LOSS" if pnl < 0 else "EVEN"
        log.info(
            f"RESOLVED {result} | {pos['question'][:55]}... | "
            f"{pos['entry_price']:.3f} → {final_price:.3f} | PnL ${pnl:+.2f}"
        )

    return still_open + newly_resolved, newly_resolved

def update_state_with_resolved(newly_resolved: List[dict]):
    if not newly_resolved:
        return
    state = load_state()
    for pos in newly_resolved:
        pnl = pos.get("pnl") or 0.0
        state["total_realized_pnl"] = round(state["total_realized_pnl"] + pnl, 4)
        state["current_balance"] = round(state["starting_balance"] + state["total_realized_pnl"], 2)
        state["all_time_trades"] += 1
        if pnl > 0:
            state["all_time_wins"] += 1
        elif pnl < 0:
            state["all_time_losses"] += 1
        else:
            state["all_time_even"] += 1
    save_state(state)

# ====================== DAILY JOB ======================
def daily_scan_and_trade():
    log.info("=" * 60)
    log.info("DAILY SCAN STARTED (10:00 AM ET)")

    positions = load_json(POSITIONS_FILE, [])
    history = load_json(HISTORY_FILE, [])

    # 1. Resolve open positions + update balance
    positions, newly_resolved = check_resolutions(positions)
    update_state_with_resolved(newly_resolved)

    # 2. Fetch temperature events
    events = fetch_weather_events(limit=MAX_MARKETS)
    log.info(f"Fetched {len(events)} temperature events")

    bought_today = 0
    skipped_low_concentration = 0
    skipped_high_concentration = 0
    widened_legs_today = 0

    for event in events:
        markets = event.get("markets") or []
        candidates = []

        for m in markets:
            if m.get("closed") or not m.get("active", True):
                continue
            if not is_temperature_market(m.get("question") or m.get("groupItemTitle") or ""):
                continue
            for o in parse_outcomes(m):
                candidates.append((event, m, o))

        if len(candidates) < TOP_OUTCOMES_PER_MARKET:
            continue

        candidates.sort(key=lambda x: x[2]["price"], reverse=True)
        top = candidates[:TOP_OUTCOMES_PER_MARKET]

        top3_sum = sum(o["price"] for _, _, o in top)

        # Floor: not enough market coverage / conviction
        if top3_sum < MIN_TOP3_SUM:
            skipped_low_concentration += 1
            log.info(f"SKIP (top3={top3_sum:.1%} < {MIN_TOP3_SUM:.0%}) | {event.get('title', '')[:60]}...")
            continue

        # Ceiling: not enough margin left to guarantee profit on a hit.
        # (Backtesting: every group with top3 >= 100% lost money on EVERY
        # trade despite hitting the correct bucket 100% of the time -- cost
        # exceeded the fixed payout by construction. This ceiling exists to
        # keep guaranteed positive ROI-if-hit.)
        if top3_sum >= MAX_TOP3_SUM:
            skipped_high_concentration += 1
            log.info(f"SKIP (top3={top3_sum:.1%} >= {MAX_TOP3_SUM:.0%}, no margin) | {event.get('title', '')[:60]}...")
            continue

        shares = shares_for_group(top3_sum)
        implied_roi = (1.0 / top3_sum) - 1.0
        log.info(
            f"BUY GROUP (top3={top3_sum:.1%}, shares={shares}, "
            f"implied ROI if hit={implied_roi:.1%}) | {event.get('title', '')[:60]}..."
        )

        legs = list(top)

        # Widening: tack on a couple of cheap extra legs beyond the top-3 to
        # cut "total miss" risk (the true outcome landing outside all bought
        # buckets), which was the single biggest source of loss in backtesting.
        if ENABLE_WIDENING and WIDEN_MAX_EXTRA_LEGS > 0:
            extras = [
                c for c in candidates[TOP_OUTCOMES_PER_MARKET:]
                if c[2]["price"] <= WIDEN_MAX_PRICE
            ][:WIDEN_MAX_EXTRA_LEGS]
            if extras:
                legs = legs + extras

        for event, market, outcome in legs:
            already = any(
                p["token_id"] == outcome["token_id"] and p["status"] == "open"
                for p in positions
            )
            if already:
                continue
            is_widen_leg = outcome not in [o for _, _, o in top]
            leg_shares = shares  # keep payout aligned across the core top-3;
                                  # widened legs use the same share count so a
                                  # widen-leg hit still pays the group's target
            paper_buy(positions, event, market, outcome, leg_shares)
            bought_today += 1
            if is_widen_leg:
                widened_legs_today += 1

    # 3. Daily report
    state = load_state()
    today_str = datetime.now(ET).strftime("%Y-%m-%d")

    resolved_today = [
        p for p in positions
        if (p.get("resolved_at") or "").startswith(today_str)
    ]

    wins_today = [p for p in resolved_today if (p.get("pnl") or 0) > 0]
    losses_today = [p for p in resolved_today if (p.get("pnl") or 0) < 0]
    even_today = [p for p in resolved_today if (p.get("pnl") or 0) == 0]
    pnl_today = sum(p.get("pnl") or 0 for p in resolved_today)
    win_pct_today = (len(wins_today) / len(resolved_today) * 100) if resolved_today else 0.0

    open_count = sum(1 for p in positions if p["status"] == "open")
    open_cost = sum(p["cost"] for p in positions if p["status"] == "open")

    at_wins = state["all_time_wins"]
    at_losses = state["all_time_losses"]
    at_even = state["all_time_even"]
    at_total = state["all_time_trades"]
    at_win_pct = (at_wins / at_total * 100) if at_total > 0 else 0.0

    log.info("-" * 55)
    log.info(f"DAILY REPORT — {today_str}")
    log.info("-" * 55)
    log.info(f"New paper buys this run      : {bought_today}")
    log.info(f"  of which widened legs      : {widened_legs_today}")
    log.info(f"Skipped (top3 < {MIN_TOP3_SUM:.0%})         : {skipped_low_concentration}")
    log.info(f"Skipped (top3 >= {MAX_TOP3_SUM:.0%}, no margin): {skipped_high_concentration}")
    log.info("")
    log.info("RESOLVED TODAY")
    log.info(f"  Total resolved             : {len(resolved_today)}")
    log.info(f"  Won                        : {len(wins_today)}")
    log.info(f"  Lost                       : {len(losses_today)}")
    log.info(f"  Even                       : {len(even_today)}")
    log.info(f"  Win rate today             : {win_pct_today:.1f}%")
    log.info(f"  Profit / Loss today        : ${pnl_today:+.2f}")
    log.info("")
    log.info("ACCOUNT")
    log.info(f"  Starting balance           : ${state['starting_balance']:,.2f}")
    log.info(f"  Current balance            : ${state['current_balance']:,.2f}")
    log.info(f"  All-time realized P&L      : ${state['total_realized_pnl']:+.2f}")
    log.info("")
    log.info("ALL-TIME PERFORMANCE")
    log.info(f"  Total trades resolved      : {at_total}")
    log.info(f"  Won / Lost / Even          : {at_wins} / {at_losses} / {at_even}")
    log.info(f"  All-time win rate          : {at_win_pct:.1f}%")
    log.info("")
    log.info(f"Open positions remaining     : {open_count}")
    log.info(f"Capital still in open trades : ${open_cost:.2f}")
    log.info("=" * 55)

    # Save everything
    daily_record = {
        "date": today_str,
        "bought": bought_today,
        "widened_legs": widened_legs_today,
        "skipped_low_concentration": skipped_low_concentration,
        "skipped_high_concentration": skipped_high_concentration,
        "resolved": len(resolved_today),
        "wins": len(wins_today),
        "losses": len(losses_today),
        "even": len(even_today),
        "win_pct": round(win_pct_today, 1),
        "pnl_today": round(pnl_today, 2),
        "current_balance": state["current_balance"],
        "all_time_pnl": state["total_realized_pnl"],
        "all_time_win_pct": round(at_win_pct, 1),
        "open_positions": open_count,
        "open_capital": round(open_cost, 2),
    }
    history.append(daily_record)
    history = history[-90:]

    save_json(POSITIONS_FILE, positions)
    save_json(HISTORY_FILE, history)

def check_open_positions_job():
    """Lightweight job that checks resolutions every 30 min."""
    positions = load_json(POSITIONS_FILE, [])
    before = sum(1 for p in positions if p["status"] == "open")
    positions, newly_resolved = check_resolutions(positions)

    if newly_resolved:
        update_state_with_resolved(newly_resolved)
        save_json(POSITIONS_FILE, positions)
        state = load_state()
        log.info(
            f"Resolution check: {len(newly_resolved)} positions closed | "
            f"Balance now ${state['current_balance']:,.2f}"
        )

# ====================== SCHEDULER ======================
def run_scheduler():
    schedule.every().day.at("10:00").do(daily_scan_and_trade).tag("daily")
    schedule.every(POLL_RESOLVED_EVERY_MIN).minutes.do(check_open_positions_job)

    log.info("Scheduler started. Waiting for 10:00 AM ET ...")
    log.info(f"Data directory: {DATA_DIR}")
    log.info(f"Starting balance: ${STARTING_BALANCE:,.2f}")
    log.info(
        f"Rule: buy groups where top-3 Yes prices sum is in "
        f"[{MIN_TOP3_SUM:.0%}, {MAX_TOP3_SUM:.0%}) -- dutching for guaranteed "
        f"margin on a hit, sized {MIN_SHARES}-{BASE_SHARES} shares by margin"
    )
    if ENABLE_WIDENING:
        log.info(
            f"Widening: up to {WIDEN_MAX_EXTRA_LEGS} extra legs priced "
            f"≤ {WIDEN_MAX_PRICE:.2f} to reduce total-miss risk"
        )

    if os.getenv("RUN_ON_START", "false").lower() == "true":
        log.info("RUN_ON_START=true → executing daily job now")
        daily_scan_and_trade()

    while True:
        schedule.run_pending()
        time.sleep(20)

# ====================== ENTRY ======================
if __name__ == "__main__":
    log.info("Polymarket Weather Paper Bot starting...")
    run_scheduler()