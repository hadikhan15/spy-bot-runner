"""
SPY Signal Bot - learns from market data, suggests trades, never places them.

  python spy_bot.py backtest   Walk-forward test of the model on years of history.
                               Read the report before trusting anything.
  python spy_bot.py alert      Daily run (about 9:45 AM New York): retrain on all data,
                               log today's prediction,
                               paper-trade any signal, check your open position, and
                               (if the signal is strong and you've approved the model)
                               send one SPY debit spread that fits your risk rules.
  python spy_bot.py test-notify  Send a test notification.

You review every suggestion and place any order yourself in Robinhood.
"""

import copy
import subprocess
import time
import re
import datetime as dt
from contextlib import contextmanager
import json
import math
import os
import smtplib
import sys
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
for _s in (sys.stdout, sys.stderr):                  # Windows consoles/log files: never crash on "→" or "≥"
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
# Where the bot is running: "github" (the public runner), "pc" (the home server,
# see server/) or "local". GitHub and the PC both keep the repo in sync during the
# day; while the PC checks in, GitHub's scheduled runs stand by so nothing trades twice.
IN_ACTIONS = bool(os.environ.get("GITHUB_ACTIONS"))
HOST = os.environ.get("SPYBOT_HOST") or ("github" if IN_ACTIONS else "local")
SYNC = IN_ACTIONS or HOST == "pc"
HEARTBEAT_REF = "pc-heartbeat"                      # branch the PC server re-points every 5 minutes
PC_FRESH_SEC = 15 * 60                              # PC counts as running if it checked in this recently
CONFIG_PATH = HERE / "config.json"
CFG = json.loads(CONFIG_PATH.read_text())
RULES, ACCT, MODEL, TRADE = CFG["rules"], CFG["account"], CFG["model"], CFG["trade"]
OUT = HERE / "reports"
OUT.mkdir(exist_ok=True)
RATE = 0.04  # risk-free rate used for option pricing estimates

FEATURES = ["ret1", "ret5", "ret10", "ret20", "dist_sma20", "dist_sma50", "dist_sma200",
            "rv20", "vix_prev", "vix_chg5", "vix_minus_rv", "rsi14", "gap", "vix_gap",
            "is_fomc", "is_cpi", "is_jobs"]
EVENT_NAMES = {"fomc": "Fed decision (2 PM)", "cpi": "CPI inflation report (8:30 AM)",
               "jobs": "jobs report (8:30 AM)", "election": "US election day"}
# Longer-term picture (months, not days): 3/6/12-month returns, how the 50- and
# 200-day averages are sloping, distance from the 52-week high/low, and how
# steady the last 3 months' trend has been. All use closes through yesterday.
TREND_FEATURES = ["ret60", "ret120", "ret250", "sma50_slope", "sma200_slope",
                  "d_hi250", "d_lo250", "trend60", "trend60_fit", "above200_share"]
# US elections (known years in advance): election week, and the midterm-year cycle.
ELECTION_FEATURES = ["is_election", "elec_week", "midterm_year"]
# Volatility regime (research: intraday momentum is strongest when volatility is high
# and in stress; option buyers do better when options are cheap vs realized moves).
# ts_9d = VIX9D/VIX (above 1 = short-term stress), ts_3m = VIX/VIX3M (above 1 =
# backwardation), rv5 = last week's realized volatility, iv_cheap = rv5 / VIX9D.
REGIME_DAILY = ["ts_9d", "ts_3m", "rv5", "iv_cheap"]
REGIME_NEUTRAL = {"ts_9d": 1.0, "ts_3m": 0.9, "rv5": 0.15, "iv_cheap": 1.0}
DAILY_CONTEXT = FEATURES + TREND_FEATURES + ELECTION_FEATURES + REGIME_DAILY


def load_events():
    path = HERE / "events.json"
    if not path.exists():
        return {k: set() for k in EVENT_NAMES}
    ev = json.loads(path.read_text())
    return {k: {dt.date.fromisoformat(d) for d in ev.get(k, [])} for k in EVENT_NAMES}


EVENTS = load_events()


def events_on(day):
    return [k for k in EVENT_NAMES if day in EVENTS[k]]


# =================================================================== data

def load_history():
    """Daily SPY and VIX open/close. Today's row (if the market is open) has
    today's real open and the latest price as 'close'."""
    start = MODEL["history_start"]
    spy = yf.Ticker("SPY").history(start=start, auto_adjust=False)
    vix = yf.Ticker("^VIX").history(start=start, auto_adjust=False)
    if spy.empty or vix.empty:
        raise RuntimeError("Could not download SPY/VIX history from Yahoo Finance.")
    spy.index = spy.index.tz_localize(None).normalize()
    vix.index = vix.index.tz_localize(None).normalize()
    raw = pd.DataFrame({"open": spy["Open"], "high": spy["High"], "low": spy["Low"],
                        "close": spy["Close"],
                        "vix_open": vix["Open"], "vix": vix["Close"]}).dropna(subset=["open", "close", "vix"])
    bad = ~(raw["vix_open"] > 0)                      # old VIX opens can be missing or 0
    raw.loc[bad, "vix_open"] = raw["vix"].shift(1)[bad]
    for sym, name in (("^VIX9D", "vix9d"), ("^VIX3M", "vix3m")):   # optional: regime inputs
        try:
            h = yf.Ticker(sym).history(start=start, auto_adjust=False)
            if not h.empty:
                h.index = h.index.tz_localize(None).normalize()
                raw[name] = h["Close"].reindex(raw.index)
        except Exception as e:
            print(f"{sym} unavailable ({e}); using neutral values", file=sys.stderr)
    return raw


def election_day(year):
    """US general election: the Tuesday after the first Monday in November."""
    d = dt.date(year, 11, 1)
    while d.weekday() != 0:
        d += dt.timedelta(days=1)
    return d + dt.timedelta(days=1)


def election_week(day):
    """1 in the 7 days before a US general election (even years) through 3 days after."""
    if day.year % 2:
        return 0.0
    gap = (day - election_day(day.year)).days
    return 1.0 if -7 <= gap <= 3 else 0.0


def build_features(raw):
    """Each row = what is known at ~9:45 AM that day: every close up to YESTERDAY,
    plus TODAY's opening move (overnight gap in SPY and VIX).
    Target = SPY closes higher than today's open after `horizon_days` trading days."""
    df = raw.copy()
    c, v = df["close"], df["vix"]
    rets = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20):
        f[f"ret{n}"] = c.pct_change(n)
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["rv20"] = rets.rolling(20).std() * math.sqrt(252)
    f["vix_prev"] = v
    f["vix_chg5"] = v.pct_change(5)
    f["vix_minus_rv"] = v / 100 - f["rv20"]
    gain = rets.clip(lower=0).rolling(14).mean()
    loss = (-rets.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # the longer-term trend (months)
    for n in (60, 120, 250):
        f[f"ret{n}"] = c.pct_change(n)
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["sma50_slope"] = sma50 / sma50.shift(20) - 1
    f["sma200_slope"] = sma200 / sma200.shift(20) - 1
    f["d_hi250"] = c / df["high"].rolling(250).max() - 1
    f["d_lo250"] = c / df["low"].rolling(250).min() - 1
    # trend line through the last 60 closes: slope (per year) and how well it fits (R²)
    y = np.log(c)
    t = pd.Series(np.arange(len(c), dtype=float), index=c.index)
    cov = y.rolling(60).cov(t)
    slope = cov / t.rolling(60).var()
    f["trend60"] = slope * 252
    f["trend60_fit"] = (cov ** 2 / (t.rolling(60).var() * y.rolling(60).var())).clip(0, 1)
    f["above200_share"] = (c > sma200).astype(float).where(sma200.notna()).rolling(60).mean()
    # volatility regime (closes through yesterday, like everything above)
    v9 = df["vix9d"] if "vix9d" in df else pd.Series(np.nan, index=df.index)
    v3 = df["vix3m"] if "vix3m" in df else pd.Series(np.nan, index=df.index)
    f["ts_9d"] = (v9 / v).clip(0.5, 2.0)
    f["ts_3m"] = (v / v3).clip(0.5, 2.0)
    f["rv5"] = rets.rolling(5).std() * math.sqrt(252)
    f["iv_cheap"] = (f["rv5"] / (v9.fillna(v) / 100)).clip(0, 5)
    f = f.shift(1)                                   # only closes through yesterday
    f["gap"] = df["open"] / c.shift(1) - 1           # today's opening move
    f["vix_gap"] = df["vix_open"] / v.shift(1) - 1
    days = [d.date() for d in df.index]
    for k in EVENT_NAMES:                              # scheduled events are known in advance
        f[f"is_{k}"] = [1.0 if d in EVENTS[k] else 0.0 for d in days]
    f["elec_week"] = [election_week(d) for d in days]
    f["midterm_year"] = [1.0 if d.year % 4 == 2 else 0.0 for d in days]
    for col in f.columns:
        df[col] = f[col]
    h = MODEL["horizon_days"]
    fwd = c.shift(-(h - 1)) / df["open"] - 1          # from today's open to close h days later
    df["target"] = np.where(fwd.isna(), np.nan, (fwd > 0).astype(float))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    for k_, v_ in REGIME_NEUTRAL.items():             # before VIX9D existed (2011): neutral
        df[k_] = df[k_].fillna(v_)
    return df


def make_model(model_type=None):
    if (model_type or MODEL["model_type"]) == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                              max_iter=200, l2_regularization=1.0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))


# ========================================================== option math

def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fees():
    """Per round trip (buy + sell), in dollars. Robinhood charges no commission but
    passes on small regulatory fees; your real fills showed about $0.09-0.10."""
    return float(TRADE.get("fees_per_trade", 0.10))


def iv_from_vix(vix, ts9=None):
    """Option volatility used for simulated prices. VIX is a 30-day measure; our
    ~7-day options track VIX9D much better, so by default VIX is scaled by
    yesterday's VIX9D/VIX ratio (`model.iv_source` = "vix9d"). Then times a scale
    the monthly pricing check calibrates from real recorded SPY option quotes."""
    base = float(vix)
    try:
        if MODEL.get("iv_source", "vix9d") == "vix9d" and ts9 is not None and np.isfinite(ts9) and ts9 > 0:
            base *= float(ts9)
    except TypeError:
        pass
    return max(base / 100 * float(MODEL.get("iv_scale", 1.0)), 0.05)


def bs_price(kind, s, k, days, iv):
    if days <= 0:
        return max(s - k, 0.0) if kind == "call" else max(k - s, 0.0)
    t = days / 365
    iv = max(iv, 0.05)
    d1 = (math.log(s / k) + (RATE + iv * iv / 2) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    if kind == "call":
        return s * _ncdf(d1) - k * math.exp(-RATE * t) * _ncdf(d2)
    return k * math.exp(-RATE * t) * _ncdf(-d2) - s * _ncdf(-d1)


def spread_value(kind, s, k_long, k_short, days, iv):
    return max(bs_price(kind, s, k_long, days, iv) - bs_price(kind, s, k_short, days, iv), 0.0)


def max_debit_allowed():
    """Largest debit per share that satisfies every account rule."""
    by_loss = RULES["max_loss"]
    by_size = RULES["max_position_pct"] * ACCT["value"]
    by_cash = ACCT["cash"] - RULES["cash_buffer"]
    return max(min(by_loss, by_size, by_cash), 0) / 100


def is_single():
    """True = buy a single call/put; False = $1-wide debit spread."""
    return TRADE.get("structure", "spread") == "single"


def max_premium_allowed():
    """Largest single-option price (per share) the account rules allow."""
    by_size = RULES["max_position_pct"] * ACCT["value"]
    by_cash = ACCT["cash"] - RULES["cash_buffer"]
    return max(min(TRADE["max_premium"], by_size, by_cash), 0) / 100


def exit_levels(entry, peak, single):
    """Current stop price after the trailing ladder, whether it has moved up,
    and the next ladder step (price that triggers it, new stop) if any."""
    if single:
        stop = max(entry - TRADE["stop_loss_dollars"] / 100, 0.0)
    else:
        stop = entry * (1 - TRADE["stop_loss_pct"])
    first_stop = stop
    peak_gain = peak / entry - 1
    nxt = None
    for trigger, lock in TRADE.get("trail_ladder", []):
        if peak_gain >= trigger:
            stop = max(stop, entry * (1 + lock))
        elif nxt is None:
            nxt = (entry * (1 + trigger), entry * (1 + lock))
    return stop, stop > first_stop, nxt


def exit_reason(entry, value, peak, single, today, time_exit):
    stop, trailed, _ = exit_levels(entry, peak, single)
    tp = TRADE.get("profit_target_pct")
    if tp and value >= entry * (1 + tp):
        return "target"
    if value <= stop:
        return "trail stop" if trailed else "stop"
    if today >= time_exit:
        return "time"
    return None


def ladder_lines(entry):
    out = []
    for trigger, lock in TRADE.get("trail_ladder", []):
        new = "breakeven" if lock == 0 else f"+{lock:.0%}"
        out.append(f"  When it's worth ${entry * (1 + trigger):.2f} (+{trigger:.0%}): "
                   f"move your stop up to ${entry * (1 + lock):.2f} ({new})")
    return out


EXIT_PROFILES = {
    # sell fast: breakeven stop at +15%, lock +10% at +25%, take profit at +30%
    "quick": {"profit_target_pct": 0.30, "trail_ladder": [[0.15, 0.0], [0.25, 0.10]]},
    # middle: breakeven at +25%, lock +20% at +40%, take profit at +60%
    "balanced": {"profit_target_pct": 0.60, "trail_ladder": [[0.25, 0.0], [0.40, 0.20]]},
    # let winners run: no fixed target, ladder only
    "runner": {"profit_target_pct": None, "trail_ladder": [[0.5, 0.0], [1.0, 0.5], [2.0, 1.0]]},
}


def apply_exit_profile(name=None):
    name = name or TRADE.get("exit_profile", "custom")
    if name in EXIT_PROFILES:
        TRADE.update(EXIT_PROFILES[name])


apply_exit_profile()


def is_day_trade():
    return bool(TRADE.get("day_trade", False))


def horizon_text():
    h = MODEL["horizon_days"]
    return "by today's close" if h == 1 else f"in {h} trading days"


def describe(kind, long, short):
    if short is None or (isinstance(short, float) and math.isnan(short)):
        return f"{float(long):g} {kind}"
    return f"{float(long):g}/{float(short):g} {kind} spread"


# ============================================================== backtest

# ============================================================ intraday model
# Used for signals after the morning: at each 30-minute check until the last
# entry time, "will SPY close today above where it is right now?"
INTRA_BASE = FEATURES + ["intr_ret", "intr_range", "hours_in", "vix_intr"]
INTRA_FEATURES = INTRA_BASE + ["or_pos", "prev_hi_dist", "prev_lo_dist", "vwap_dist", "vol_ratio"]
# Market structure: support/resistance, fair value gaps, liquidity sweeps
STRUCT_FEATURES = ["d_hi20", "d_lo20", "d_hi60", "d_lo60", "d_round", "sr_touches",
                   "d_fvg_below", "d_fvg_above", "in_fvg", "sweep"]


def daily_fvgs(prior):
    """Fair value gaps in the prior daily bars (last 30): a 3-bar gap where
    bar 1's high < bar 3's low (bullish) or bar 1's low > bar 3's high (bearish).
    Returns lists of [bottom, top, lowest low since, highest high since]."""
    h, l = prior["high"].values[-30:], prior["low"].values[-30:]
    bull, bear = [], []
    for j in range(2, len(h)):
        after_lo = l[j + 1:].min() if j + 1 < len(l) else np.inf
        after_hi = h[j + 1:].max() if j + 1 < len(h) else -np.inf
        if h[j - 2] < l[j]:
            bull.append([h[j - 2], l[j], after_lo])
        if l[j - 2] > h[j]:
            bear.append([h[j], l[j - 2], after_hi])
    return bull, bear


def structure_features(prior, fvgs, now, day_hi, day_lo):
    """Where price sits relative to support/resistance, the nearest unfilled
    fair value gaps, and whether a liquidity sweep (stop hunt) just happened."""
    cap = 0.05
    hi20, lo20 = prior["high"].tail(20).max(), prior["low"].tail(20).min()
    hi60, lo60 = prior["high"].tail(60).max(), prior["low"].tail(60).min()
    lvl = [hi20, lo20, hi60, lo60, prior["high"].iloc[-1], prior["low"].iloc[-1]]
    # how many of the last 60 days' highs/lows sit within 0.25% of now (a crowded level)
    pts = np.concatenate([prior["high"].tail(60).values, prior["low"].tail(60).values])
    touches = float(np.sum(np.abs(pts / now - 1) < 0.0025))
    bull, bear = fvgs
    below = [top for bot, top, lo_since in bull if min(lo_since, day_lo) > bot and top <= now]
    above = [bot for bot, top, hi_since in bear if max(hi_since, day_hi) < top and bot >= now]
    in_bull = any(min(lo_since, day_lo) > bot and bot < now < top for bot, top, lo_since in bull)
    in_bear = any(max(hi_since, day_hi) < top and bot < now < top for bot, top, hi_since in bear)
    in_fvg = 1.0 if in_bull and not in_bear else -1.0 if in_bear and not in_bull else 0.0
    y_hi, y_lo = prior["high"].iloc[-1], prior["low"].iloc[-1]
    sweep = 0.0
    if day_lo < y_lo and now > y_lo and not (day_hi > y_hi):
        sweep = 1.0      # took out yesterday's low, then reclaimed it (bullish)
    elif day_hi > y_hi and now < y_hi and not (day_lo < y_lo):
        sweep = -1.0     # took out yesterday's high, then lost it (bearish)
    return {"d_hi20": now / hi20 - 1, "d_lo20": now / lo20 - 1,
            "d_hi60": now / hi60 - 1, "d_lo60": now / lo60 - 1,
            "d_round": (now - round(now / 5) * 5) / now, "sr_touches": touches,
            "d_fvg_below": min(now / max(below) - 1, cap) if below else cap,
            "d_fvg_above": min(min(above) / now - 1, cap) if above else cap,
            "in_fvg": in_fvg, "sweep": sweep}

# "Smart money" concepts: market structure (swing highs/lows, break of structure,
# change of character), order blocks, and sweeps of the opening range and the
# 20-day high/low. Daily pieces use only bars BEFORE today; intraday pieces use
# only the hourly bars that had finished at that moment.
SMC_FEATURES = ["ms_trend", "bos", "choch", "d_swing_hi", "d_swing_lo", "d_bull_ob", "d_bear_ob",
                "in_ob", "sweep_or", "sweep_20", "ms_intraday"]


def smc_daily(prior, look=60, ob_look=30, disp=0.007):
    """Once per day, from the prior daily bars: confirmed swing points (a high/low
    with 2 lower highs / higher lows on each side), the structure they form, and
    the order blocks still untouched. Bullish order block = the last down day
    before a strong up move (a close at least `disp` above that day's high within
    3 days); bearish = the last up day before a strong down move."""
    pr = prior.tail(look)
    h, l = pr["high"].values, pr["low"].values
    o, c = pr["open"].values, pr["close"].values
    sh = [h[i] for i in range(2, len(h) - 2) if h[i] > max(h[i - 2], h[i - 1], h[i + 1], h[i + 2])]
    sl = [l[i] for i in range(2, len(l) - 2) if l[i] < min(l[i - 2], l[i - 1], l[i + 1], l[i + 2])]
    trend = 0
    if len(sh) >= 2 and len(sl) >= 2:
        if sh[-1] > sh[-2] and sl[-1] > sl[-2]:
            trend = 1                   # higher highs and higher lows
        elif sh[-1] < sh[-2] and sl[-1] < sl[-2]:
            trend = -1                  # lower highs and lower lows
    bull, bear = [], []
    n = len(h)
    for j in range(max(0, n - ob_look), n - 1):
        nxt = c[j + 1:j + 4]
        if c[j] < o[j] and len(nxt) and nxt.max() > h[j] * (1 + disp):
            later_lo = l[j + 2:].min() if j + 2 < n else np.inf
            if later_lo > h[j]:         # price hasn't come back into it yet: still fresh
                bull.append((l[j], h[j]))
        if c[j] > o[j] and len(nxt) and nxt.min() < l[j] * (1 - disp):
            later_hi = h[j + 2:].max() if j + 2 < n else -np.inf
            if later_hi < l[j]:
                bear.append((l[j], h[j]))
    return {"trend": trend, "sh": sh[-1] if sh else h.max(), "sl": sl[-1] if sl else l.min(),
            "bull": bull, "bear": bear,
            "hi20": float(prior["high"].tail(20).max()), "lo20": float(prior["low"].tail(20).min())}


def smc_features(d, now, day_hi, day_lo, bars, cap=0.05):
    """What the smart-money map says at this moment. `bars` = today's finished
    hourly bars (Open/High/Low/Close)."""
    sh, sl, tr = d["sh"], d["sl"], d["trend"]
    bos = 1.0 if tr >= 0 and now > sh else -1.0 if tr <= 0 and now < sl else 0.0
    choch = 1.0 if tr == -1 and now > sh else -1.0 if tr == 1 and now < sl else 0.0
    below = [top for bot, top in d["bull"] if top <= now and day_lo > bot]
    above = [bot for bot, top in d["bear"] if bot >= now and day_hi < top]
    in_ob = 0.0
    if any(day_lo <= top and now >= bot for bot, top in d["bull"]):
        in_ob = 1.0                     # tapped a fresh bullish order block and held it
    elif any(day_hi >= bot and now <= top for bot, top in d["bear"]):
        in_ob = -1.0
    sweep_or = 0.0
    if len(bars) >= 2:
        orh, orl = float(bars["High"].iloc[0]), float(bars["Low"].iloc[0])
        later = bars.iloc[1:]
        if later["Low"].min() < orl and now > orl and not later["High"].max() > orh:
            sweep_or = 1.0              # broke the first hour's low, then reclaimed it
        elif later["High"].max() > orh and now < orh and not later["Low"].min() < orl:
            sweep_or = -1.0
    sweep_20 = (1.0 if day_lo < d["lo20"] and now > d["lo20"]
                else -1.0 if day_hi > d["hi20"] and now < d["hi20"] else 0.0)
    ms_i = 0.0
    if len(bars) >= 3:
        hh = (bars["High"].diff().dropna() > 0).sum()
        ll = (bars["Low"].diff().dropna() < 0).sum()
        ms_i = float((hh - ll) / (len(bars) - 1))
    return {"ms_trend": float(tr), "bos": bos, "choch": choch,
            "d_swing_hi": float(np.clip(now / sh - 1, -cap, cap)),
            "d_swing_lo": float(np.clip(now / sl - 1, -cap, cap)),
            "d_bull_ob": min(now / max(below) - 1, cap) if below else cap,
            "d_bear_ob": min(min(above) / now - 1, cap) if above else cap,
            "in_ob": in_ob, "sweep_or": sweep_or, "sweep_20": sweep_20, "ms_intraday": ms_i}


# Intraday momentum (the best-replicated intraday effect: the day-so-far move
# predicts the rest of the day, mostly on volatile/stressed days).
MOMO_FEATURES = ["move_z", "noise_pos"]


# The committee: three models that each see the day a little differently and vote.
COMMITTEE = {
    "simple": ("logistic", INTRA_BASE),          # simple model, basic information
    "levels": ("logistic", INTRA_FEATURES),      # simple model + day-trader levels
    "flexible": ("gbm", INTRA_FEATURES),         # flexible model + day-trader levels
    "structure": ("logistic", INTRA_FEATURES + STRUCT_FEATURES),  # + support/resistance, FVGs, sweeps
    "trend": ("logistic", INTRA_FEATURES + TREND_FEATURES + ELECTION_FEATURES),  # + months-long trend, elections
    "smc": ("logistic", INTRA_FEATURES + ["sweep"] + SMC_FEATURES),  # + market structure, order blocks, sweeps
    "momentum": ("logistic", INTRA_BASE + MOMO_FEATURES + REGIME_DAILY + ["or_pos", "vwap_dist"]),  # momentum × regime
}


def committee_names():
    """Names of the voters in use. Can include "judge" (the decision-maker)."""
    choice = MODEL.get("intraday_model", "committee")
    if choice != "committee":
        return [choice]
    names = MODEL.get("committee_members") or list(COMMITTEE)
    return [n for n in names if n in COMMITTEE or n == "judge"]


def committee_members():
    """The base models that must be trained for the voters in use. The judge
    needs every base model's opinion, so it pulls in all of them."""
    names = committee_names()
    if "judge" in names:
        return dict(COMMITTEE)
    return {n: COMMITTEE[n] for n in names}


# ------------------------------------------------------------- the judge
# The decision-maker: learns from history WHICH models to trust in WHICH
# situations, using every model's opinion plus market structure and context.
JUDGE_CONTEXT = ["sweep", "in_fvg", "d_fvg_below", "d_fvg_above", "d_hi20", "d_lo20",
                 "sr_touches", "vwap_dist", "or_pos", "vix_now", "hours_in", "gap", "intr_ret"]


def judge_inputs(frame):
    J = pd.DataFrame(index=frame.index)
    for n in COMMITTEE:
        J[f"p_{n}"] = frame[f"p_{n}"] - 0.5
    for c in JUDGE_CONTEXT:
        J[c] = frame[c]
    J["vix_now"] = J["vix_now"] / 100
    # interactions: lets the judge learn "trust model X more when Y"
    for n in COMMITTEE:
        J[f"{n}_x_sweep"] = J[f"p_{n}"] * frame["sweep"]
        J[f"{n}_x_vix"] = J[f"p_{n}"] * J["vix_now"]
    J["structure_x_fvg"] = J["p_structure"] * frame["in_fvg"]
    return J


def make_judge():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))


def add_judge(X, min_days=60, step=21):
    """Walk-forward: the judge only learns from days where every model's opinion
    was itself out-of-sample, and is tested on the days after."""
    X = X.copy()
    X["p_judge"] = np.nan
    ready = X[[f"p_{n}" for n in COMMITTEE]].notna().all(axis=1)
    days = sorted(X.loc[ready, "day"].unique())
    for i in range(min_days, len(days), step):
        tr = X[ready & (X["day"] < days[i])]
        mask = ready & X["day"].isin(set(days[i:i + step]))
        if tr["target"].nunique() < 2 or not mask.any():
            continue
        jm = make_judge().fit(judge_inputs(tr), tr["target"])
        X.loc[mask, "p_judge"] = jm.predict_proba(judge_inputs(X.loc[mask]))[:, 1]
    return X


# ------------------------------------------------------------- helper bots
# Each can be switched on/off in config (model.bots). The self-tuner tests
# them and switches them on only when the evidence says they help.
BOT_DEFAULTS = {"risk": False, "mood": False, "exit": False}
BOT_SETTINGS_DEFAULT = {"risk_max_vix": 30, "risk_max_gap": 0.015, "risk_max_range": 0.02,
                        "exit_vix_split": 20}


def bots_on():
    return {**BOT_DEFAULTS, **MODEL.get("bots", {})}


def bot_cfg(key):
    return MODEL.get("bot_settings", {}).get(key, BOT_SETTINGS_DEFAULT[key])


def bot_gate(kind, row, bots=None):
    """Risk bot and mood bot can veto a signal. Returns the reason, or None."""
    bots = bots or bots_on()
    if bots["risk"]:
        if row["vix_now"] > bot_cfg("risk_max_vix"):
            return "risk bot: VIX too high"
        if abs(row["gap"]) > bot_cfg("risk_max_gap"):
            return "risk bot: opening gap too big"
        if row["intr_range"] > bot_cfg("risk_max_range"):
            return "risk bot: the day is already swinging too much"
    if bots["mood"]:
        if kind == "call" and row["vwap_dist"] <= 0:
            return "mood bot: SPY is below VWAP, not a call moment"
        if kind == "put" and row["vwap_dist"] >= 0:
            return "mood bot: SPY is above VWAP, not a put moment"
    return None


def exit_style_for(vix_now, bots=None):
    """Exit bot: runner when the market is calm, quick when it's wild."""
    bots = bots or bots_on()
    if bots["exit"]:
        return "runner" if vix_now < bot_cfg("exit_vix_split") else "quick"
    return TRADE.get("exit_profile", "runner")


@contextmanager
def using_exit(style):
    """Temporarily apply one exit style (for one trade)."""
    saved = {k: copy.deepcopy(TRADE.get(k)) for k in ("profit_target_pct", "trail_ladder")}
    if style in EXIT_PROFILES:
        TRADE.update(copy.deepcopy(EXIT_PROFILES[style]))
    try:
        yield
    finally:
        TRADE.update(saved)


def save_config():
    CONFIG_PATH.write_text(json.dumps(CFG, indent=2))


def log_change(text):
    """Add a row to CHANGELOG.md (newest first)."""
    path = HERE / "CHANGELOG.md"
    if not path.exists():
        return
    s_ = path.read_text(encoding="utf-8")
    marker = "|---|---|\n"
    row = f"| {dt.date.today()} | [automatic] {text} |\n"
    path.write_text(s_.replace(marker, marker + row, 1) if marker in s_ else s_ + row, encoding="utf-8")


def committee_vote(probs, base):
    """Average the members' probabilities; 'agree' = all lean the same way."""
    p = float(np.mean(list(probs.values())))
    sides = {np.sign(v - base) for v in probs.values()}
    return p, len(sides) == 1


def load_hourly():
    """Hourly SPY/VIX bars for about the last 2 years (Yahoo's limit)."""
    spy = yf.Ticker("SPY").history(period="730d", interval="1h")
    if spy.empty:
        raise RuntimeError("Could not download hourly SPY data.")
    spy.index = spy.index.tz_convert("America/New_York")
    vix = yf.Ticker("^VIX").history(period="730d", interval="1h")
    if not vix.empty:
        vix.index = vix.index.tz_convert("America/New_York")
    return spy, vix


def intraday_extras(bars, now, prev_hi, prev_lo):
    """Day-trader levels known at this moment: first-hour range, yesterday's
    high/low, VWAP (average price paid today) and volume so far."""
    oh, ol = float(bars["High"].iloc[0]), float(bars["Low"].iloc[0])
    rng = oh - ol
    vol = bars["Volume"].astype(float) if "Volume" in bars else pd.Series(0.0, index=bars.index)
    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    vwap = float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else now
    return {"or_pos": float(np.clip((now - ol) / rng, -2, 3)) if rng > 0 else 0.5,
            "prev_hi_dist": now / prev_hi - 1, "prev_lo_dist": now / prev_lo - 1,
            "vwap_dist": now / vwap - 1, "cumvol": float(vol.sum())}


def intraday_features(ctx, day_open, hi, lo, now, hours_in, vix_open, vix_now, extras=None):
    row = {f: float(ctx[f]) for f in DAILY_CONTEXT if f in ctx.index}
    row.update(intr_ret=now / day_open - 1, intr_range=(hi - lo) / day_open, hours_in=hours_in,
               vix_intr=(vix_now / vix_open - 1) if vix_open and vix_open > 0 else 0.0)
    # day-so-far move in units of the move VIX expects for one day (intraday momentum)
    day_sigma = max(float(vix_now), 5.0) / 100 / math.sqrt(252)
    row["move_z"] = float(np.clip((now / day_open - 1) / day_sigma, -6, 6))
    if extras:
        row.update(extras)
    return row


def build_intraday(daily, spy_h, vix_h):
    """One row per (day, hourly checkpoint): known info at that moment, and
    whether SPY closed that day above the price at that moment."""
    dmap = {d.date(): i for i, d in enumerate(daily.index)}
    vix_days = {d: g for d, g in vix_h.groupby(vix_h.index.date)} if not vix_h.empty else {}
    rows, day_bars = [], {}
    last_entry = dt.time.fromisoformat(TRADE.get("last_entry_time", "14:30"))
    for day, g in spy_h.groupby(spy_h.index.date):
        if day not in dmap or dmap[day] < 1 or len(g) < 6:     # skip half days
            continue
        day_bars[day] = g
        ctx = daily.iloc[dmap[day]]
        yday = daily.iloc[dmap[day] - 1]
        prior = daily.iloc[max(0, dmap[day] - 60):dmap[day]]
        fvgs = daily_fvgs(prior)
        smc_d = smc_daily(prior)
        o, close = float(g["Open"].iloc[0]), float(g["Close"].iloc[-1])
        vg = vix_days.get(day)
        v_open = float(vg["Open"].iloc[0]) if vg is not None and len(vg) else float(ctx["vix_open"])
        for k in range(1, len(g)):
            ts = g.index[k]
            if ts.time() > last_entry:
                break
            now = float(g["Open"].iloc[k])
            v_now = v_open
            if vg is not None:
                prev = vg[vg.index < ts]
                if len(prev):
                    v_now = float(prev["Close"].iloc[-1])
            row = intraday_features(ctx, o, float(g["High"].iloc[:k].max()), float(g["Low"].iloc[:k].min()),
                                    now, (ts.hour * 60 + ts.minute - 570) / 60, v_open, v_now,
                                    intraday_extras(g.iloc[:k], now, float(yday["high"]), float(yday["low"])))
            row.update(structure_features(prior, fvgs, now, float(g["High"].iloc[:k].max()),
                                          float(g["Low"].iloc[:k].min())))
            row.update(smc_features(smc_d, now, float(g["High"].iloc[:k].max()),
                                    float(g["Low"].iloc[:k].min()), g.iloc[:k]))
            row.update(day=day, time=ts.strftime("%H:%M"), k=k, price=now, vix_now=v_now,
                       target=float(close > now))
            rows.append(row)
    X = pd.DataFrame(rows)
    if X.empty:
        return X, day_bars
    X = X.sort_values(["day", "k"]).reset_index(drop=True)
    # volume so far vs the average of the previous 20 days at the same time of day
    X["vol_ratio"] = X.groupby("k")["cumvol"].transform(
        lambda v: v / v.shift(1).rolling(20, min_periods=5).mean())
    X["vol_ratio"] = X["vol_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 5)
    # "noise area": today's move vs the average size of the move by this time of day
    # over the previous 14 days (Zarattini/Aziz/Barbon); beyond ±1 = a real breakout
    X["abs_move"] = X["intr_ret"].abs()
    X["noise_ref"] = X.groupby("k")["abs_move"].transform(lambda v: v.shift(1).rolling(14, min_periods=5).mean())
    X["noise_pos"] = (X["intr_ret"] / X["noise_ref"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(subset=INTRA_FEATURES + STRUCT_FEATURES)
    for c_ in TREND_FEATURES + ELECTION_FEATURES + SMC_FEATURES + MOMO_FEATURES:   # never drop rows
        X[c_] = X[c_].fillna(0.0) if c_ in X else 0.0
    for c_, v_ in REGIME_NEUTRAL.items():
        X[c_] = X[c_].fillna(v_) if c_ in X else v_
    return X, day_bars


def walk_forward_intraday(X, min_days=120, step=21, members=None):
    """Retrain every `step` days on earlier days only. Every model gets its own
    column p_<name>, the judge gets p_judge; `p` is the vote of `members`."""
    members = members or committee_names()
    days = sorted(X["day"].unique())
    X = X.copy()
    for name in COMMITTEE:
        X[f"p_{name}"] = np.nan
    X["base"] = np.nan
    for i in range(min_days, len(days), step):
        test_days = set(days[i:i + step])
        train = X[X["day"] < days[i]]
        mask = X["day"].isin(test_days)
        for name, (mt, feats) in COMMITTEE.items():
            m = make_model(mt).fit(train[feats], train["target"])
            X.loc[mask, f"p_{name}"] = m.predict_proba(X.loc[mask, feats])[:, 1]
        X.loc[mask, "base"] = train["target"].mean()
    X = add_judge(X)
    cols = [f"p_{n}" for n in members]
    X["p"] = X[cols].mean(axis=1)
    b = X["base"].values[:, None]
    lean = np.sign(X[cols].values - b)
    X["agree"] = (lean == lean[:, :1]).all(axis=1)
    return X


def pick_contract(kind, s0, iv0):
    """Strike(s) the bot would buy at price s0, and a pricing function."""
    single, dte0 = is_single(), TRADE["target_dte"]
    slip = TRADE["slippage_per_leg"] * (1 if single else 2)
    base_k = math.floor(s0) if kind == "call" else math.ceil(s0)
    step = 1 if kind == "call" else -1
    k_long = k_short = None
    if single:
        cap, floor_ = max_premium_allowed() - slip, TRADE["min_premium"] / 100 - slip
        for off in range(0, 300):
            k = base_k + step * off
            if bs_price(kind, s0, k, dte0, iv0) <= cap:
                if bs_price(kind, s0, k, dte0, iv0) >= floor_:
                    k_long = k
                break
    else:
        w, cap = TRADE["spread_width"], max_debit_allowed() - slip
        for off in range(0, 60):
            k1 = base_k + step * off
            if spread_value(kind, s0, k1, k1 + step * w, dte0, iv0) <= cap:
                k_long, k_short = k1, k1 + step * w
                break
    if k_long is None:
        return None

    def value(s, iv, hours=0.0):
        days = max(dte0 - hours / 24, 0.01)          # time decay as the day goes on
        v = bs_price(kind, s, k_long, days, iv) if single else \
            spread_value(kind, s, k_long, k_short, days, iv)
        return max(v - slip, 0.0)
    return k_long, value(s0, iv0) + 2 * slip, value


def sim_bars(kind, s0, iv0, bars):
    """Trade entered at price s0, then walked through hourly bars
    (high, low, close), with time decay. Within a bar the WORST price is
    assumed to come first."""
    c = pick_contract(kind, s0, iv0)
    if not c:
        return None
    k, entry, value = c
    single = is_single()
    tp = TRADE.get("profit_target_pct")
    peak = entry
    for j, (hi, lo, cl) in enumerate(bars):
        hrs = j + 1
        worst, best = (lo, hi) if kind == "call" else (hi, lo)
        stop, trailed, _ = exit_levels(entry, peak, single)
        if value(worst, iv0, hrs) <= stop:
            return k, entry, stop, "trail stop" if trailed else "stop"
        peak = max(peak, value(best, iv0, hrs))
        if tp and peak >= entry * (1 + tp):
            return k, entry, entry * (1 + tp), "target"
        stop, trailed, _ = exit_levels(entry, peak, single)
        if value(cl, iv0, hrs) <= stop:
            return k, entry, stop, "trail stop" if trailed else "stop"
    return k, entry, value(bars[-1][2], iv0, len(bars)), "close"


def run_intraday(Xt, day_bars, decide, bots=None):
    """First (un-vetoed) signal of each day -> one simulated trade.
    decide(row) -> 'call'/'put'/None. Helper bots can veto and pick the exit style."""
    bots = bots or bots_on()
    trades = []
    skip = set(TRADE.get("skip_events", []))
    for day, g in Xt.groupby("day"):
        if skip & set(events_on(day)):
            continue
        for _, r in g.sort_values("k").iterrows():
            kind = decide(r)
            if not kind or bot_gate(kind, r, bots):
                continue
            style = exit_style_for(r["vix_now"], bots)
            bars = day_bars[day].iloc[int(r["k"]):]
            with using_exit(style):
                res = sim_bars(kind, r["price"], iv_from_vix(r["vix_now"], r.get("ts_9d")),
                               list(zip(bars["High"], bars["Low"], bars["Close"])))
            if res:
                k, entry, exit_v, why = res
                trades.append({"entry_date": day, "time": r["time"], "kind": kind, "exit_style": style,
                               "edge": r["p"] - r["base"] if "p" in r else 0.0,
                               "entry": round(entry, 3), "exit": round(exit_v, 3),
                               "pnl": round((exit_v - entry) * 100 - fees(), 2), "reason": why})
            break
    return pd.DataFrame(trades)


# ------------------------------------------- the EV filter (meta-labeling)
# The direction models predict where SPY closes. What we actually earn is the
# OPTION trade after stops, the ladder, time decay and fees. A second, small model
# learns from the bot's own past signals which ones turned into winning option
# trades, then trades only when expected value (chance of a win × average win −
# chance of a loss × average loss) is positive. Idea: López de Prado's
# meta-labeling; the primary model picks the side, the meta model decides to act.
META_THR = 0.04                 # loose primary bar: let the EV filter do the picking
META_FEATURES = ["m_edge", "m_align", "m_hours_left", "m_vix", "m_ivcheap", "m_ts9", "m_or", "m_vwap"]
_OUTCOMES = {}


def option_outcomes(Xt, day_bars, bots=None):
    """Simulated option P&L (after fees) of buying a call and of buying a put at
    every checkpoint, with the exit style the trade would really get (the exit bot
    can switch it by VIX). Cached per setup."""
    bots = bots_on() if bots is None else bots
    key = (len(Xt), str(Xt["day"].min()), str(Xt["day"].max()), TRADE.get("exit_profile"),
           json.dumps([TRADE.get("trail_ladder"), TRADE.get("profit_target_pct")], default=str),
           tuple(sorted(bots.items())),
           MODEL.get("iv_scale", 1.0), MODEL.get("iv_source", "vix9d"))
    if key in _OUTCOMES:
        return _OUTCOMES[key]
    pc, pp = [], []
    for _, r in Xt.iterrows():
        bars = day_bars[r["day"]].iloc[int(r["k"]):]
        hlc = list(zip(bars["High"], bars["Low"], bars["Close"]))
        iv = iv_from_vix(r["vix_now"], r.get("ts_9d"))
        out = []
        with using_exit(exit_style_for(r["vix_now"], bots)):
            for kind in ("call", "put"):
                res = sim_bars(kind, r["price"], iv, hlc)
                out.append((res[2] - res[1]) * 100 - fees() if res else np.nan)
        pc.append(out[0])
        pp.append(out[1])
    res = pd.DataFrame({"pnl_call": pc, "pnl_put": pp}, index=Xt.index)
    if len(_OUTCOMES) >= 4:
        _OUTCOMES.pop(next(iter(_OUTCOMES)))
    _OUTCOMES[key] = res
    return res


def meta_frame(R, edge):
    """The few things that decide whether a direction call pays as an option trade."""
    sgn = np.sign(edge)
    return pd.DataFrame({
        "m_edge": np.abs(edge),
        "m_align": sgn * R["move_z"],                     # is the day already moving our way?
        "m_hours_left": 6.5 - R["hours_in"],              # time left for the move (and decay)
        "m_vix": R["vix_now"] / 100,
        "m_ivcheap": R["iv_cheap"],                       # options cheap vs recent real moves?
        "m_ts9": R["ts_9d"],                              # short-term stress
        "m_or": sgn * (R["or_pos"] - 0.5),                # where in the first-hour range, our way
        "m_vwap": sgn * R["vwap_dist"] * 100,             # above/below VWAP, our way
    }, index=R.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_meta():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def meta_walk_forward(Xp, outcomes, min_days=40, step=21):
    """Walk-forward EV estimates for rows with a primary signal (|edge| > META_THR),
    learning only from earlier days. Returns (meta_p, meta_ev) Series."""
    e = (Xp["p"] - Xp["base"]).values
    cand = (np.abs(e) > META_THR) & np.isfinite(e)
    pnl = np.where(e > 0, outcomes["pnl_call"].values, outcomes["pnl_put"].values)
    cand &= np.isfinite(pnl)
    M = meta_frame(Xp, e)
    y = (pnl > 0).astype(int)
    days = np.array(Xp["day"].values)
    uniq = sorted(set(days[cand]))
    mp = pd.Series(np.nan, index=Xp.index)
    mev = pd.Series(np.nan, index=Xp.index)
    for i in range(min_days, len(uniq), step):
        tr = cand & (days < uniq[i])
        te = cand & np.isin(days, uniq[i:i + step])
        if tr.sum() < 80 or len(set(y[tr])) < 2 or not te.any():
            continue
        cnt = pd.Series(days[tr]).map(pd.Series(days[tr]).value_counts()).values
        m = make_meta().fit(M[tr], y[tr], logisticregression__sample_weight=1.0 / cnt)
        pw = m.predict_proba(M[te])[:, 1]
        wins, losses = pnl[tr & (pnl > 0)], pnl[tr & (pnl <= 0)]
        aw = wins.mean() if len(wins) else 0.0
        al = -losses.mean() if len(losses) else 0.0
        mp[te] = pw
        mev[te] = pw * aw - (1 - pw) * al
    return mp, mev


def with_meta(Xt, day_bars, members, bots=None):
    """Xt with p for `members` plus walk-forward meta_p / meta_ev columns."""
    P = Xt[[f"p_{n}" for n in members]]
    X2 = Xt.assign(p=P.mean(axis=1))
    lean = np.sign(P.sub(Xt["base"], axis=0))                 # each member's side
    agree = lean.eq(np.sign(X2["p"] - Xt["base"]), axis=0).all(axis=1)
    mp, mev = meta_walk_forward(X2, option_outcomes(Xt, day_bars, bots))
    return X2.assign(meta_p=mp, meta_ev=mev, agree=agree)


def meta_decide(r, min_ev=None):
    e = r["p"] - r["base"]
    min_ev = MODEL.get("meta_min_ev", 0.0) if min_ev is None else min_ev
    if not (abs(e) > META_THR) or not (r.get("meta_ev", np.nan) > min_ev):
        return None
    if MODEL.get("require_agreement") and not r.get("agree", True):
        return None
    k_ = "call" if e > 0 else "put"
    return None if (k_ == "put" and not TRADE.get("allow_puts", True)) else k_


# ----------------------------------------------------- quant checks (skill or luck?)
FEATURE_GROUPS = {
    "today's move (simple)": INTRA_BASE,
    "day-trader levels": ["or_pos", "prev_hi_dist", "prev_lo_dist", "vwap_dist", "vol_ratio"],
    "support/resistance + FVG": STRUCT_FEATURES,
    "months-long trend + elections": TREND_FEATURES + ELECTION_FEATURES,
    "smart money (structure, order blocks, sweeps)": SMC_FEATURES,
    "intraday momentum + volatility regime": MOMO_FEATURES + REGIME_DAILY,
}


def risk_stats(trades, days):
    """Sharpe/Sortino on daily P&L (zero on days without a trade), drawdown, profit factor."""
    if trades.empty:
        return None
    pnl = trades.groupby(pd.to_datetime(trades["entry_date"]))["pnl"].sum()
    daily = pnl.reindex(pd.to_datetime(sorted(days)), fill_value=0.0)
    eq = daily.cumsum()
    dd = eq - eq.cummax()
    under = (dd < 0).astype(int)
    longest = int(under.groupby((under == 0).cumsum()).sum().max()) if len(under) else 0
    sd = daily.std()
    down = float(np.sqrt(np.mean(np.minimum(daily.values, 0.0) ** 2))) if len(daily) else 0.0   # downside deviation
    wins, losses = trades.loc[trades["pnl"] > 0, "pnl"], trades.loc[trades["pnl"] <= 0, "pnl"]
    return {"sharpe": float(daily.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
            "sortino": float(daily.mean() / down * math.sqrt(252)) if down and down > 0 else 0.0,
            "max_dd": float(dd.min()), "longest_dd": longest,
            "pf": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf"),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0, "drawdown": dd}


def permutation_test(Xt, day_bars, trades, n=2000, seed=7):
    """Monte Carlo permutation test. On the SAME days the model traded, how often
    would random trades (a random check time and a random call/put) have made at
    least as much? Also: keeping the model's entry times, how often does a random
    call/put choice do as well (tests the direction calls on their own)?"""
    if trades.empty or len(trades) < 20:
        return None
    rng = np.random.default_rng(seed)
    days = list(trades["entry_date"])
    table = {}                          # (day, k) -> (call pnl, put pnl)
    for day in set(days):
        g = Xt[Xt["day"] == day]
        for _, r in g.iterrows():
            bars = day_bars[day].iloc[int(r["k"]):]
            hlc = list(zip(bars["High"], bars["Low"], bars["Close"]))
            iv = iv_from_vix(r["vix_now"], r.get("ts_9d"))
            out = []
            for kind in ("call", "put"):
                res = sim_bars(kind, r["price"], iv, hlc)
                out.append((res[2] - res[1]) * 100 if res else 0.0)
            table[(day, int(r["k"]), r["time"])] = out
    actual = float(trades["pnl"].sum())
    by_day = {}
    for (day, k, t), v in table.items():
        by_day.setdefault(day, []).append((t, v))
    # 1) random time + random direction on the model's trading days
    choices = [by_day[d] for d in days]
    rand_tot = np.zeros(n)
    for opts in choices:
        pick = rng.integers(0, len(opts), n)
        side = rng.integers(0, 2, n)
        vals = np.array([o[1] for o in opts])            # shape (len(opts), 2)
        rand_tot += vals[pick, side]
    # 2) the model's own entry times, random direction
    flip = []
    for _, tr in trades.iterrows():
        v = dict(by_day[tr["entry_date"]]).get(tr["time"])
        flip.append(v if v else [tr["pnl"], tr["pnl"]])
    flip = np.array(flip)
    side = rng.integers(0, 2, (n, len(flip)))
    dir_tot = np.where(side == 0, flip[:, 0], flip[:, 1]).sum(axis=1)
    return {"actual": actual, "n": n,
            "p_random": float((np.sum(rand_tot >= actual) + 1) / (n + 1)),
            "rand_med": float(np.median(rand_tot)), "rand_95": float(np.percentile(rand_tot, 95)),
            "p_direction": float((np.sum(dir_tot >= actual) + 1) / (n + 1)),
            "dir_med": float(np.median(dir_tot))}


def feature_group_importance(X, repeats=5, seed=3):
    """Which kinds of information actually help? One model with every input is
    trained on the older 70% of days and scored on the newest 30%; each group of
    inputs is then scrambled in the test days. The bigger the drop in accuracy
    (AUC), the more that group matters. Near zero or negative = not helping."""
    feats = [f for f in dict.fromkeys(sum(FEATURE_GROUPS.values(), [])) if f in X.columns]
    days = sorted(X["day"].unique())
    cut = days[int(len(days) * 0.7)]
    tr, te = X[X["day"] < cut].dropna(subset=feats), X[X["day"] >= cut].dropna(subset=feats)
    if len(tr) < 300 or len(te) < 100 or te["target"].nunique() < 2:
        return None
    m = make_model("logistic").fit(tr[feats], tr["target"])
    base = roc_auc_score(te["target"], m.predict_proba(te[feats])[:, 1])
    rng = np.random.default_rng(seed)
    out = []
    for name, cols in FEATURE_GROUPS.items():
        cols = [c for c in cols if c in feats]
        drops = []
        for _ in range(repeats):
            sh = te[feats].copy()
            idx = rng.permutation(len(sh))
            sh[cols] = sh[cols].values[idx]          # scramble the group together
            drops.append(base - roc_auc_score(te["target"], m.predict_proba(sh)[:, 1]))
        out.append((name, float(np.mean(drops)), float(np.std(drops))))
    return base, cut, sorted(out, key=lambda x: -x[1])


def quant_report(Xt, day_bars, trades, cdir, tag, decide=None):
    L = ["", "## Skill or luck? (quant checks)", ""]
    rs = risk_stats(trades, Xt["day"].unique())
    if rs:
        L += ["| Measure | Value | What it means |", "|---|---|---|",
              f"| Sharpe ratio | {rs['sharpe']:.2f} | return per unit of day-to-day swings, yearly. Above 1 is good, above 2 is rare |",
              f"| Sortino ratio | {rs['sortino']:.2f} | like Sharpe, but only counts the down days as risk |",
              f"| Worst drawdown | ${rs['max_dd']:.0f} | biggest drop from a high point (1 contract at a time) |",
              f"| Longest drawdown | {rs['longest_dd']} trading days | longest stretch below a previous high |",
              f"| Profit factor | {rs['pf']:.2f} | dollars won per dollar lost. Above 1.3 is solid |",
              f"| Average win / loss | ${rs['avg_win']:.0f} / ${rs['avg_loss']:.0f} | |", ""]
        try:
            ok = safe_chart(chart_equity, {"Drawdown": rs["drawdown"]}, cdir / f"bt_drawdown_{tag}.png",
                            "Drawdown: how far below its last high", "Simulated later-in-day trades")
            if ok:
                L += [f"![Drawdown](charts/bt_drawdown_{tag}.png)", ""]
        except Exception:
            pass
    if decide is not None:
        late = Xt.copy()
        g_ = late.groupby("day")
        for c_ in ("p", "base", "meta_ev"):
            if c_ in late:
                late[c_] = g_[c_].shift(1)        # decide at one check, buy at the next
        lt = run_intraday(late, day_bars, decide)
        if len(lt):
            L += [f"**If you act an hour late** (signal at one check, buy at the next): "
                  f"{len(lt)} trades, ${lt['pnl'].sum():.0f} (vs ${trades['pnl'].sum():.0f} on time). "
                  f"{'The edge survives delays.' if lt['pnl'].sum() > 0 else 'The edge does NOT survive a delay: act fast or not at all.'}", ""]
    pt = permutation_test(Xt, day_bars, trades)
    if pt:
        verdict = ("**looks like skill**" if pt["p_random"] < 0.05 and pt["p_direction"] < 0.05 else
                   "**not clearly better than luck yet**")
        L += [f"**Monte Carlo permutation test** ({pt['n']:,} random runs). The model made **${pt['actual']:.0f}**.",
              f"- Random trades on the same days (random time, random call/put): typical ${pt['rand_med']:.0f}, "
              f"top 5% ${pt['rand_95']:.0f}. Chance random does as well: **{pt['p_random']:.1%}**.",
              f"- Same entry times, random call/put: typical ${pt['dir_med']:.0f}. "
              f"Chance random direction does as well: **{pt['p_direction']:.1%}**.",
              f"- Verdict: {verdict} (both chances under 5% = skill).", ""]
    fi = feature_group_importance(Xt)
    if fi:
        base, cut, rows = fi
        L += [f"**Which information helps?** One model with every input, trained before {cut}, scored after "
              f"(accuracy AUC {base:.3f}). Each group is scrambled to see how much accuracy it was carrying:", "",
              "| Information | Accuracy lost when scrambled | Helps? |", "|---|---|---|"]
        for name, d, sd in rows:
            helps = "yes" if d > max(0.003, 2 * sd) else "no" if d <= 0 else "maybe"
            L.append(f"| {name} | {d:+.4f} (±{sd:.4f}) | {helps} |")
        L += ["", "_Groups that don't help are candidates to drop; the self-tuner tests that with real P&L._", ""]
    return L


# ------------------------------------------- look-ahead self-test
def selftest(n_checks=40, seed=11):
    """The #1 way trading bots fool themselves is a feature that peeks at the future.
    Builds features on made-up prices twice: once with the full history, once cut
    off right after the moment being checked. Any difference = look-ahead."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", "2026-06-30")
    c = 300 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
    o = c * (1 + rng.normal(0, 0.004, len(idx)))
    raw = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996,
                        "close": c, "vix_open": 18 + rng.normal(0, 1, len(idx)),
                        "vix": 18 + rng.normal(0, 1, len(idx)),
                        "vix9d": 17 + rng.normal(0, 1, len(idx)), "vix3m": 20 + rng.normal(0, 1, len(idx))},
                       index=idx)
    rows = []
    for d in idx[-160:]:
        p_ = raw.loc[d, "open"]
        for h in range(7):
            ts = pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30 + 60 * h)
            n_ = p_ * (1 + rng.normal(0, 0.003))
            rows.append((ts, p_, max(p_, n_) * 1.001, min(p_, n_) * 0.999, n_, 1e6))
            p_ = n_
    spy_h = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"]).set_index("ts")
    vix_h = spy_h.copy()
    vix_h[["Open", "High", "Low", "Close"]] = 18.0
    df_full = build_features(raw)
    X_full, _ = build_intraday(df_full, spy_h, vix_h)
    feats = [f for f in dict.fromkeys(sum([f for _, f in COMMITTEE.values()], [])) if f in X_full]
    bad = set()
    days = sorted(X_full["day"].unique())
    for d in rng.choice(days[30:], size=min(n_checks, len(days) - 30), replace=False):
        sub = X_full[X_full["day"] == d]
        k = int(rng.choice(sub["k"].values))
        cut_ts = spy_h[spy_h.index.date == d].index[k]          # the moment being checked
        # drop every later day, and scramble everything not yet known at that moment
        raw_t = raw[raw.index <= pd.Timestamp(d)].copy()
        for col in ("high", "low", "close", "vix", "vix9d", "vix3m"):
            raw_t.loc[raw_t.index[-1], col] *= float(rng.uniform(0.9, 1.1))
        df_t = build_features(raw_t)
        sh = spy_h[spy_h.index.date <= d].copy()
        later = sh.index >= cut_ts                               # bar k and after haven't finished
        sh.loc[later, ["High", "Low", "Close"]] *= rng.uniform(0.95, 1.05, (int(later.sum()), 3))
        sh.loc[later & (sh.index > cut_ts), "Open"] *= rng.uniform(0.95, 1.05, int((later & (sh.index > cut_ts)).sum()))
        sh.loc[later, "Volume"] *= rng.uniform(0.5, 2.0, int(later.sum()))
        vh = vix_h[vix_h.index.date <= d].copy()
        vh.loc[vh.index >= cut_ts, ["High", "Low", "Close"]] *= 1.3
        Xt_, _ = build_intraday(df_t, sh, vh)
        a = X_full[(X_full["day"] == d) & (X_full["k"] == k)][feats]
        b = Xt_[(Xt_["day"] == d) & (Xt_["k"] == k)][feats] if len(Xt_) else a.iloc[0:0]
        if a.empty:
            continue
        if b.empty:
            bad.add("(row missing when cut off)")
            continue
        diff = (a.reset_index(drop=True) - b.reset_index(drop=True)).abs().iloc[0]
        bad |= set(diff[diff > 1e-9].index)
    return sorted(bad)


# ------------------------------------------------------------- self-tuner
def current_setup():
    return {"bots": bots_on(), "members": list(committee_names()),
            "exit": TRADE.get("exit_profile", "runner"),
            "thr": MODEL.get("base_edge_threshold", MODEL["edge_threshold"]),
            "meta": bool(MODEL.get("meta_filter", False))}


def candidate_setups(cur):
    """One change at a time, so we always know what made the difference."""
    cands = []
    for b in ("risk", "mood", "exit"):
        c = copy.deepcopy(cur)
        c["bots"][b] = not c["bots"][b]
        cands.append((f"{'turn ON' if c['bots'][b] else 'turn OFF'} the {b} bot", c))
    # bots working together: every pair, and all three at once
    for combo in (("risk", "mood"), ("risk", "exit"), ("mood", "exit"), ("risk", "mood", "exit")):
        c = copy.deepcopy(cur)
        for b in BOT_DEFAULTS:
            c["bots"][b] = b in combo
        if c["bots"] != cur["bots"]:
            cands.append((f"bots together: {' + '.join(combo)}", c))
    for ex in EXIT_PROFILES:
        if ex != cur["exit"]:
            c = copy.deepcopy(cur)
            c["exit"] = ex
            cands.append((f"exit style → {ex}", c))
    for mem in (["simple"], ["simple", "levels"], ["simple", "levels", "flexible"],
                ["simple", "levels", "structure"], ["simple", "structure"], ["structure"],
                ["judge"], ["judge", "simple"], ["judge", "simple", "levels"],
                ["simple", "levels", "trend"], ["simple", "trend"], ["levels", "trend"], ["trend"],
                ["simple", "levels", "structure", "trend"],
                ["smc"], ["simple", "smc"], ["simple", "levels", "smc"], ["simple", "levels", "flexible", "smc"],
                ["simple", "trend", "smc"], ["simple", "levels", "flexible", "trend"],
                ["momentum"], ["simple", "momentum"], ["simple", "levels", "flexible", "momentum"],
                ["momentum", "smc"], ["simple", "momentum", "smc"]):
        if sorted(mem) != sorted(cur["members"]):
            c = copy.deepcopy(cur)
            c["members"] = mem
            cands.append((f"committee → {' + '.join(mem)}", c))
    pick = review_pick()
    if pick and sorted(pick) != sorted(cur["members"]):
        c = copy.deepcopy(cur)
        c["members"] = pick
        cands.append((f"trade-review team → {' + '.join(pick)}", c))
    c = copy.deepcopy(cur)
    c["meta"] = not cur.get("meta", False)
    cands.append((f"{'turn ON' if c['meta'] else 'turn OFF'} the EV filter (trade only when expected value > 0)", c))
    for d in (-0.02, 0.02):
        t = round(cur["thr"] + d, 3)
        if 0.04 <= t <= 0.16:
            c = copy.deepcopy(cur)
            c["thr"] = t
            cands.append((f"confidence bar → {t:.0%}", c))
    return cands


def eval_setup(setup, Xt, day_bars):
    thr = setup["thr"]

    def dec(r):
        e = r["p"] - r["base"]
        k_ = "call" if e > thr else "put" if e < -thr else None
        return None if (k_ == "put" and not TRADE.get("allow_puts", True)) else k_

    saved = TRADE.get("exit_profile")
    TRADE["exit_profile"] = setup["exit"]
    try:
        if setup.get("meta"):
            X2 = with_meta(Xt, day_bars, setup["members"], setup["bots"])
            tr = run_intraday(X2, day_bars, meta_decide, bots=setup["bots"])
            ready = X2.loc[X2["meta_ev"].notna(), "day"]
            tr.attrs["start"] = ready.min() if len(ready) else None   # EV filter's first live day
            return tr
        X2 = Xt.assign(p=Xt[[f"p_{n}" for n in setup["members"]]].mean(axis=1))
        return run_intraday(X2, day_bars, dec, bots=setup["bots"])
    finally:
        TRADE["exit_profile"] = saved


def hac_t(x, lags=5):
    """t-statistic of the mean that allows for day-to-day correlation (Newey-West)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20:
        return 0.0
    d = x - x.mean()
    var = d @ d / n
    for l_ in range(1, lags + 1):
        var += 2 * (1 - l_ / (lags + 1)) * (d[l_:] @ d[:-l_]) / n
    return float(x.mean() / math.sqrt(var / n)) if var > 0 else 0.0


def pbo_cscv(M, S=10):
    """Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, Zhu):
    split the days into S blocks; for every half/half split, pick the best setup on
    one half and see where it ranks on the other. PBO = how often the in-sample
    winner lands in the bottom half out of sample. Near 0 = picking works; 0.5+ =
    picking the best backtest is no better than chance."""
    from itertools import combinations
    from scipy.stats import rankdata
    M = np.asarray(M, dtype=float)                   # days × setups, daily P&L
    M = np.unique(M, axis=1) if M.ndim == 2 and M.shape[1] else M   # identical setups count once
    T, N = M.shape
    if N < 4 or T < S * 5:
        return None
    blocks = np.array_split(np.arange(T), S)

    def sr(A):
        sd = A.std(axis=0, ddof=1)
        return np.where(sd > 0, A.mean(axis=0) / np.where(sd > 0, sd, 1), 0.0)
    lam = []
    for J in combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in J])
        te = np.setdiff1d(np.arange(T), tr)
        best = int(np.argmax(sr(M[tr])))
        w = rankdata(sr(M[te]))[best] / (N + 1)      # average rank, so ties don't flatter
        lam.append(math.log(w / (1 - w)))
    return float(np.mean(np.array(lam) <= 0))


def deflated_sharpe(daily, sr_trials, n_trials):
    """Deflated Sharpe Ratio (Bailey & López de Prado): the chance the chosen
    setup's Sharpe beats the best Sharpe you'd expect by luck after n_trials tries."""
    from scipy.stats import norm, skew, kurtosis
    r = np.asarray(daily, dtype=float)
    if len(r) < 30 or r.std(ddof=1) == 0 or n_trials < 2:
        return None
    sr_ = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    eg = 0.5772156649
    v = np.var(sr_trials, ddof=1) if len(sr_trials) > 1 else 0.0
    sr0 = math.sqrt(max(v, 0)) * ((1 - eg) * norm.ppf(1 - 1 / n_trials) + eg * norm.ppf(1 - 1 / (n_trials * math.e)))
    den = math.sqrt(max(1 - g3 * sr_ + (g4 - 1) / 4 * sr_ ** 2, 1e-9))
    return float(norm.cdf((sr_ - sr0) * math.sqrt(len(r) - 1) / den))


TUNE_MIN_T = 3.0   # self-tuner: minimum steadiness (Newey-West t-stat) of a change's daily edge
TUNE_MAX_PBO = 0.5  # self-tuner: refuse to adopt when picking-the-best looks like overfitting
LEDGER_PATH = HERE / "reports" / "tuner_trials.csv"


def self_tune(Xt, day_bars):
    """Test every one-step change against the current setup on the hourly history.
    Adopt the best one ONLY if (1) it adds a real amount of money, (2) it does better
    in BOTH the older and the newer half of the data, and (3) its day-by-day edge over
    the current setup is steady enough that luck is an unlikely explanation
    (t-statistic of the daily P&L differences >= TUNE_MIN_T). With ~25 candidates
    tested each month, (3) is what stops the bot from adopting noise."""
    days = sorted(Xt["day"].unique())
    mid = days[len(days) // 2]
    all_days = pd.Index(days)

    def score(tr, start=None):
        """Totals over the days from `start` on (an EV-filter setup can't trade before its
        filter has learned, so both sides are compared only on days both could trade)."""
        win = [d for d in days if start is None or d >= start]
        mid_w = win[len(win) // 2] if win else mid
        if not tr.empty and start is not None:
            tr = tr[pd.to_datetime(tr["entry_date"]).dt.date >= start]
        if tr.empty:
            return {"n": 0, "total": 0.0, "old": 0.0, "new": 0.0, "mid": mid_w, "start": start,
                    "daily": pd.Series(0.0, index=all_days)}
        d = pd.to_datetime(tr["entry_date"]).dt.date
        daily = tr.groupby(d.values)["pnl"].sum().reindex(all_days, fill_value=0.0)
        return {"n": len(tr), "total": tr["pnl"].sum(), "mid": mid_w, "start": start,
                "old": tr.loc[d < mid_w, "pnl"].sum(), "new": tr.loc[d >= mid_w, "pnl"].sum(),
                "daily": daily}

    def t_stat(a, b, start=None):
        diff = a["daily"] - b["daily"]
        return hac_t((diff[all_days >= start] if start is not None else diff).values)

    cur = current_setup()
    champ_tr = eval_setup(cur, Xt, day_bars)
    champ = score(champ_tr)
    margin = max(200.0, 0.15 * abs(champ["total"]))
    rows, winners, starts = [], [], [champ_tr.attrs.get("start")]
    for name, cand in candidate_setups(cur):
        try:
            tr = eval_setup(cand, Xt, day_bars)
        except Exception as e:
            print(f"Tuner skipped '{name}': {e}", file=sys.stderr)
            continue
        st = max([s for s in (champ_tr.attrs.get("start"), tr.attrs.get("start")) if s is not None], default=None)
        starts.append(tr.attrs.get("start"))
        sc, ch = score(tr, st), (score(champ_tr, st) if st is not None else champ)
        sc["t"] = t_stat(sc, ch, st)
        mg = max(200.0, 0.15 * abs(ch["total"]))
        ok = (sc["n"] >= 100 and sc["total"] >= ch["total"] + mg
              and sc["old"] > ch["old"] and sc["new"] > ch["new"]
              and sc["t"] >= TUNE_MIN_T)
        sc["vs"] = ch["total"]
        rows.append((name, sc, ok))
        if ok:
            winners.append((sc["total"], name, cand))
    # overfitting checks across everything tried this month, on days every setup could trade
    st_all = max([s for s in starts if s is not None], default=None)
    keep = (all_days >= st_all) if st_all is not None else np.ones(len(all_days), bool)
    mat = np.column_stack([champ["daily"].values[keep]] + [sc["daily"].values[keep] for _, sc, _ in rows])
    pbo = pbo_cscv(mat)
    srs = [d.mean() / d.std(ddof=1) for d in mat.T if d.std(ddof=1) > 0]
    try:
        old_n = len(pd.read_csv(LEDGER_PATH)) if LEDGER_PATH.exists() else 0
        pd.DataFrame([{"date": str(dt.date.today()), "change": n_, "trades": sc_["n"],
                       "total": round(sc_["total"], 2), "t": round(sc_["t"], 2)} for n_, sc_, _ in rows]
                     ).to_csv(LEDGER_PATH, mode="a", header=not LEDGER_PATH.exists(), index=False)
    except Exception:
        old_n = 0
    n_trials = old_n + len(rows) + 1
    best_name = max(winners)[1] if winners else (max(rows, key=lambda x: x[1]["total"])[0] if rows else None)
    best_daily = next((sc["daily"] for n_, sc, _ in rows if n_ == best_name), None)
    dsr = deflated_sharpe(best_daily.values, srs, n_trials) if best_daily is not None else None
    pbo_blocked = bool(winners) and pbo is not None and pbo >= TUNE_MAX_PBO
    if pbo_blocked:
        winners = []                                 # the "best" is likely luck: adopt nothing
    L = ["", "## Self-improvement check (one change at a time vs the current setup)", "",
         f"Current setup: committee {' + '.join(cur['members'])}, exit {cur['exit']}, confidence bar "
         f"{cur['thr']:.0%}, bots on: {', '.join(b for b, v in cur['bots'].items() if v) or 'none'}. "
         f"A change is adopted only if it adds at least ${margin:.0f}, does better in both halves "
         f"(before and after {mid}), AND its day-by-day edge is steady enough to rule out luck "
         f"(steadiness score ≥ {TUNE_MIN_T}, a Newey-West t-statistic; 3+ is strong), AND the month's "
         f"tests don't look overfit (PBO below {TUNE_MAX_PBO:.0%}).", "",
         f"**Overfitting checks:** probability of backtest overfitting (PBO) "
         f"**{'n/a' if pbo is None else f'{pbo:.0%}'}** across {len(rows) + 1} setups tested this month "
         f"(near 0% = picking the best works; 50%+ = it's luck). Deflated Sharpe of the best change"
         f"{f' ({best_name})' if best_name else ''}: **{'n/a' if dsr is None else f'{dsr:.0%}'}** after "
         f"{n_trials} setups tried in total (above 95% = its Sharpe beats what luck alone would produce).", "",
         "| Change | Trades | Total P&L | Older half | Newer half | Steadiness | Verdict |",
         "|---|---|---|---|---|---|---|",
         f"| **current setup** | {champ['n']} | ${champ['total']:.2f} | ${champ['old']:.2f} | ${champ['new']:.2f} | - | - |"]
    for name, sc, ok in rows:
        note = (f" (compared from {sc['start']} on, when the EV filter had learned enough; current setup "
                f"${sc['vs']:.2f} over the same days)" if sc.get("start") is not None else "")
        L.append(f"| {name}{note} | {sc['n']} | ${sc['total']:.2f} | ${sc['old']:.2f} | ${sc['new']:.2f} | "
                 f"{sc['t']:+.1f} | {'✅ better' if ok else 'no'} |")
    if winners and MODEL.get("auto_improve", True):
        _, name, cand = max(winners)
        champ = {**champ, "total": next(sc["vs"] for n_, sc, _ in rows if n_ == name)}
        MODEL["bots"] = cand["bots"]
        MODEL["meta_filter"] = bool(cand.get("meta", False))
        MODEL["committee_members"] = cand["members"]
        MODEL["base_edge_threshold"] = cand["thr"]
        MODEL["edge_threshold"] = cand["thr"]
        TRADE["exit_profile"] = cand["exit"]
        apply_exit_profile(cand["exit"])
        save_config()
        msg = (f"Self-tuner adopted: {name}. Backtest P&L ${champ['total']:.0f} → "
               f"${max(winners)[0]:.0f}, better in both halves of the data.")
        log_change(msg)
        notify("SPY bot improved itself", msg + " Details in the latest backtest report.", important=True)
        L += ["", f"**Adopted: {name}.** config.json and CHANGELOG.md were updated."]
    elif winners:
        L += ["", "A better setup was found but `model.auto_improve` is off, so nothing was changed."]
    else:
        L += ["", ("A change passed the money and steadiness tests, but the overfitting check (PBO) says picking "
                   "winners this month is unreliable, so nothing was changed." if pbo_blocked else
                   "No change cleared the bar. The current setup stays.")]
    return L


def confidence_table(edge, correct, pnl_by_edge=None):
    """Does higher model confidence lead to better results?"""
    buckets = [(0, .02, "under 2%"), (.02, .05, "2-5%"), (.05, .08, "5-8%"), (.08, 1, "8% or more")]
    L = ["| How far from normal | Predictions | Direction right | Trades | Trade P&L | Avg per trade |",
         "|---|---|---|---|---|---|"]
    a = edge.abs()
    for lo, hi, name in buckets:
        m = (a >= lo) & (a < hi)
        n = int(m.sum())
        right = f"{correct[m].mean():.0%}" if n else "-"
        if pnl_by_edge is not None and len(pnl_by_edge):
            pm = pnl_by_edge[(pnl_by_edge["edge"].abs() >= lo) & (pnl_by_edge["edge"].abs() < hi)]["pnl"]
            tr = f"| {len(pm)} | ${pm.sum():.2f} | {'$' + format(pm.mean(), '.2f') if len(pm) else '-'} |"
        else:
            tr = "| - | - | - |"
        L.append(f"| {name} | {n} | {right} " + tr)
    L += ["", "_If \"direction right\" doesn't rise as the model gets more confident, the model "
          "isn't finding anything real, whatever the totals say._"]
    return L


def walk_forward(df):
    """Retrain every N days using only data that was known at the time."""
    h, step = MODEL["horizon_days"], MODEL["retrain_every_days"]
    n, start = len(df), MODEL["min_train_years"] * 252
    preds = pd.Series(np.nan, index=df.index)
    base = pd.Series(np.nan, index=df.index)
    for i in range(start, n - h, step):
        train = df.iloc[: i - h]          # embargo: every training label was known by day i
        test = df.iloc[i: min(i + step, n - h)]
        m = make_model().fit(train[FEATURES], train["target"])
        preds.iloc[i: i + len(test)] = m.predict_proba(test[FEATURES])[:, 1]
        base.iloc[i: i + len(test)] = train["target"].mean()
    return preds, base


def simulate_trade(df, i, kind):
    """Estimated P&L of one trade entered at day i's open, using the same stop,
    trailing ladder and time exit as live. IV proxy = VIX. Checked on daily
    closes only, so a stop is assumed hit at that day's close (can be worse
    than a real stop order, and gaps are included)."""
    single = is_single()
    dte0 = TRADE["target_dte"]
    slip = TRADE["slippage_per_leg"] * (1 if single else 2)
    s0, iv0 = df["open"].iloc[i], df["vix_open"].iloc[i] / 100
    base_k = math.floor(s0) if kind == "call" else math.ceil(s0)
    step = 1 if kind == "call" else -1
    k_long = k_short = None
    if single:
        cap, floor_ = max_premium_allowed() - slip, TRADE["min_premium"] / 100 - slip
        for off in range(0, 300):
            k = base_k + step * off
            if bs_price(kind, s0, k, dte0, iv0) <= cap:
                if bs_price(kind, s0, k, dte0, iv0) >= floor_:
                    k_long = k
                break
    else:
        w, cap = TRADE["spread_width"], max_debit_allowed() - slip
        for off in range(0, 60):
            k1 = base_k + step * off
            k2 = k1 + step * w
            if spread_value(kind, s0, k1, k2, dte0, iv0) <= cap:
                k_long, k_short = k1, k2
                break
    if k_long is None:
        return None

    def value(s, days, iv):
        if single:
            return bs_price(kind, s, k_long, days, max(iv, 0.05))
        return spread_value(kind, s, k_long, k_short, days, iv)

    entry = value(s0, dte0, iv0) + slip
    if entry <= slip:
        return None
    d0 = df.index[i]
    last = min(i + TRADE["exit_after_days"], len(df) - 1)
    peak = entry
    tp = TRADE.get("profit_target_pct")
    for j in range(i, last + 1):
        days_left = dte0 - (df.index[j] - d0).days
        val = max(value(df["close"].iloc[j], days_left, df["vix"].iloc[j] / 100) - slip, 0.0)
        peak = max(peak, val)
        stop, trailed, _ = exit_levels(entry, peak, single)
        reason = None
        if tp and val >= entry * (1 + tp):
            reason = "target"
        elif val <= stop:
            reason = "trail stop" if trailed else "stop"
        elif j == last:
            reason = "time"
        if reason:
            return {"entry_date": d0.date(), "exit_date": df.index[j].date(), "kind": kind,
                    "strike": k_long, "entry": round(entry, 3), "exit": round(val, 3),
                    "pnl": round((val - entry) * 100 - fees(), 2), "exit_idx": j, "reason": reason}
    return None


def simulate_day(df, i, kind):
    """One same-day trade: buy at the open, out by the close. Uses the day's
    high/low to see whether the stop or the trailing ladder was hit. We don't
    know whether the high or the low came first, so it assumes the WORST
    order: the low first (a call gets stopped out before any rally)."""
    single = is_single()
    dte0 = TRADE["target_dte"]
    slip = TRADE["slippage_per_leg"] * (1 if single else 2)
    s0, iv0 = df["open"].iloc[i], iv_from_vix(df["vix_open"].iloc[i], df["ts_9d"].iloc[i] if "ts_9d" in df else None)
    base_k = math.floor(s0) if kind == "call" else math.ceil(s0)
    step = 1 if kind == "call" else -1
    k_long = k_short = None
    if single:
        cap, floor_ = max_premium_allowed() - slip, TRADE["min_premium"] / 100 - slip
        for off in range(0, 300):
            k = base_k + step * off
            if bs_price(kind, s0, k, dte0, iv0) <= cap:
                if bs_price(kind, s0, k, dte0, iv0) >= floor_:
                    k_long = k
                break
    else:
        w, cap = TRADE["spread_width"], max_debit_allowed() - slip
        for off in range(0, 60):
            k1 = base_k + step * off
            k2 = k1 + step * w
            if spread_value(kind, s0, k1, k2, dte0, iv0) <= cap:
                k_long, k_short = k1, k2
                break
    if k_long is None:
        return None

    def value(s, days, iv):
        if single:
            return bs_price(kind, s, k_long, days, iv)
        return spread_value(kind, s, k_long, k_short, days, iv)

    entry = value(s0, dte0, iv0) + slip
    if entry <= slip:
        return None
    iv1 = iv_from_vix(df["vix"].iloc[i], df["ts_9d"].iloc[i] if "ts_9d" in df else None)
    worst = df["low"].iloc[i] if kind == "call" else df["high"].iloc[i]
    best = df["high"].iloc[i] if kind == "call" else df["low"].iloc[i]
    v_worst = max(value(worst, dte0, iv1) - slip, 0.0)
    v_best = max(value(best, dte0, iv1) - slip, 0.0)
    v_close = max(value(df["close"].iloc[i], dte0 - 0.3, iv1) - slip, 0.0)
    stop0, _, _ = exit_levels(entry, entry, single)
    tp = TRADE.get("profit_target_pct")
    if v_worst <= stop0:
        exit_v, reason = stop0, "stop"
    else:
        peak = max(entry, v_best)
        stop, trailed, _ = exit_levels(entry, peak, single)
        if tp and v_best >= entry * (1 + tp):
            exit_v, reason = entry * (1 + tp), "target"
        elif v_close <= stop:
            exit_v, reason = stop, "trail stop" if trailed else "stop"
        else:
            exit_v, reason = v_close, "close"
    return {"entry_date": df.index[i].date(), "exit_date": df.index[i].date(), "kind": kind,
            "strike": k_long, "entry": round(entry, 3), "exit": round(exit_v, 3),
            "pnl": round((exit_v - entry) * 100 - fees(), 2), "exit_idx": i, "reason": reason}


def run_trades(df, decide):
    trades, i = [], 0
    while i < len(df) - 1:
        kind = decide(i)
        if kind:
            t = simulate_day(df, i, kind) if is_day_trade() else simulate_trade(df, i, kind)
            if t:
                trades.append(t)
                i = t["exit_idx"] + 1      # one position at a time
                continue
        i += 1
    return pd.DataFrame(trades)


def summarize(tr, label):
    if tr.empty:
        return f"### {label}\nNo trades.\n"
    pnl = tr["pnl"]
    eq = pnl.cumsum()
    streak = worst = 0
    for p in pnl:
        streak = streak + 1 if p <= 0 else 0
        worst = max(worst, streak)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return (f"### {label}\n"
            f"- Trades: {len(tr)}  |  Win rate: {len(wins) / len(tr):.0%}\n"
            f"- Avg win: ${wins.mean() if len(wins) else 0:.2f}  |  "
            f"Avg loss: ${losses.mean() if len(losses) else 0:.2f}\n"
            f"- Total P&L: ${pnl.sum():.2f}  |  Per trade: ${pnl.mean():.2f}\n"
            f"- Worst drawdown: ${(eq - eq.cummax()).min():.2f}  |  "
            f"Longest losing streak: {worst}\n"
            f"- Exits: {tr['reason'].value_counts().to_dict()}\n")


def backtest():
    df = build_features(load_history())
    preds, base = walk_forward(df)
    df["p_up"], df["base"] = preds, base
    thr = MODEL["edge_threshold"]
    tested = df.dropna(subset=["p_up", "target"])
    auc = roc_auc_score(tested["target"], tested["p_up"])

    skip = set(TRADE.get("skip_events", []))

    def model_signal(i):
        p, b = df["p_up"].iloc[i], df["base"].iloc[i]
        if np.isnan(p):
            return None
        if skip and skip & set(events_on(df.index[i].date())):
            return None
        if p - b > thr:
            return "call"
        if p - b < -thr:
            return "put"
        return None

    first = df["p_up"].first_valid_index()
    start_i = df.index.get_loc(first)
    def always_call(i):
        return "call" if i >= start_i else None

    model_tr = run_trades(df, model_signal)
    naive_tr = run_trades(df, always_call)

    longs = tested[tested["p_up"] - tested["base"] > thr]
    shorts = tested[tested["p_up"] - tested["base"] < -thr]
    lines = [
        f"# SPY model backtest - {dt.date.today()}", "",
        f"Tested {tested.index[0].date()} to {tested.index[-1].date()} "
        f"({len(tested)} days), model = {MODEL['model_type']}, "
        f"prediction = SPY closes above the open {horizon_text()}, edge threshold = {thr}.", "",
        "## Does the model predict direction?",
        f"- AUC: **{auc:.3f}** (0.50 = coin flip; below ~0.53 is not a usable edge)",
        f"- SPY closed above the open {horizon_text()} {tested['target'].mean():.0%} of the time overall.",
        f"- When the model said UP ({len(longs)} days): rose "
        f"{longs['target'].mean() if len(longs) else float('nan'):.0%} of the time.",
        f"- When the model said DOWN ({len(shorts)} days): fell "
        f"{1 - shorts['target'].mean() if len(shorts) else float('nan'):.0%} of the time.", "",
        (f"## Simulated single options (${TRADE['min_premium']}-"
         f"${max_premium_allowed() * 100:.0f} premium, ${TRADE['stop_loss_dollars']} stop, "
         f"trailing ladder, ~{TRADE['target_dte']} DTE"
         f"{', same-day trades' if is_day_trade() else ''})" if is_single() else
         f"## Simulated spreads (max debit ${max_debit_allowed() * 100:.0f}, "
         f"${TRADE['spread_width']} wide, ~{TRADE['target_dte']} DTE)"), "",
        summarize(model_tr, "Morning model signals (analysis: the bot doesn't buy at the open unless trade.morning_entries is true)"),
        summarize(naive_tr, "Baseline: buy a call whenever flat (no model)"),
        "## How to read this",
        "- The model is only worth using if it clearly beats the baseline AND the AUC is "
        "meaningfully above 0.50. If not, it has no edge.",
        "- Option prices are estimated with Black-Scholes using VIX as volatility. Real "
        "fills, skew and spreads differ, so treat dollar figures as rough.",
        ("- Same-day trades: the stop and ladder are checked against each day's high and "
         "low, assuming the worst order (low first for calls). Real results with a live "
         "stop order can differ either way." if is_day_trade() else
         "- Stops are checked on daily closes, so a stopped trade can lose more than the "
         "stop amount here, as it can in real life after a gap."),
        "- If you decide to trust it, set \"approved\": true in config.json. "
        "Until then the alert only reports the signal.",
    ]
    if not model_tr.empty:
        tagged = model_tr.assign(event=[", ".join(events_on(d)) or "normal day"
                                        for d in pd.to_datetime(model_tr["entry_date"]).dt.date])
        lines += ["", "## Model trades on event days vs normal days", "",
                  "| Day type | Trades | Win rate | P&L | Avg per trade |", "|---|---|---|---|---|"]
        for ev, g in tagged.groupby("event"):
            lines.append(f"| {ev} | {len(g)} | {(g['pnl'] > 0).mean():.0%} | "
                         f"${g['pnl'].sum():.2f} | ${g['pnl'].mean():.2f} |")
        lines += ["", "_If one event type loses money consistently, add it to `trade.skip_events` "
                  "in config.json (e.g. [\"fomc\"]) and rerun the backtest._"
                  + (f" Currently skipping: {', '.join(sorted(skip))}." if skip else "")]
        by_year = model_tr.assign(year=pd.to_datetime(model_tr["entry_date"]).dt.year) \
            .groupby("year")["pnl"].agg(["count", "sum"]).round(2)
        lines += ["", "## Model P&L by year", "", "| Year | Trades | P&L |", "|---|---|---|"]
        lines += [f"| {y} | {int(r['count'])} | ${r['sum']:.2f} |" for y, r in by_year.iterrows()]
    # ---- 1. Does confidence matter? (morning model)
    edge_all = tested["p_up"] - tested["base"]
    correct = ((edge_all > 0) == (tested["target"] > 0.5)).astype(float)
    tr_edges = None
    if not model_tr.empty:
        e_map = (df["p_up"] - df["base"]).to_dict()
        tr_edges = model_tr.assign(edge=[e_map.get(pd.Timestamp(d), np.nan) for d in model_tr["entry_date"]])
    lines += ["", "## Does confidence matter? (morning model)", ""] + confidence_table(edge_all, correct, tr_edges)

    # ---- 2. Later-in-the-day entries + honest comparisons (hourly data, ~2 years)
    try:
        try:
            bad_ = selftest()
            lines += ["", "## Look-ahead self-test", "",
                      ("**PASSED.** Every model input was rebuilt with the future cut off or scrambled, "
                       "and nothing changed: no input peeks ahead." if not bad_ else
                       f"**FAILED** for: {', '.join(bad_)}. These inputs peek at the future, so backtest "
                       "results that use them are too optimistic."), ""]
        except Exception as e:
            lines += ["", f"_Look-ahead self-test failed to run: {e}_"]
        try:
            lines += pricing_check()
        except Exception as e:
            lines += ["", f"_Option pricing check failed: {e}_"]
        spy_h, vix_h = load_hourly()
        X, day_bars = build_intraday(df, spy_h, vix_h)
        X = walk_forward_intraday(X)
        Xt = X.dropna(subset=["p", "p_judge"])       # same days for every model, incl. the judge

        def model_decide(r):
            e = r["p"] - r["base"]
            kind = "call" if e > thr else "put" if e < -thr else None
            if kind and MODEL.get("require_agreement") and not r.get("agree", True):
                kind = None
            return None if (kind == "put" and not TRADE.get("allow_puts", True)) else kind

        if MODEL.get("meta_filter"):                  # the EV filter decides when it's on
            Xt = with_meta(Xt, day_bars, committee_names())
            model_decide = meta_decide
        it = run_intraday(Xt, day_bars, model_decide)
        try:                                          # for the dashboard's equity chart
            it.to_csv(OUT / f"backtest_intraday_{dt.date.today()}.csv", index=False)
        except Exception:
            pass
        base_it = run_intraday(Xt, day_bars, lambda r: "call" if r["k"] == 1 else None)
        ie = Xt["p"] - Xt["base"]
        ic = ((ie > 0) == (Xt["target"] > 0.5)).astype(float)
        try:
            iauc = roc_auc_score(Xt["target"], Xt["p"])
        except ValueError:
            iauc = float("nan")
        lines += ["", f"## Later-in-the-day entries (hourly data, {Xt['day'].min()} to {Xt['day'].max()})", "",
                  f"At each check from ~10:30 AM to {TRADE.get('last_entry_time', '14:30')}, a second model "
                  "predicts whether SPY closes today above its price at that moment. It also sees the "
                  "first-hour range, yesterday's high/low, VWAP and volume. Trained on hourly data only "
                  "(~2 years), so less certain than the morning model. Includes time decay.", "",
                  f"- AUC: **{iauc:.3f}** over {len(Xt)} checkpoints on {Xt['day'].nunique()} days", "",
                  summarize(it, "Model trades (first signal of each day)"),
                  summarize(base_it, "Baseline: buy a call at 10:30 every day (no model)"),
                  "_The model only deserves credit for what it makes ABOVE this baseline._", ""]
        cdir, tag = OUT / "charts", str(dt.date.today())

        def _daily(tr):
            if tr.empty:
                return None
            return tr.assign(d=pd.to_datetime(tr["entry_date"])).groupby("d")["pnl"].sum()
        if safe_chart(chart_equity, {"Model": _daily(it), "No model (call at 10:30 daily)": _daily(base_it)},
                      cdir / f"bt_equity_{tag}.png", "Later-in-the-day trades: model vs no model",
                      "simulated total P&L over the hourly-data period"):
            lines += [f"![Model vs no model](charts/bt_equity_{tag}.png)", ""]
        if len(it):
            rb = real_bar()
            strong, normal = it[it["edge"].abs() >= rb], it[it["edge"].abs() < rb]
            weeks = max(1, pd.to_datetime(it["entry_date"]).dt.to_period("W").nunique())
            def _row(name, t):
                return (f"| {name} | {len(t)} | {(t['pnl'] > 0).mean() if len(t) else 0:.0%} | "
                        f"${t['pnl'].sum() if len(t) else 0:.2f} | ${t['pnl'].mean() if len(t) else 0:.2f} |")
            lines += ["### Strongest signals only (what would trigger a real alert)", "",
                      f"Real alerts need confidence ≥ {rb:.0%}; paper trades need ≥ {thr:.0%}. "
                      f"Strong signals came about {len(strong) / weeks:.1f} times per week.", "",
                      "| Group | Trades | Win rate | P&L | Avg per trade |", "|---|---|---|---|---|",
                      _row(f"strong (≥ {rb:.0%}): real-alert trades", strong),
                      _row(f"normal ({thr:.0%}–{rb:.0%}): paper only", normal), "",
                      "_If the strong group isn't clearly better per trade, the real-alert bar isn't "
                      "picking better trades and should be revisited._", ""]
            bt_h = it.groupby("time")["pnl"].sum()
            if safe_chart(chart_signed_bars, list(bt_h.index), list(bt_h.astype(float)),
                          cdir / f"bt_by_time_{tag}.png", "Simulated P&L by entry time",
                          "later-in-the-day model"):
                lines += [f"![P&L by entry time](charts/bt_by_time_{tag}.png)", ""]
            lines += ["### By entry time", "", "| Entry time | Trades | Win rate | P&L | Avg |", "|---|---|---|---|---|"]
            for tm, g in it.groupby("time"):
                lines.append(f"| {tm} | {len(g)} | {(g['pnl'] > 0).mean():.0%} | ${g['pnl'].sum():.2f} | ${g['pnl'].mean():.2f} |")
        lines += ["", "### Does confidence matter? (later-in-the-day model)", ""] + confidence_table(ie, ic, it if len(it) else None)
        buckets = [(0, .02, "<2%"), (.02, .05, "2-5%"), (.05, .08, "5-8%"), (.08, 1, "8%+")]
        bl, bv = [], []
        for lo_, hi_, nm in buckets:
            m_ = (ie.abs() >= lo_) & (ie.abs() < hi_)
            if m_.sum():
                bl.append(nm)
                bv.append(float(ic[m_].mean()))
        if bv and safe_chart(chart_pct_bars, bl, bv, cdir / f"bt_confidence_{tag}.png",
                             "How often the direction was right, by confidence",
                             "later-in-the-day model; bars should rise left to right"):
            lines += ["", f"![Right more often when more confident?](charts/bt_confidence_{tag}.png)"]

        # ---- the committee: each member alone vs voting together
        crow = []
        def trades_with(pcol, agree_only=False):
            Xc = Xt.assign(p=Xt[pcol])
            def dec(r):
                k_ = model_decide(r)
                return None if (agree_only and not r["agree"]) else k_
            return run_intraday(Xc, day_bars, dec)
        for name in list(COMMITTEE) + ["judge"]:
            a_ = roc_auc_score(Xt["target"], Xt[f"p_{name}"])
            tr_ = trades_with(f"p_{name}")
            label = "**judge** (decides from all models + structure)" if name == "judge" \
                else f"{name} ({COMMITTEE[name][0]})"
            crow.append((label, a_, tr_))
        Xt = Xt.assign(p_all3=Xt[[f"p_{n}" for n in ("simple", "levels", "flexible")]].mean(axis=1),
                       p_committee=Xt[[f"p_{n}" for n in committee_names()]].mean(axis=1))
        cm_names = " + ".join(committee_names())
        crow.append(("all 3 averaged", roc_auc_score(Xt["target"], Xt["p_all3"]), trades_with("p_all3")))
        crow.append((f"**committee in use ({cm_names})**", roc_auc_score(Xt["target"], Xt["p_committee"]),
                     trades_with("p_committee")))
        crow.append((f"committee in use, only when members agree", float("nan"), trades_with("p_committee", True)))
        cur_m = MODEL.get("intraday_model", "committee")
        lines += ["", "## The committee: each model alone vs voting together", "",
                  f"Current setting: `intraday_model = {cur_m}`, members = {', '.join(committee_names())}, "
                  f"`require_agreement = {MODEL.get('require_agreement', False)}`", "",
                  "| Model | AUC | Trades | Win rate | P&L | Avg per trade |", "|---|---|---|---|---|---|"]
        for nm, a_, tr_ in crow:
            n_ = len(tr_)
            lines.append(f"| {nm} | {'-' if np.isnan(a_) else format(a_, '.3f')} | {n_} | "
                         f"{(tr_['pnl'] > 0).mean() if n_ else 0:.0%} | ${tr_['pnl'].sum() if n_ else 0:.2f} | "
                         f"${tr_['pnl'].mean() if n_ else 0:.2f} |")
        lines += ["", "_A committee usually makes fewer wild calls than any single member. Prefer it unless "
                  "one member is clearly and consistently better._"]
        try:
            lines += quant_report(Xt, day_bars, it, cdir, tag, decide=model_decide)
        except Exception as e:
            lines += ["", f"_Quant checks failed: {e}_"]
        lines += self_tune(Xt, day_bars)

        # ---- exit styles: same signals, different ways of getting out
        saved = {k_: copy.deepcopy(TRADE.get(k_)) for k_ in ("profit_target_pct", "trail_ladder", "exit_profile")}
        prof_rows = []
        for name in EXIT_PROFILES:
            TRADE["exit_profile"] = name          # the exit bot / per-trade style reads this
            apply_exit_profile(name)
            mt = run_trades(df, model_signal)
            itp = run_intraday(Xt, day_bars, model_decide)
            prof_rows.append((name, mt, itp))
        TRADE.update(saved)
        cur = TRADE.get("exit_profile", "custom")
        lines += ["", "## Exit styles compared (same signals, different exits)", "",
                  "| Exit style | Morning trades P&L | Morning win rate | Later-in-day P&L | Later-in-day win rate |",
                  "|---|---|---|---|---|"]
        for name, mt, itp in prof_rows:
            mark = " ← current" if name == cur else ""
            lines.append(f"| {name}{mark} | ${mt['pnl'].sum() if len(mt) else 0:.2f} | "
                         f"{(mt['pnl'] > 0).mean() if len(mt) else 0:.0%} | "
                         f"${itp['pnl'].sum() if len(itp) else 0:.2f} | "
                         f"{(itp['pnl'] > 0).mean() if len(itp) else 0:.0%} |")
        lines += ["", "- **quick**: breakeven stop at +15%, lock +10% at +25%, sell at +30%",
                  "- **balanced**: breakeven at +25%, lock +20% at +40%, sell at +60%",
                  "- **runner**: no target; breakeven at +50%, +50% at +100%, +100% at +200%",
                  "", "_Pick with care: choosing the best of three is a small form of fitting the past. "
                  "Prefer a style that does well in BOTH columns._"]

        # ---- model comparison
        saved_mt = MODEL["model_type"]
        comp = []
        for mt_name in ("logistic", "gbm"):
            MODEL["model_type"] = mt_name
            pr, bs = walk_forward(df)
            ok = ~(pr.isna() | df["target"].isna())
            a1 = roc_auc_score(df["target"][ok], pr[ok])
            a2 = roc_auc_score(Xt["target"], Xt["p_levels" if mt_name == "logistic" else "p_flexible"])
            comp.append((mt_name, a1, a2))
        MODEL["model_type"] = saved_mt
        lines += ["", "## Simple vs flexible model (AUC, higher is better, 0.50 = coin flip)", "",
                  "| Model | Morning | Later-in-day |", "|---|---|---|"]
        lines += [f"| {n}{' ← current' if n == saved_mt else ''} | {a:.3f} | {b:.3f} |" for n, a, b in comp]
        lines += ["", "_Switch `model.model_type` only if the other one is clearly better in both columns._"]

        # realistic entry for the morning model: same trades, bought ~1 hour after the open
        if not model_tr.empty:
            cmp_rows = []
            for _, t in model_tr.iterrows():
                d = t["entry_date"]
                if d not in day_bars:
                    continue
                g = day_bars[d]
                drow = df.loc[pd.Timestamp(d)]
                late = sim_bars(t["kind"], float(g["Open"].iloc[1]),
                                iv_from_vix(float(drow["vix_open"]), drow["ts_9d"] if "ts_9d" in drow else None),
                                list(zip(g["High"].iloc[1:], g["Low"].iloc[1:], g["Close"].iloc[1:])))
                if late:
                    cmp_rows.append((t["pnl"], (late[2] - late[1]) * 100 - fees()))
            if cmp_rows:
                a = np.array(cmp_rows)
                lines += ["", "## Morning trades: bought at the open vs about an hour later", "",
                          f"Same {len(a)} morning-signal days in the hourly-data period:", "",
                          "| Entry | Total P&L | Avg per trade | Win rate |", "|---|---|---|---|",
                          f"| At the 9:30 open (what the main backtest assumes) | ${a[:, 0].sum():.2f} | ${a[:, 0].mean():.2f} | {(a[:, 0] > 0).mean():.0%} |",
                          f"| ~10:30 AM (closer to when you'd really buy) | ${a[:, 1].sum():.2f} | ${a[:, 1].mean():.2f} | {(a[:, 1] > 0).mean():.0%} |",
                          "", "_If the later entry is much worse, the morning edge disappears by the time you can act on it._"]
    except Exception as ex:
        lines += ["", "## Later-in-the-day entries", f"Skipped: couldn't build the hourly test ({ex})."]

    report = "\n".join(lines)
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule":      # the monthly automatic re-test
        keep = [l for l in lines if l.startswith("- AUC") or l.startswith("| committee")
                or "← current" in l]
        notify("SPY bot monthly re-test",
               "Fresh backtest on the latest data. Full report in reports/ on GitHub; ask Claude "
               "to review it.\n\n" + "\n".join(keep[:8]), important=True)
    path = OUT / f"backtest_{dt.date.today()}.md"
    path.write_text(report, encoding="utf-8")
    if not model_tr.empty:
        model_tr.drop(columns="exit_idx").to_csv(OUT / f"backtest_trades_{dt.date.today()}.csv",
                                                 index=False)
    print(report)
    print(f"\nSaved to {path}")


# ================================================================= alerts

def pick_expiration(t):
    today = dt.date.today()
    best = None
    for e in t.options:
        dte = (dt.date.fromisoformat(e) - today).days
        if dte < TRADE["min_dte"]:
            continue
        if best is None or abs(dte - TRADE["target_dte"]) < abs(best[1] - TRADE["target_dte"]):
            best = (e, dte)
    return best


def leg_quote(row):
    bid, ask = float(row["bid"] or 0), float(row["ask"] or 0)
    if not (bid > 0 and ask > 0):
        return None
    mid = (bid + ask) / 2
    oi = row.get("openInterest")
    oi = 0 if oi is None or (isinstance(oi, float) and math.isnan(oi)) else int(oi)
    return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": (ask - bid) / mid, "oi": oi,
            "iv": float(row.get("impliedVolatility") or 0)}


def find_spread(kind, spot, exp, chain):
    table = chain.calls if kind == "call" else chain.puts
    rows = {float(r["strike"]): r for _, r in table.iterrows()}
    w, cap = TRADE["spread_width"], max_debit_allowed()
    strikes = sorted(k for k in rows if (k >= math.floor(spot) - 1 if kind == "call"
                                         else k <= math.ceil(spot) + 1))
    if kind == "put":
        strikes = strikes[::-1]
    for k1 in strikes:
        k2 = k1 + w if kind == "call" else k1 - w
        if k2 not in rows:
            continue
        q1, q2 = leg_quote(rows[k1]), leg_quote(rows[k2])
        if not q1 or not q2:
            continue
        if max(q1["spread_pct"], q2["spread_pct"]) > RULES["max_spread_pct"]:
            continue
        if min(q1["oi"], q2["oi"]) < RULES["min_open_interest"]:
            continue
        debit = round(q1["mid"] - q2["mid"], 2)
        if 0 < debit <= cap:
            return {"kind": kind, "expiration": exp, "long": k1, "short": k2,
                    "debit": debit, "long_q": q1, "short_q": q2}
    return None


def find_single(kind, spot, exp, chain):
    """Most expensive (closest to the money) liquid option priced inside
    the min/max premium and your account limits."""
    table = chain.calls if kind == "call" else chain.puts
    cap, floor_ = max_premium_allowed(), TRADE["min_premium"] / 100
    best = None
    for _, r in table.iterrows():
        q = leg_quote(r)
        if not q or q["spread_pct"] > RULES["max_spread_pct"] or q["oi"] < RULES["min_open_interest"]:
            continue
        price = round(q["mid"], 2)
        if floor_ <= price <= cap and (best is None or price > best["debit"]):
            best = {"kind": kind, "expiration": exp, "long": float(r["strike"]), "short": None,
                    "debit": price, "long_q": q}
    return best


def find_trade(kind, spot, exp, chain):
    return find_single(kind, spot, exp, chain) if is_single() else find_spread(kind, spot, exp, chain)


def trade_message(sp, spot, p_up, base):
    if sp.get("short") is None:
        return single_message(sp, spot, p_up, base)
    return spread_message(sp, spot, p_up, base)


def why_lines(sp):
    w = sp.get("why")
    if not w:
        return [""]
    out = ["",
           f"Why this contract: checked {w['checked']} options priced ${TRADE['min_premium']}-"
           f"${TRADE['max_premium']} across {w['expirations']} expirations ({w['liquid']} liquid enough).",
           f"  If SPY moves {w['move']:.1%} your way: about {sp['gain']:+.0%}. If it moves {w['move']:.1%} "
           f"against you: about {sp['loss']:+.0%}. Spread cost {sp['cost']:.0%}, time decay by the close "
           f"{sp['decay']:+.0%}. Expected {sp['ev']:+.1%}."]
    if w["alts"]:
        out.append("  Runners-up: " + "; ".join(
            f"{a['long']:g} {a['kind']} exp {a['expiration']} ${a['debit']:.2f} ({a['ev']:+.1%})"
            for a in w["alts"]))
    return out + [""]


def single_message(sp, spot, p_up, base):
    d, k = sp["debit"], sp["long"]
    stop, _, _ = exit_levels(d, d, True)
    be = k + d if sp["kind"] == "call" else k - d
    te = time_exit_date(str(dt.date.today()), sp["expiration"])
    side = "BULLISH (buy a call)" if sp["kind"] == "call" else "BEARISH (buy a put)"
    tp = TRADE.get("profit_target_pct")
    return "\n".join([
        f"SPY SIGNAL: {side}",
        f"Model: {p_up:.0%} chance SPY closes above today's open {horizon_text()} "
        f"(normal {base:.0%}). SPY now ${spot:.2f}.",
        "",
        "Suggested order (1 contract, place it yourself):",
        f"  BUY SPY {sp['expiration']} {k:g} {sp['kind']}  "
        f"(bid {sp['long_q']['bid']:.2f} / ask {sp['long_q']['ask']:.2f})",
        f"  Limit: ${d:.2f} (= ${d * 100:.0f}). Don't pay more than "
        f"${min(d + 0.05, max_premium_allowed()):.2f}.",
        "",
        *why_lines(sp),
        f"Stop loss: sell if it falls to ${stop:.2f} (about -${TRADE['stop_loss_dollars']}). "
        "You can set this as a stop-limit sell order in Robinhood.",
        f"  A gap can open below the stop; the most you can lose is ${d * 100:.0f}.",
        "Moving the stop up as it works:",
        *ladder_lines(d),
        (f"Take profit: sell when it's worth ${d * (1 + tp):.2f}." if tp else
         "Take profit: no fixed target; the trailing stop locks in gains."),
        ("Time exit: SELL BY 3:45 PM TODAY, win or lose. Don't hold it overnight."
         if is_day_trade() else f"Time exit: close by {te} whatever happens."),
        *([] if is_day_trade() else [f"Breakeven at expiration: SPY ${be:.2f}."]),
        "",
        "After you buy it, tell Claude \"log my trade\" (or add it to open_position in "
        "config.json) so the bot can tell you when to move the stop.",
        "Quotes are from Yahoo and can lag. Check the live price in Robinhood first.",
    ])


def spread_message(sp, spot, p_up, base):
    d, w = sp["debit"], TRADE["spread_width"]
    be = sp["long"] + d if sp["kind"] == "call" else sp["long"] - d
    entry = dt.date.today()
    time_exit = time_exit_date(str(entry), sp["expiration"])
    side = "BULLISH (call debit spread)" if sp["kind"] == "call" else "BEARISH (put debit spread)"
    return "\n".join([
        f"SPY SIGNAL: {side}",
        f"Model: {p_up:.0%} chance SPY closes above today's open {horizon_text()} "
        f"(normal {base:.0%}). SPY now ${spot:.2f}.",
        "",
        "Suggested order (1 contract, place it yourself):",
        f"  BUY  SPY {sp['expiration']} {sp['long']:g} {sp['kind']}  "
        f"(bid {sp['long_q']['bid']:.2f} / ask {sp['long_q']['ask']:.2f})",
        f"  SELL SPY {sp['expiration']} {sp['short']:g} {sp['kind']}  "
        f"(bid {sp['short_q']['bid']:.2f} / ask {sp['short_q']['ask']:.2f})",
        f"  Limit: ${d:.2f} debit. Don't pay more than ${min(d + 0.03, max_debit_allowed()):.2f}.",
        "",
        f"Max loss: ${d * 100:.0f}   Max gain: ${(w - d) * 100:.0f}   Breakeven: SPY ${be:.2f}",
        f"Stop loss: close when spread is worth ${exit_levels(d, d, False)[0]:.2f}",
        "Moving the stop up as it works:",
        *ladder_lines(d),
        ("Time exit: SELL BY 3:45 PM TODAY, win or lose." if is_day_trade()
         else f"Time exit:  close by {time_exit} whatever happens."),
        "",
        "Robinhood may not accept stop orders on spreads: set a price alert and close manually.",
        "After you place it, add it to open_position in config.json so the bot tracks it.",
        "Quotes are from Yahoo and can lag. Check the live price in Robinhood first.",
    ])


def check_position(pos, state, cache, force_close=False):
    """Your real open trade: value, trailing stop, and whether to act."""
    short = pos.get("short")
    single = short in (None, "", 0)
    val = current_value(pos["kind"], pos["expiration"], pos["long"], None if single else short, cache)
    desc = "SPY " + describe(pos["kind"], pos["long"], None if single else float(short))
    if val is None:
        return f"OPEN POSITION: {desc}: couldn't get quotes today. Check it in Robinhood."
    entry = float(pos["entry_debit"])
    key = f"{desc}-{pos['expiration']}-{pos['entry_date']}"
    ps = state.setdefault("position", {})
    if ps.get("key") != key:
        ps.clear()
        ps.update(key=key, peak=entry, stop_told=None)
    ps["peak"] = max(ps["peak"], val)
    stop, trailed, nxt = exit_levels(entry, ps["peak"], single)
    te = time_exit_date(pos["entry_date"], pos["expiration"])
    chg = val / entry - 1
    status = (f"OPEN POSITION: {desc} exp {pos['expiration']} worth ~${val:.2f} vs "
              f"${entry:.2f} paid ({chg:+.0%}, ${(val - entry) * 100:+.0f}).")
    if is_day_trade():
        te = dt.date.max
    with using_exit(pos.get("exit_style") or TRADE.get("exit_profile")):
        reason = exit_reason(entry, val, ps["peak"], single, dt.date.today(), te)
        stop, trailed, nxt = exit_levels(entry, ps["peak"], single)
    if force_close and not reason:
        return status + ("\n>>> MARKET CLOSES SOON: SELL IT NOW. Day trades aren't held "
                         "overnight. Then tell Claude \"log my trade\".")
    if reason == "target":
        return status + "\n>>> TAKE-PROFIT level reached. Consider closing it now."
    if reason in ("stop", "trail stop"):
        return status + (f"\n>>> STOP level ${stop:.2f} reached. Close it now if your stop "
                         "order hasn't already.")
    if reason == "time":
        return status + f"\n>>> TIME EXIT ({te}) reached. Consider closing it now."
    if trailed and ps.get("stop_told") != round(stop, 2):
        ps["stop_told"] = round(stop, 2)
        return status + (f"\n>>> MOVE YOUR STOP UP to ${stop:.2f} (it reached "
                         f"${ps['peak']:.2f}). Update your stop order in Robinhood.")
    hold = f"\nHold. Stop at ${stop:.2f}"
    if nxt:
        hold += f"; when it's worth ${nxt[0]:.2f}, move the stop to ${nxt[1]:.2f}"
    return status + hold + ("; sell by 3:45 PM." if is_day_trade() else f"; time exit {te}.")


def log_prediction(day, p_up, base, kind, spot):
    """Append today's prediction (one row per market day) to predictions.csv."""
    path = HERE / "predictions.csv"
    row = pd.DataFrame([{"date": str(day), "p_up": round(p_up, 4), "base": round(base, 4),
                         "signal": kind or "none", "spot": round(spot, 2)}])
    if path.exists():
        old = pd.read_csv(path, dtype={"date": str})
        row = pd.concat([old[old["date"] != str(day)], row], ignore_index=True)
    row.to_csv(path, index=False)


def live_track_record(df):
    """Score past live predictions whose horizon has passed: did SPY close higher
    than that day's open after `horizon_days` trading days?"""
    path = HERE / "predictions.csv"
    if not path.exists():
        return "Live track record: starts today."
    preds = pd.read_csv(path, dtype={"date": str})
    h = MODEL["horizon_days"]
    close, opens = df["close"], df["open"]
    dates = list(close.index.date)
    outcomes = []
    for _, r in preds.iterrows():
        d = dt.date.fromisoformat(r["date"])
        if d not in dates:
            continue
        i = dates.index(d)
        end = i + h - 1
        if end < len(dates) - 1:            # only fully finished days
            outcomes.append((r["signal"], r["p_up"] > r["base"], close.iloc[end] > opens.iloc[i]))
    if not outcomes:
        return (f"Live track record: {len(preds)} predictions logged, first results after "
                f"{h} trading days.")
    lean = sum(1 for _, up_call, went_up in outcomes if up_call == went_up) / len(outcomes)
    sig = [(s, w) for s, _, w in outcomes if s != "none"]
    sig_txt = "no trade signals scored yet"
    if sig:
        right = sum(1 for s, w in sig if (s == "call") == w)
        sig_txt = f"trade signals right {right}/{len(sig)} ({right / len(sig):.0%})"
    return (f"Live track record ({len(outcomes)} scored days): leaned the right way "
            f"{lean:.0%} of the time; {sig_txt}. Judge it on 50+ days, not a handful.")


def intraday_track_record(df):
    """Score the 30-minute checks: did SPY close above the price at that moment,
    grouped by time of day. Returns markdown lines."""
    path = HERE / "predictions_intraday.csv"
    if not path.exists():
        return ["_No intraday checks scored yet._"]
    pr = pd.read_csv(path, dtype={"date": str, "time": str})
    closes = {d.date(): c for d, c in df["close"].items()}
    today = dt.date.today()
    pr["d"] = pr["date"].map(dt.date.fromisoformat)
    pr = pr[(pr["d"] < today) & pr["d"].isin(closes)]
    if pr.empty:
        return ["_No intraday checks scored yet (scored after each day's close)._"]
    pr["went_up"] = [closes[d] > px for d, px in zip(pr["d"], pr["price"])]
    pr["right"] = (pr["p_up"] > pr["base"]) == pr["went_up"]
    pr["hour"] = pr["time"].str[:2] + ":00"
    L = ["| Hour | Checks | Leaned right | Signals | Signals right |", "|---|---|---|---|---|"]
    for hr, g in pr.groupby("hour"):
        sg = g[g["signal"] != "none"]
        sr = f"{((sg['signal'] == 'call') == sg['went_up']).mean():.0%}" if len(sg) else "-"
        L.append(f"| {hr} | {len(g)} | {g['right'].mean():.0%} | {len(sg)} | {sr} |")
    return L


# ============================================================ paper trades

PAPER_PATH = HERE / "paper_trades.csv"
PAPER_COLS = ["id", "entry_date", "entry_time", "source", "exit_style", "strength", "exp_value", "kind", "expiration", "long", "short", "entry",
              "status", "last_value", "peak", "exit_date", "exit", "reason", "pnl", "opinions"]


def load_paper():
    if PAPER_PATH.exists():
        p = pd.read_csv(PAPER_PATH, dtype={"entry_date": str, "exit_date": str,
                                           "expiration": str, "id": str})
    else:
        p = pd.DataFrame(columns=PAPER_COLS)
    for col in PAPER_COLS:
        if col not in p.columns:
            p[col] = np.nan
    for col in ("id", "entry_date", "entry_time", "source", "exit_style", "kind", "expiration", "status", "exit_date", "reason", "opinions"):
        p[col] = p[col].astype(object)
    for col in ("exp_value", "long", "short", "entry", "last_value", "peak", "exit", "pnl", "strength"):
        p[col] = pd.to_numeric(p[col], errors="coerce").astype(float)
    return p


def current_value(kind, expiration, k_long, k_short, cache):
    """Mid price right now of a single option (k_short None/NaN) or a spread."""
    if expiration not in cache:
        cache[expiration] = yf.Ticker("SPY").option_chain(expiration)
    table = cache[expiration].calls if kind == "call" else cache[expiration].puts
    rows = {float(r["strike"]): r for _, r in table.iterrows()}
    q1 = leg_quote(rows[float(k_long)]) if float(k_long) in rows else None
    if not q1:
        return None
    if k_short is None or (isinstance(k_short, float) and math.isnan(k_short)):
        return q1["mid"]
    q2 = leg_quote(rows[float(k_short)]) if float(k_short) in rows else None
    return None if not q2 else max(q1["mid"] - q2["mid"], 0.0)


def time_exit_date(entry_date, expiration):
    return min((pd.Timestamp(entry_date) + pd.offsets.BDay(TRADE["exit_after_days"])).date(),
               dt.date.fromisoformat(expiration) - dt.timedelta(days=7))


def update_paper(paper, cache, force_close=False):
    """Mark open paper trades to market and close them by the same rules as real
    ones. force_close = end of day for day trades."""
    today = dt.date.today()
    closed = []
    for i, r in paper[paper["status"] == "open"].iterrows():
        single = pd.isna(r["short"])
        try:
            val = current_value(r["kind"], r["expiration"], r["long"],
                                None if single else r["short"], cache)
        except Exception:
            val = None
        if val is None:
            if not (force_close or (is_day_trade() and r["entry_date"] < str(today))):
                continue
            val = float(r["last_value"])          # no quote: use the last one we saw
        else:
            val = max(val - TRADE["slippage_per_leg"] * (1 if single else 2), 0.0)
        entry = float(r["entry"])
        peak = max(float(r["peak"]) if not pd.isna(r["peak"]) else entry, val)
        paper.at[i, "last_value"] = round(val, 3)
        paper.at[i, "peak"] = round(peak, 3)
        te = dt.date.max if is_day_trade() else time_exit_date(r["entry_date"], r["expiration"])
        style = r["exit_style"] if isinstance(r.get("exit_style"), str) else TRADE.get("exit_profile")
        with using_exit(style):
            reason = exit_reason(entry, val, peak, single, today, te)
        if not reason and is_day_trade() and r["entry_date"] < str(today):
            reason = "missed close"                # a closing run didn't happen
        if not reason and force_close:
            reason = "close"
        if reason:
            pnl = round((val - entry) * 100 - fees(), 2)
            paper.loc[i, ["status", "exit_date", "exit", "reason", "pnl"]] = \
                ["closed", str(today), round(val, 3), reason, pnl]
            closed.append(f"{describe(r['kind'], r['long'], r['short'])} from "
                          f"{r['entry_date']}: {reason}, ${pnl:+.0f}")
    return closed


def open_paper(paper, sp, source="morning", entry_time=None, exit_style=None, edge=None, opinions=None):
    """Open a paper trade unless one is already open or today's limit is reached."""
    today = str(dt.date.today())
    if (paper["status"] == "open").any():
        return paper, False
    if (paper["entry_date"] == today).sum() >= TRADE.get("max_paper_trades_per_day", 2):
        return paper, False
    entry_time = entry_time or ny_now().strftime("%H:%M")
    entry = round(sp["debit"], 3)   # same limit price you'd be told to pay; exits pay the spread
    row = {"id": f"{today}-{entry_time}-{sp['kind']}", "entry_date": today, "entry_time": entry_time,
           "exp_value": round(sp.get("ev", np.nan), 4) if sp.get("ev") is not None else np.nan,
           "source": source, "exit_style": exit_style or TRADE.get("exit_profile"),
           "strength": round(abs(edge), 4) if edge is not None else np.nan,
           "kind": sp["kind"], "expiration": sp["expiration"], "long": sp["long"],
           "short": np.nan if sp["short"] is None else sp["short"],
           "entry": entry, "status": "open", "last_value": entry, "peak": entry,
           # what each specialist thought at entry (its edge vs normal), for the trade review
           "opinions": json.dumps({k: round(float(v), 4) for k, v in opinions.items()}) if opinions else np.nan}
    return pd.concat([paper, pd.DataFrame([row])], ignore_index=True), True


def paper_summary(paper, closed_today):
    lines = []
    if closed_today:
        lines.append("Paper trades closed today: " + "; ".join(closed_today))
    op = paper[paper["status"] == "open"]
    for _, r in op.iterrows():
        chg = float(r["last_value"]) / float(r["entry"]) - 1
        lines.append(f"Paper open: {describe(r['kind'], r['long'], r['short'])} {r['expiration']} "
                     f"(from {r['entry_date']}) {chg:+.0%}")
    done = paper[paper["status"] == "closed"]
    if len(done):
        pnl = done["pnl"].astype(float)
        lines.append(f"Paper record: {len(done)} closed, {(pnl > 0).mean():.0%} winners, "
                     f"total ${pnl.sum():+.0f}, avg ${pnl.mean():+.1f}/trade. "
                     "Needs 20+ closed trades to mean much.")
    else:
        lines.append("Paper record: no closed paper trades yet.")
    return "\n".join(lines)


# ================================================================= alerts

def ny_now():
    return dt.datetime.now(ZoneInfo("America/New_York"))


def trades_this_week(state, today):
    week = today.isocalendar()[:2]
    return sum(1 for d in state.get("trade_alert_dates", [])
               if dt.date.fromisoformat(d).isocalendar()[:2] == week)


def market_open_today(today_d):
    h = yf.Ticker("SPY").history(period="5d")
    return not h.empty and h.index[-1].date() == today_d


def last_entry_time():
    return dt.time.fromisoformat(TRADE.get("last_entry_time", "14:30"))


def first_entry_time():
    return dt.time.fromisoformat(TRADE.get("first_entry_time", "10:30"))


def pick_phase(now, state, scheduled):
    """open = morning prediction; monitor = check stops and look for new signals
    during the day; close = end-of-day exits. None = nothing to do this run."""
    today = str(now.date())
    t = now.time()
    if not scheduled:
        if state.get("last_open") != today:
            return "open"
        if is_day_trade() and t >= dt.time(15, 10) and state.get("last_close") != today:
            return "close"
        return "monitor"
    if t < dt.time(9, 40) or t >= dt.time(16, 5):
        return None
    if state.get("last_open") != today and t < dt.time(11, 30):
        return "open"
    if t >= dt.time(15, 10):
        return "close" if is_day_trade() and state.get("last_close") != today else None
    return "monitor"


def live_guard(state):
    """Daily feedback from live paper results. If the last N closed paper trades
    are doing clearly worse than expected, raise the confidence bar one step
    (pickier). When results recover, step back toward the base. Waits for at
    least `cooldown` new closed trades between adjustments."""
    g = {"enabled": True, "window": 20, "worse_than": -300, "recover_above": 100,
         "step": 0.02, "max": 0.16, "cooldown": 10, **MODEL.get("live_guard", {})}
    if not g["enabled"]:
        return None
    done = load_paper()
    done = done[done["status"] == "closed"]
    n_closed = len(done)
    if n_closed < g["window"] or n_closed - state.get("guard_last_count", 0) < g["cooldown"]:
        return None
    pnl = float(done["pnl"].astype(float).tail(g["window"]).sum())
    base = MODEL.get("base_edge_threshold", MODEL["edge_threshold"])
    cur = MODEL["edge_threshold"]
    if pnl < g["worse_than"] and cur + g["step"] <= g["max"] + 1e-9:
        new = round(cur + g["step"], 3)
        msg = (f"Live guard: last {g['window']} paper trades made ${pnl:.0f}, worse than expected. "
               f"Confidence bar raised {cur:.0%} → {new:.0%} (pickier).")
    elif pnl > g["recover_above"] and cur > base + 1e-9:
        new = round(max(base, cur - g["step"]), 3)
        msg = (f"Live guard: last {g['window']} paper trades made ${pnl:.0f}, recovering. "
               f"Confidence bar lowered {cur:.0%} → {new:.0%}.")
    else:
        return None
    MODEL["edge_threshold"] = new
    state["guard_last_count"] = n_closed
    save_config()
    log_change(msg)
    return msg


def breaker_status():
    """Circuit breaker: pause REAL suggestions after a losing streak or if the
    account falls too far. Paper trading continues."""
    cb = RULES.get("circuit_breaker", {})
    res = ACCT.get("recent_real_results", [])
    streak = 0
    for r in reversed(res):
        if float(r) <= 0:
            streak += 1
        else:
            break
    if ACCT["value"] < cb.get("min_account_value", 0):
        return (f"account is ${ACCT['value']:.2f}, below the ${cb['min_account_value']} floor")
    if streak >= cb.get("max_losing_streak", 10 ** 6):
        return f"{streak} losing real trades in a row"
    return None


EARN_DEFAULT = {"enabled": True, "levels": [[30, 1]],        # always 1 a week: better, not more
                "auto_max_per_week": 1, "recent_window": 20}


def earned_status():
    """Real alerts are EARNED by the paper record. Each level needs that many closed
    paper trades AND a healthy record: total paper P&L above $0, the most recent
    `recent_window` trades not losing, and the strongest-signal paper trades (the
    ones that would have been real alerts) not losing once there are 5+ of them.
    The bot moves up on its own only to `auto_max_per_week`; above that it tells you
    and you decide. It moves down on its own whenever the record turns bad."""
    E = {**EARN_DEFAULT, **TRADE.get("earn", {})}
    p = load_paper()
    done = p[p["status"] == "closed"]
    pnl = done["pnl"].astype(float)
    n, total = len(done), float(pnl.sum()) if len(done) else 0.0
    recent = float(pnl.tail(E["recent_window"]).sum()) if n else 0.0
    strong = done[done["strength"].astype(float) >= real_bar()]
    strong_pnl = float(strong["pnl"].astype(float).sum()) if len(strong) else 0.0
    problems = []
    if total <= 0:
        problems.append(f"paper P&L is ${total:+.0f}")
    if n >= E["recent_window"] and recent < 0:
        problems.append(f"last {E['recent_window']} trades ${recent:+.0f}")
    if len(strong) >= 5 and strong_pnl <= 0:
        problems.append(f"strongest-signal trades ${strong_pnl:+.0f}")
    levels = sorted((int(a), int(b)) for a, b in E["levels"])
    earned = 0
    if not problems:
        for need, per in levels:
            if n >= need:
                earned = per
    nxt = next(((need, per) for need, per in levels if per > earned), None)
    return {"enabled": bool(E["enabled"]), "n": n, "total": total, "recent": recent,
            "strong_n": len(strong), "strong_pnl": strong_pnl, "problems": problems,
            "earned": earned, "allowed": min(earned, int(E["auto_max_per_week"])),
            "auto_max": int(E["auto_max_per_week"]), "next": nxt}


def earned_text(es=None):
    es = es or earned_status()
    if es["allowed"]:
        t = f"EARNED: up to {es['allowed']} real alert{'s' if es['allowed'] > 1 else ''}/week"
    elif es["problems"]:
        t = f"not earned ({'; '.join(es['problems'])})"
    else:
        need = es["next"][0] if es["next"] else 30
        t = f"not earned yet: {es['n']}/{need} closed paper trades"
    if es["earned"] > es["allowed"]:
        t += f" · qualifies for {es['earned']}/week: tell Claude if you want that"
    return t


def earn_check(state):
    """Daily at the close: announce when real alerts are earned, paused, or when the
    record qualifies for more than the automatic limit."""
    es = earned_status()
    if not es["enabled"] or CFG.get("approved"):
        return None
    before, now_ = state.get("earn_allowed", 0), es["allowed"]
    msg = None
    if now_ > before:
        msg = (f"Real alerts EARNED: {es['n']} closed paper trades, paper P&L ${es['total']:+.0f}. "
               f"From now on the strongest signals (confidence ≥ {real_bar():.0%}) come as REAL buy alerts, "
               f"up to {now_} a week. Paper trading continues.")
    elif now_ < before:
        msg = (f"Real alerts PAUSED ({'; '.join(es['problems']) or 'record dipped'}). Back to paper only "
               "until the record recovers. Nothing for you to do.")
    if msg:
        state["earn_allowed"] = now_
        log_change("[automatic] " + msg)
    if es["earned"] > es["auto_max"] and state.get("earn_offer") != es["earned"]:
        state["earn_offer"] = es["earned"]
        offer = (f"The paper record now qualifies for {es['earned']} real alerts a week "
                 f"({es['n']} trades, ${es['total']:+.0f}). It stays at {es['auto_max']} until you tell Claude.")
        msg = (msg + " " + offer) if msg else offer
    return msg


def review_pick():
    """Trade review: which specialists were right on the bot's OWN paper trades.
    Returns the specialists whose agreement went with better trades (needs 50+
    reviewed trades and 10+ trades each way), else []. The self-tuner then tests
    that team on history like any other change."""
    p = load_paper()
    done = p[(p["status"] == "closed") & p["opinions"].notna()]
    if len(done) < 50:
        return []
    picks = []
    for name in COMMITTEE:
        agree, disagree = [], []
        for _, r in done.iterrows():
            try:
                e = json.loads(r["opinions"]).get(name)
            except Exception:
                e = None
            if e is None:
                continue
            (agree if (e > 0) == (r["kind"] == "call") else disagree).append(float(r["pnl"]))
        if len(agree) >= 10 and len(disagree) >= 10 and np.mean(agree) > np.mean(disagree) + 5:
            picks.append(name)
    return picks


def trade_review_lines():
    p = load_paper()
    done = p[(p["status"] == "closed") & p["opinions"].notna()]
    L = ["## Trade review: which specialists were right on its own trades", ""]
    if len(done) < 30:
        return L + [f"Starts at 30 reviewed trades ({len(done)} so far). From 50, the self-tuner tests "
                    "a team of the specialists that did best here.", ""]
    L += ["| Specialist | Trades it agreed with | Their avg P&L | Trades it disagreed with | Their avg P&L |",
          "|---|---|---|---|---|"]
    for name in COMMITTEE:
        agree, disagree = [], []
        for _, r in done.iterrows():
            try:
                e = json.loads(r["opinions"]).get(name)
            except Exception:
                e = None
            if e is not None:
                (agree if (e > 0) == (r["kind"] == "call") else disagree).append(float(r["pnl"]))
        fa = f"${np.mean(agree):+.1f}" if agree else "-"
        fd = f"${np.mean(disagree):+.1f}" if disagree else "-"
        L.append(f"| {name} | {len(agree)} | {fa} | {len(disagree)} | {fd} |")
    pick = review_pick()
    L += ["", f"Team the review points to: **{' + '.join(pick) if pick else 'none yet'}** "
          "(tested by the self-tuner each month before it's ever used).", ""]
    return L


def real_bar():
    """Confidence needed for a REAL buy alert (stronger than the paper bar)."""
    return max(TRADE.get("real_edge_threshold", 0.12), MODEL["edge_threshold"])


def real_decision(kind, sp, pos, state, today_d, skipped=None, signal_before_skip=None, edge=None):
    """Decide whether a signal becomes a REAL suggestion. Returns (text, send_it).
    send_it is True (real alert), "practice" (same alert, labeled don't trade), or False."""
    if skipped and signal_before_skip:
        return (f"Signal said {signal_before_skip.upper()}, but today is a "
                f"{' and '.join(EVENT_NAMES[e] for e in skipped)} day, which you've set to skip.", False)
    if skipped:
        return f"No trade today ({' and '.join(EVENT_NAMES[e] for e in skipped)} day, set to skip).", False
    if not kind:
        return "No trade signal right now.", False
    if not sp:
        return f"Signal says {kind.upper()}, but no liquid contract fits your limits right now.", False
    if pos:
        return "Signal is active, but you already have a position. One at a time. (Paper-traded.)", False
    if edge is not None and abs(edge) < real_bar():
        return (f"Signal says {kind.upper()} (confidence {abs(edge):.0%}): strong enough to paper-trade, "
                f"not strong enough for a real alert (needs {real_bar():.0%}).", False)
    es = earned_status() if not CFG.get("approved") else None
    week_cap = (TRADE.get("max_real_trades_per_week", 1) if CFG.get("approved")
                else max(es["allowed"], 1))
    if trades_this_week(state, today_d) >= week_cap:
        return (f"Signal says {kind.upper()}, but you've already had this week's "
                f"{'practice ' if es and not es['allowed'] else ''}trade alert. Paper-traded only.", False)
    brk = breaker_status()
    if brk:
        return (f"Signal says {kind.upper()}. Real suggestions are PAUSED by the circuit breaker "
                f"({brk}). Paper-traded only. Review with Claude to resume.", False)
    if not CFG.get("approved"):
        if es["enabled"] and es["allowed"] and ACCT.get("options_enabled"):
            return None, True                   # earned by the paper record
        until = CFG["notify"].get("practice_until")
        in_period = not until or today_d <= dt.date.fromisoformat(until)
        if CFG["notify"].get("practice_alerts", True) and in_period and ACCT.get("options_enabled"):
            return None, "practice"
        return (f"Signal says {kind.upper()}. Paper-traded only: real alerts {earned_text(es)}.", False)
    if not ACCT.get("options_enabled"):
        return f"Signal says {kind.upper()}. Paper-traded only: options aren't enabled.", False
    return None, True


def alert(manage_only=False):
    """One pass of the bot. manage_only=True (the live watcher's 30-second
    checks) only manages open trades: no new signals, no morning run."""
    state_path = HERE / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    now = ny_now()
    today_d = now.date()
    today = str(today_d)
    scheduled = os.environ.get("GITHUB_EVENT_NAME") == "schedule"

    phase = pick_phase(now, state, scheduled)
    if phase is None:
        print(f"{now:%H:%M} New York: nothing to do this run.")
        return
    if manage_only and phase != "monitor":
        return
    if scheduled and not market_open_today(today_d):
        print("Market is closed today (weekend or holiday). Skipping.")
        return

    pos = ACCT.get("open_position")
    paper = load_paper()
    has_open = bool(pos) or (paper["status"] == "open").any()
    can_enter = (now.time() <= last_entry_time()
                 and (paper["entry_date"] == today).sum() < TRADE.get("max_paper_trades_per_day", 2))
    if phase == "close" and not has_open:
        state["last_close"] = today
        state_path.write_text(json.dumps(state))
        print("close: nothing open to check.")
        return
    if manage_only and not has_open:
        return
    if phase == "monitor" and not has_open and (not can_enter or now.time() < first_entry_time()):
        print("monitor: nothing open and outside the entry window.")
        return

    parts, cache = [], {}
    force = phase == "close"
    if pos:
        try:
            parts.append(check_position(pos, state, cache, force_close=force))
        except Exception as e:
            parts.append(f"OPEN POSITION: check failed ({e}). Check it in Robinhood.")
    closed_today = update_paper(paper, cache, force_close=force)
    paper.to_csv(PAPER_PATH, index=False)
    if closed_today:
        parts.append("Paper trades closed: " + "; ".join(closed_today))

    sent = False
    signal_kind = None
    if phase == "open":
        out, sent, signal_kind = open_phase(state, load_paper(), pos, today, today_d, cache)
        parts += out
    elif (phase == "monitor" and not manage_only and can_enter and now.time() >= first_entry_time()
          and not (load_paper()["status"] == "open").any()):
        try:
            out, sent, signal_kind = intraday_scan(state, load_paper(), pos, now, cache)
            parts += out
        except Exception as e:
            print(f"Intraday check failed: {e}", file=sys.stderr)

    if phase == "close":
        state["last_close"] = today
        g_msg = live_guard(state)
        if g_msg:
            parts.append(">>> " + g_msg)
        try:
            e_msg = earn_check(state)
        except Exception as e:
            e_msg = None
            print(f"Earn check failed: {e}", file=sys.stderr)
        if e_msg:
            parts.append(">>> " + e_msg)
    state_path.write_text(json.dumps(state))
    msg = "\n\n".join(x for x in parts if x)
    print(f"[{phase}] {now:%H:%M} New York\n{msg}")

    if sent:
        title = alert_title(sent, signal_kind)
    elif ">>>" in msg:
        title = "SPY: action on your open trade"
    elif signal_kind:
        title = f"SPY: {signal_kind} signal (paper only)"
    else:
        title = f"SPY bot ({phase})"
    # the live watcher checks every 30 s: send each ">>>" action only once
    action_key = " / ".join(sorted(l for l in msg.splitlines() if l.startswith(">>>")))
    if action_key and not sent:
        if state.get("last_action_key") == f"{today} {action_key}":
            action_key = ""
            msg = msg.replace(">>>", "(already sent)")
        else:
            state["last_action_key"] = f"{today} {action_key}"
            state_path.write_text(json.dumps(state))
    if manage_only and not action_key and not closed_today:
        return
    notify(title, msg, important=bool(sent) or ">>>" in msg)

    if phase in ("open", "close") or ">>>" in msg or closed_today or signal_kind:
        try:
            write_status(build_features(load_history()), state.get("last_signal", {}), load_paper(), pos)
            write_paper_journal(load_paper())
        except Exception as e:
            print(f"Status page update failed: {e}", file=sys.stderr)

    recap_phase = "close" if is_day_trade() else "open"
    if phase == recap_phase and today_d.weekday() == 4 and CFG["notify"].get("weekly_summary", True):
        df = build_features(load_history())
        recap = "\n\n".join(["Weekly recap. Full details: PAPER_JOURNAL.md and STATUS.md in your repo.",
                              live_track_record(df), paper_summary(load_paper(), [])])
        notify("SPY bot weekly recap", recap, important=True)


def score_contract(kind, spot, strike, dte, q, p_dir, move, hours_left):
    """Expected result (as a fraction of the price paid) of one option for a
    same-day trade: gain if SPY moves our way by a typical rest-of-day amount,
    loss if it moves against us (capped by the stop), weighted by the model's
    probability, minus the bid/ask cost and the time decay for the rest of the day."""
    v0 = q["mid"]
    if v0 <= 0:
        return None
    iv = q["iv"] if q["iv"] and 0.03 < q["iv"] < 3 else 0.18
    days_after = max(dte - hours_left / 24, 0.02)
    up, down = (1 + move, 1 - move) if kind == "call" else (1 - move, 1 + move)
    v_fav = bs_price(kind, spot * up, strike, days_after, iv)
    v_adv = bs_price(kind, spot * down, strike, days_after, iv)
    v_flat = bs_price(kind, spot, strike, days_after, iv)
    cost = q["spread_pct"] / 2 + TRADE["slippage_per_leg"] / v0
    tp = TRADE.get("profit_target_pct")
    gain = v_fav / v0 - 1
    if tp:
        gain = min(gain, tp)
    loss = max(v_adv / v0 - 1, -(TRADE["stop_loss_dollars"] / 100) / v0)
    ev = p_dir * gain + (1 - p_dir) * loss - cost
    return {"ev": ev, "gain": gain, "loss": loss, "cost": cost, "decay": v_flat / v0 - 1, "iv": iv}


def find_best_single(kind, spot, p_up, hours_left, rv, cache):
    """Scan every liquid option in the price range across the allowed
    expirations and pick the best expected result."""
    t = yf.Ticker("SPY")
    today = dt.date.today()
    lo_dte, hi_dte = TRADE.get("min_dte", 2), TRADE.get("max_dte", 14)
    exps = [e for e in t.options if lo_dte <= (dt.date.fromisoformat(e) - today).days <= hi_dte][:8]
    cap, floor_ = max_premium_allowed(), TRADE["min_premium"] / 100
    move = max(float(rv) / math.sqrt(252) * math.sqrt(max(hours_left, 0.5) / 6.5), 0.002)
    p_dir = p_up if kind == "call" else 1 - p_up
    cands, checked = [], 0
    for e in exps:
        if e not in cache:
            cache[e] = t.option_chain(e)
        table = cache[e].calls if kind == "call" else cache[e].puts
        dte = (dt.date.fromisoformat(e) - today).days
        for _, r in table.iterrows():
            q = leg_quote(r)
            if not q:
                continue
            price = round(q["mid"], 2)
            if not (floor_ <= price <= cap):
                continue
            checked += 1
            if q["spread_pct"] > RULES["max_spread_pct"] or q["oi"] < RULES["min_open_interest"]:
                continue
            sc = score_contract(kind, spot, float(r["strike"]), dte, q, p_dir, move, hours_left)
            if sc:
                cands.append({"kind": kind, "expiration": e, "dte": dte, "long": float(r["strike"]),
                              "short": None, "debit": price, "long_q": q, **sc})
    if not cands:
        return None
    cands.sort(key=lambda c: (c["ev"], -c["cost"]), reverse=True)
    best = dict(cands[0])
    best["why"] = {"checked": checked, "liquid": len(cands), "expirations": len(exps),
                   "move": move, "p_dir": p_dir, "alts": cands[1:3]}
    return best


QUOTES_PATH = HERE / "option_quotes.csv"


_LAST_QUOTES = {}


def record_quotes(now, spot, vix, cache, width=0.01, every_min=15, ts9=None):
    """Save real SPY option quotes (near the money, the expiration the bot would
    use) a few times an hour. Over time this becomes our own history of real
    option prices, used to check and calibrate the backtest's estimated prices."""
    if _LAST_QUOTES.get("t") and (now - _LAST_QUOTES["t"]).total_seconds() < every_min * 60:
        return
    _LAST_QUOTES["t"] = now
    try:
        t = yf.Ticker("SPY")
        exp = pick_expiration(t)
        if not exp:
            return
        e = exp[0]
        if e not in cache:
            cache[e] = t.option_chain(e)
        close_t = dt.datetime.combine(dt.date.fromisoformat(e), dt.time(16, 0), tzinfo=now.tzinfo)
        days = max((close_t - now).total_seconds() / 86400, 0.01)
        rows = []
        for kind, tbl in (("call", cache[e].calls), ("put", cache[e].puts)):
            near = tbl[(tbl["strike"] >= spot * (1 - width)) & (tbl["strike"] <= spot * (1 + width))]
            for _, r in near.iterrows():
                q = leg_quote(r)
                if q and q["spread_pct"] < 0.3:
                    rows.append({"ts": now.strftime("%Y-%m-%d %H:%M"), "spot": round(spot, 2),
                                 "vix": round(float(vix), 2),
                                 "ts_9d": round(float(ts9), 4) if ts9 is not None and np.isfinite(ts9) else np.nan,
                                 "expiration": e, "days": round(days, 4),
                                 "kind": kind, "strike": float(r["strike"]),
                                 "bid": q["bid"], "ask": q["ask"]})
        if rows:
            pd.DataFrame(rows).to_csv(QUOTES_PATH, mode="a", header=not QUOTES_PATH.exists(), index=False)
    except Exception as ex:
        print(f"Quote recording skipped: {ex}", file=sys.stderr)


def pricing_check(min_rows=500, min_days=5):
    """Compare the backtest's estimated option prices (Black-Scholes with VIX) to
    the real quotes recorded live. Finds the volatility scale that makes the
    estimates match the real mid prices best, and adopts it once there's enough data."""
    L = ["", "## Option pricing check (estimated vs real quotes)", ""]
    if not QUOTES_PATH.exists():
        return L + ["No real quotes recorded yet. The watcher records them every few minutes; "
                    f"the check starts once there are {min_rows}+ quotes from {min_days}+ days.", ""]
    q = pd.read_csv(QUOTES_PATH)
    q = q[(q["bid"] > 0) & (q["ask"] > 0)].copy()
    q["mid"] = (q["bid"] + q["ask"]) / 2
    q = q[q["mid"] >= 0.20]
    n_days = q["ts"].str[:10].nunique()
    if len(q) < min_rows or n_days < min_days:
        return L + [f"{len(q)} real quotes from {n_days} days so far; the check starts at {min_rows}+ "
                    f"quotes from {min_days}+ days.", ""]

    use9 = MODEL.get("iv_source", "vix9d") == "vix9d" and "ts_9d" in q

    def est(scale):
        out = []
        for r in q.itertuples():
            v = r.vix * (r.ts_9d if use9 and np.isfinite(r.ts_9d) and r.ts_9d > 0 else 1.0)
            out.append(bs_price(r.kind, r.spot, r.strike, r.days, max(v / 100 * scale, 0.05)))
        return np.array(out)

    def err(scale):
        return float(np.median(np.abs(est(scale) / q["mid"].values - 1)))

    grid = np.round(np.arange(0.5, 2.01, 0.05), 2)
    errs = [err(s_) for s_ in grid]
    best = float(grid[int(np.argmin(errs))])
    cur = float(MODEL.get("iv_scale", 1.0))
    e1, e_cur, e_best = err(1.0), err(cur), min(errs)
    bias = float(np.median(est(cur) / q["mid"].values - 1))
    L += [f"{len(q):,} real quotes from {n_days} days. Typical pricing error of the estimates:",
          f"- plain VIX: **{e1:.1%}** · current setting (VIX × {cur:.2f}): **{e_cur:.1%}** · "
          f"best (VIX × {best:.2f}): **{e_best:.1%}**",
          f"- With the current setting, estimates run **{bias:+.1%}** vs real prices "
          f"({'too expensive: the backtest pays too much and wins too little' if bias > 0 else 'too cheap: the backtest is too optimistic'}).", ""]
    if abs(best - cur) >= 0.05 and e_best < e_cur * 0.9:
        MODEL["iv_scale"] = best
        save_config()
        log_change(f"[automatic] Option pricing calibrated from {len(q):,} real quotes: volatility scale "
                   f"{cur:.2f} → {best:.2f} (typical pricing error {e_cur:.1%} → {e_best:.1%}).")
        L += [f"**Adopted VIX × {best:.2f}** for this and future backtests.", ""]
    return L


def get_contract(kind, spot, cache, p_up=None, hours_left=None, rv=None):
    try:
        if is_single() and p_up is not None and TRADE.get("contract_picker", "closest") == "scanner":
            return find_best_single(kind, spot, p_up, hours_left, rv, cache)
        t = yf.Ticker("SPY")
        exp = pick_expiration(t)
        if exp:
            cache.setdefault(exp[0], t.option_chain(exp[0]))
            return find_trade(kind, spot, exp[0], cache[exp[0]])
    except Exception as e:
        print(f"Option chain lookup failed: {e}", file=sys.stderr)
    return None


def open_phase(state, paper, pos, today, today_d, cache):
    """Morning run: retrain, predict, paper-trade any signal, maybe suggest a real trade."""
    raw = load_history()
    spot = float(raw["close"].iloc[-1])   # latest price, used to pick strikes
    df = build_features(raw)
    h = MODEL["horizon_days"]
    train = df.iloc[: len(df) - h].dropna(subset=["target"])   # never learn from unfinished days
    m = make_model().fit(train[FEATURES], train["target"])
    p_up = float(m.predict_proba(df[FEATURES].iloc[[-1]])[:, 1][0])
    base = float(train["target"].mean())
    edge = p_up - base
    thr = MODEL["edge_threshold"]
    kind = "call" if edge > thr else "put" if edge < -thr else None
    if kind == "put" and not TRADE.get("allow_puts", True):
        kind = None
    todays = events_on(today_d)
    skipped = sorted(set(todays) & set(TRADE.get("skip_events", [])))
    signal_before_skip = kind
    if skipped:
        kind = None
    log_prediction(df.index[-1].date(), p_up, base, kind, spot)

    morning_trades = TRADE.get("morning_entries", False)
    n_ = ny_now()
    hours_left = max((15 * 60 + 45 - (n_.hour * 60 + n_.minute)) / 60, 0.5)
    sp = get_contract(kind, spot, cache, p_up, hours_left, df["rv20"].iloc[-1]) \
        if (kind and morning_trades) else None
    if sp:
        paper, _ = open_paper(paper, sp, "morning")
    paper.to_csv(PAPER_PATH, index=False)

    decision, send = real_decision(kind, sp, pos, state, today_d, skipped, signal_before_skip,
                                   edge=edge)
    if not morning_trades and not skipped:
        lean = "up" if edge > 0 else "down"
        decision = (f"Morning read: the model leans {lean} for today ({p_up:.0%} vs normal {base:.0%}). "
                    f"No trade at the open. The bot now watches today's data as it comes in and checks "
                    f"every 30 minutes from {TRADE.get('first_entry_time', '10:30')} to "
                    f"{TRADE.get('last_entry_time', '14:30')}.")
        send = False
    elif send:
        decision = trade_message(sp, spot, p_up, base)
        state.setdefault("trade_alert_dates", []).append(today)
    elif not kind and not skipped:
        decision = ("No trade this morning. The signal isn't strong enough. The bot keeps "
                    f"checking every 30 minutes until {TRADE.get('last_entry_time', '14:30')}.")
    if kind and todays:
        decision += ("\n\nHeads-up: today has a " + " and a ".join(EVENT_NAMES[e] for e in todays)
                     + ". Expect bigger, faster moves.")
    last_event = max((max(v) for v in EVENTS.values() if v), default=None)
    if last_event is None or last_event < today_d + dt.timedelta(days=60):
        decision += "\n\nNote: events.json is running out of future dates. Ask Claude to add the next year."
    brk = breaker_status()
    state["last_open"] = today
    state["last_run"] = today
    state["last_signal"] = {"events": [EVENT_NAMES[e] for e in todays], "breaker": brk,
                            "date": str(df.index[-1].date()), "p_up": p_up, "base": base,
                            "edge": edge, "kind": kind, "decision": decision.splitlines()[0],
                            "spot": spot, "vix": float(df["vix"].iloc[-1])}
    out = [decision,
           f"Morning signal {df.index[-1].date()}: P(up) {p_up:.0%} vs normal {base:.0%} "
           f"(edge {edge:+.0%}, needs ±{thr:.0%}). SPY {spot:.2f}, VIX {df['vix'].iloc[-1]:.1f}.",
           live_track_record(df),
           paper_summary(load_paper(), [])]
    (OUT / f"alert_{today}.txt").write_text("\n\n".join(out), encoding="utf-8")
    return out, send, (kind if morning_trades else None)


def log_intraday(now, p, base, kind, price):
    path = HERE / "predictions_intraday.csv"
    row = pd.DataFrame([{"date": str(now.date()), "time": now.strftime("%H:%M"), "p_up": round(p, 4),
                         "base": round(base, 4), "signal": kind or "none", "price": round(price, 2)}])
    if path.exists():
        row = pd.concat([pd.read_csv(path, dtype={"date": str, "time": str}), row], ignore_index=True)
    row.to_csv(path, index=False)


_SCAN = {}


def intraday_scan(state, paper, pos, now, cache):
    """A 30-minute check after the morning: is there a signal right now?"""
    today_d = now.date()
    todays = events_on(today_d)
    if set(todays) & set(TRADE.get("skip_events", [])):
        return [], False, None
    raw = load_history()
    df = build_features(raw)
    if df.index[-1].date() != today_d:
        return [], False, None
    spy_h, vix_h = load_hourly()
    key = (today_d, tuple(committee_names()), bool(MODEL.get("meta_filter")))
    if _SCAN.get("key") != key:                      # train once per day, reuse all day
        X, dbars = build_intraday(df, spy_h, vix_h)
        train = X[X["day"] < today_d]
        if len(train) < 200:
            return [], False, None
        _SCAN.clear()
        _SCAN.update(key=key, X=X, base=float(train["target"].mean()),
                     models={n: make_model(mt).fit(train[f], train["target"])
                             for n, (mt, f) in committee_members().items()})
        hist = None
        if "judge" in committee_names():
            hist = walk_forward_intraday(train)
            jt = hist.dropna(subset=[f"p_{n}" for n in COMMITTEE])
            _SCAN["judge"] = make_judge().fit(judge_inputs(jt), jt["target"])
        if MODEL.get("meta_filter"):                 # EV filter: learn from past out-of-sample signals
            try:
                hist = hist if hist is not None else walk_forward_intraday(train)
                ht = hist.dropna(subset=["p"])
                oc = option_outcomes(ht, dbars)
                e_ = (ht["p"] - ht["base"]).values
                pnl_ = np.where(e_ > 0, oc["pnl_call"].values, oc["pnl_put"].values)
                cm = (np.abs(e_) > META_THR) & np.isfinite(pnl_)
                if cm.sum() >= 80:
                    dd_ = ht["day"].values[cm]
                    cnt = pd.Series(dd_).map(pd.Series(dd_).value_counts()).values
                    mm = make_meta().fit(meta_frame(ht[cm], e_[cm]), (pnl_[cm] > 0).astype(int),
                                         logisticregression__sample_weight=1.0 / cnt)
                    w_, l_ = pnl_[cm][pnl_[cm] > 0], pnl_[cm][pnl_[cm] <= 0]
                    _SCAN["meta"] = (mm, float(w_.mean()) if len(w_) else 0.0,
                                     float(-l_.mean()) if len(l_) else 0.0)
            except Exception as ex_:
                print(f"EV filter training failed: {ex_}", file=sys.stderr)
    X, base, models = _SCAN["X"], _SCAN["base"], _SCAN["models"]
    g = spy_h[spy_h.index.date == today_d]
    if g.empty:
        return [], False, None
    o, price = float(g["Open"].iloc[0]), float(g["Close"].iloc[-1])
    vg = vix_h[vix_h.index.date == today_d] if not vix_h.empty else vix_h
    v_open = float(vg["Open"].iloc[0]) if len(vg) else float(df["vix_open"].iloc[-1])
    v_now = float(vg["Close"].iloc[-1]) if len(vg) else v_open
    ex = intraday_extras(g, price, float(df["high"].iloc[-2]), float(df["low"].iloc[-2]))
    # the last hourly bar is still forming; history's checkpoint k means "k finished bars"
    kk = min(max(len(g) - 1, 1), int(X["k"].max()))
    ref = X[(X["k"] == kk) & (X["day"] < today_d)]["cumvol"].tail(20)
    cv_done = float(g["Volume"].iloc[:kk].sum()) if "Volume" in g else ex["cumvol"]
    ex["vol_ratio"] = float(np.clip(cv_done / ref.mean(), 0, 5)) if len(ref) and ref.mean() > 0 else 1.0
    row = intraday_features(df.iloc[-1], o, float(g["High"].max()), float(g["Low"].min()), price,
                            (now.hour * 60 + now.minute - 570) / 60, v_open, v_now, ex)
    hist_k = X[(X["k"] == kk) & (X["day"] < today_d)]["intr_ret"].abs().tail(14)
    nref = float(hist_k.mean()) if len(hist_k) >= 5 else 0.0
    row["noise_pos"] = float(np.clip(row["intr_ret"] / nref, -5, 5)) if nref > 0 else 0.0
    record_quotes(now, price, v_now, cache, ts9=row.get("ts_9d"))
    prior = df.iloc[-61:-1]
    row.update(structure_features(prior, daily_fvgs(prior), price, float(g["High"].max()),
                                  float(g["Low"].min())))
    done_bars = g[g.index + pd.Timedelta(hours=1) <= pd.Timestamp(now)] if len(g) > 1 else g
    row.update(smc_features(smc_daily(prior), price, float(g["High"].max()), float(g["Low"].min()),
                            done_bars if len(done_bars) else g))
    frame = pd.DataFrame([row])
    probs = {n: float(m.predict_proba(frame[COMMITTEE[n][1]])[:, 1][0]) for n, m in models.items()}
    if "judge" in committee_names():
        jf = frame.assign(vix_now=v_now, **{f"p_{n}": v for n, v in probs.items()})
        probs["judge"] = float(_SCAN["judge"].predict_proba(judge_inputs(jf))[:, 1][0])
    voters = {n: probs[n] for n in committee_names()}
    p, agree = committee_vote(voters, base)
    edge = p - base
    thr = MODEL["edge_threshold"]
    kind = "call" if edge > thr else "put" if edge < -thr else None
    ev_line = ""
    if MODEL.get("meta_filter"):                     # fail closed: no EV model yet -> no trade
        kind = None
        ev_line = " EV filter on but not trained yet, so no trade."
    if MODEL.get("meta_filter") and _SCAN.get("meta"):
        mm, aw, al = _SCAN["meta"]
        ev_line = ""
        if abs(edge) > META_THR:
            rr = pd.DataFrame([{**row, "vix_now": v_now}])
            pw = float(mm.predict_proba(meta_frame(rr, np.array([edge])))[:, 1][0])
            ev = pw * aw - (1 - pw) * al
            ev_line = f" EV filter: {pw:.0%} chance the option trade wins, expected value ${ev:+.0f}."
            if ev > MODEL.get("meta_min_ev", 0.0):
                kind = "call" if edge > 0 else "put"
    if kind and MODEL.get("require_agreement") and not agree:
        kind = None
    veto = bot_gate(kind, {**row, "vix_now": v_now}) if kind else None
    if veto:
        log_intraday(now, p, base, None, price)
        print(f"Signal {kind} vetoed: {veto}")
        return [], False, None
    style = exit_style_for(v_now)
    if kind == "put" and not TRADE.get("allow_puts", True):
        kind = None
    log_intraday(now, p, base, kind, price)
    if not kind:
        return [], False, None
    hours_left = max((15 * 60 + 45 - (now.hour * 60 + now.minute)) / 60, 0.5)
    rv = float(df["rv20"].iloc[-1]) if not pd.isna(df["rv20"].iloc[-1]) else 0.15
    sp = get_contract(kind, price, cache, p, hours_left, rv)
    if sp:
        paper, _ = open_paper(paper, sp, "intraday", now.strftime("%H:%M"), style, edge,
                              opinions={n: probs[n] - base for n in probs})
        paper.to_csv(PAPER_PATH, index=False)
    decision, send = real_decision(kind, sp, pos, state, today_d, edge=edge)
    if send:
        with using_exit(style):
            decision = trade_message(sp, price, p, base).replace(
                "closes above today's open", "closes above its current price")
        expires = (now + dt.timedelta(minutes=15)).strftime("%H:%M")
        band = price * 0.0015
        guard = (f"ACT FAST OR SKIP: this signal expires at {expires}. Only buy while SPY is between "
                 f"${price - band:.2f} and ${price + band:.2f} (now ${price:.2f}). The backtest shows this kind "
                 f"of edge is gone within an hour, so a late entry is worse than no entry.")
        decision = (f"Strongest kind of signal: confidence {abs(edge):.0%} (real-alert bar "
                    f"{real_bar():.0%}).\n{guard}\n\n" + decision)
        if send == "practice":
            decision = ("PRACTICE ALERT: DON'T TRADE. This is exactly what a real alert will look "
                        "like once you turn on real trades. The bot is paper-trading it.\n\n" + decision)
        decision += (f"\n\nExit style: {style}"
                     + (f" (exit bot: VIX {v_now:.1f} is {'calm' if style == 'runner' else 'wild'})"
                        if bots_on()["exit"] else "")
                     + f". When you log it, Claude records exit_style = {style}.")
        state.setdefault("trade_alert_dates", []).append(str(today_d))
    votes = ", ".join(f"{n} {v:.0%}" for n, v in probs.items())
    ctx = []
    if row.get("sweep") == 1:
        ctx.append("swept yesterday's low and reclaimed it")
    elif row.get("sweep") == -1:
        ctx.append("swept yesterday's high and lost it")
    if abs(row.get("d_hi20", 1)) < 0.003:
        ctx.append("right at the 20-day high (resistance)")
    if abs(row.get("d_lo20", 1)) < 0.003:
        ctx.append("right at the 20-day low (support)")
    if row.get("d_fvg_below", 1) < 0.004:
        ctx.append("just above an unfilled fair value gap")
    if row.get("d_fvg_above", 1) < 0.004:
        ctx.append("just below an unfilled fair value gap")
    if row.get("in_fvg") == 1:
        ctx.append("inside an unfilled bullish fair value gap")
    elif row.get("in_fvg") == -1:
        ctx.append("inside an unfilled bearish fair value gap")
    line = (f"{now:%H:%M} signal: P(SPY closes above ${price:.2f}) {p:.0%} vs normal {base:.0%} "
            f"(edge {edge:+.0%}). Committee votes: {votes}"
            f"{' (all agree)' if agree else ' (split)'}."
            + (f" Structure: {'; '.join(ctx)}." if ctx else "") + ev_line)
    return [decision, line], send, kind


def alert_title(sent, kind):
    what = f"buy {kind.upper()}" if is_single() else f"{kind.upper()} spread"
    return f"PRACTICE (don't trade): SPY {what}" if sent == "practice" else f"SPY TRADE: {what}"


REAL_PATH = HERE / "real_trades.csv"


def load_real():
    """Real trades, written by Claude when you say "log my trade"."""
    if REAL_PATH.exists():
        r = pd.read_csv(REAL_PATH, dtype={"date": str})
        return r[pd.to_numeric(r["pnl"], errors="coerce").notna()]
    return pd.DataFrame(columns=["date", "contract", "entry", "exit", "pnl", "result", "notes"])


def weekly_learning(df, paper):
    """Week by week: how often the 30-minute checks leaned the right way, and paper results."""
    rows = {}
    path = HERE / "predictions_intraday.csv"
    if path.exists():
        pr = pd.read_csv(path, dtype={"date": str, "time": str})
        closes = {d.date(): c for d, c in df["close"].items()}
        pr["d"] = pr["date"].map(dt.date.fromisoformat)
        pr = pr[(pr["d"] < dt.date.today()) & pr["d"].isin(closes)]
        if len(pr):
            pr["right"] = (pr["p_up"] > pr["base"]) == [closes[d] > px for d, px in zip(pr["d"], pr["price"])]
            pr["wk"] = pd.to_datetime(pr["d"]).dt.to_period("W-FRI")
            for wk, g in pr.groupby("wk"):
                rows.setdefault(wk, {})["checks"] = len(g)
                rows[wk]["right"] = g["right"].mean()
    done = paper[paper["status"] == "closed"]
    if len(done):
        done = done.assign(wk=pd.to_datetime(done["entry_date"]).dt.to_period("W-FRI"))
        for wk, g in done.groupby("wk"):
            rows.setdefault(wk, {})["trades"] = len(g)
            rows[wk]["pnl"] = g["pnl"].astype(float).sum()
            rows[wk]["wins"] = int((g["pnl"].astype(float) > 0).sum())
    return [(wk, rows[wk]) for wk in sorted(rows)]


def write_status(df, sig, paper, pos):
    """STATUS.md: a page in the repo showing everything the bot is doing."""
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# SPY bot: how it's doing", f"_Last updated: {now}_", "",
         "## Today",
         f"- SPY {df['close'].iloc[-1]:.2f}, VIX {df['vix'].iloc[-1]:.1f}",
         (f"- Model ({sig.get('date')}): {sig.get('p_up', 0):.0%} chance SPY closes above the "
          f"open {horizon_text()} (normal {sig.get('base', 0):.0%}, edge "
          f"{sig.get('edge', 0):+.0%}, needs ±{MODEL['edge_threshold']:.0%})" if sig else
          "- Model: no prediction yet"),
         f"- Morning signal: **{(sig.get('kind') or 'none').upper()}** (the bot also checks every 30 min "
         f"until {TRADE.get('last_entry_time', '14:30')})",
         f"- Circuit breaker: **{'PAUSED: ' + sig['breaker'] if sig.get('breaker') else 'OK'}**",
         f"- Scheduled events today: {', '.join(sig.get('events') or []) or 'none'}"
         + (f" · skipping: {', '.join(TRADE.get('skip_events'))}" if TRADE.get('skip_events') else ""),
         f"- Decision: {sig.get('decision', '-')}",
         f"- Trade style: {'same-day (in after the open, out before 3:45 PM)' if is_day_trade() else 'multi-day'}"
         f" · stop ${TRADE.get('stop_loss_dollars', '-')} · option price "
         f"${TRADE.get('min_premium', '-')}-${TRADE.get('max_premium', '-')}",
         f"- Model approved for real suggestions: **{CFG.get('approved')}** · "
         f"options enabled: **{ACCT.get('options_enabled')}**", ""]
    # ---- scoreboard: paper vs real
    done = paper[paper["status"] == "closed"]
    real = load_real()

    def sb(name, pnl):
        st_ = _stats(pnl) if len(pnl) else _stats(pd.Series([], dtype=float))
        return (f"| {name} | {st_['n']} | {st_['rate']} | {st_['total']} | {st_['avg']} | "
                f"{st_['best']} | {st_['worst']} |")
    strong = done[pd.to_numeric(done["strength"], errors="coerce") >= real_bar()]
    L += ["## Scoreboard", "", "| | Trades | Win rate | Total P&L | Avg per trade | Best | Worst |",
          "|---|---|---|---|---|---|---|",
          sb("**Real trades** (your money)", real["pnl"].astype(float) if len(real) else pd.Series([], dtype=float)),
          sb("**Paper trades** (all signals)", done["pnl"].astype(float)),
          sb(f"Paper: strongest signals only (≥ {real_bar():.0%}, would-be real alerts)",
             strong["pnl"].astype(float)), "",
          "_Real trades: TRADE_JOURNAL.md (kept by Claude). Every paper trade: PAPER_JOURNAL.md._", ""]

    # ---- is it learning?
    wl = weekly_learning(df, paper)
    L += ["## Is it learning?", "",
          "Week by week: how often the 30-minute checks leaned the right way (50% = coin flip), "
          "and how the paper trades did. Look for the \"right\" column staying above 50% and paper "
          "P&L trending up over several weeks; a single week means little.", ""]
    if wl:
        L += ["| Week ending | Checks scored | Leaned right | Paper trades | Paper wins | Paper P&L |",
              "|---|---|---|---|---|---|"]
        for wk, r in wl[-12:]:
            L.append(f"| {wk.end_time.date()} | {r.get('checks', 0)} | "
                     f"{format(r['right'], '.0%') if 'right' in r else '-'} | {r.get('trades', 0)} | "
                     f"{r.get('wins', 0)} | {_money(r['pnl'], True) if 'pnl' in r else '-'} |")
        L.append("")
        cdir = HERE / "charts"
        acc = [(str(wk.end_time.date())[5:], r["right"]) for wk, r in wl[-12:] if "right" in r]
        if acc and safe_chart(chart_pct_bars, [a for a, _ in acc], [b for _, b in acc],
                              cdir / "learning_accuracy.png", "Leaned the right way, by week",
                              "30-minute checks scored after each close"):
            L += ["![Accuracy by week](charts/learning_accuracy.png)", ""]
        pw = [(str(wk.end_time.date())[5:], r["pnl"]) for wk, r in wl[-12:] if "pnl" in r]
        if pw and safe_chart(chart_signed_bars, [a for a, _ in pw], [b for _, b in pw],
                             cdir / "learning_paper_weekly.png", "Paper P&L by week",
                             "gains above the line, losses below"):
            L += ["![Paper P&L by week](charts/learning_paper_weekly.png)", ""]
    else:
        L += ["_Nothing scored yet. The first week fills in after a few trading days._", ""]

    auto = []
    cl = HERE / "CHANGELOG.md"
    if cl.exists():
        auto = [l for l in cl.read_text(encoding="utf-8").splitlines()
                if re.match(r"^\| \d{4}-\d{2}-\d{2} \| \[automatic\]", l)][:5]
    L += ["## Current setup (the bot can change this itself)", "",
          f"- Models voting: **{' + '.join(committee_names())}**",
          f"- Confidence bar: paper **{MODEL['edge_threshold']:.0%}** (base "
          f"{MODEL.get('base_edge_threshold', MODEL['edge_threshold']):.0%}; the live guard raises it after "
          f"a bad run) · real alerts **{real_bar():.0%}**, max {TRADE.get('max_real_trades_per_week', 1)}/week",
          f"- Exit style: **{TRADE.get('exit_profile')}** · helper bots on: "
          f"**{', '.join(b for b, v in bots_on().items() if v) or 'none'}**",
          f"- Real alerts: **{'ON (approved by you)' if CFG.get('approved') else earned_text()}**"
          + ("" if CFG.get("approved") else
             f" · practice alerts until {CFG['notify'].get('practice_until', '-')}"),
          "- How real alerts are earned: 30+ closed paper trades with paper P&L above $0, the last 20 "
          "trades not losing, and the strongest-signal trades not losing → 1 real alert/week for the "
          "strongest signals. It stays at 1 a week: as the models improve, the goal is more of those "
          "alerts winning, not more alerts. Pauses automatically if the record turns bad.", "",
          "**Recent changes it made on its own:**", ""]
    L += [("- " + l.strip("| ").replace(" | ", ": ", 1)) for l in auto] or ["- none yet"]
    L.append("")

    if pos:
        L += ["## Your open trade",
              f"SPY {describe(pos['kind'], pos['long'], pos.get('short') or None)} exp {pos['expiration']}, "
              f"paid ${float(pos['entry_debit']):.2f} on {pos['entry_date']}", ""]
    L += ["## Live track record (morning model)", live_track_record(df), "",
          "## Intraday checks by time of day", ""] + intraday_track_record(df) + [""]

    L += ["## Paper trading (last 10; full list in PAPER_JOURNAL.md)",
          paper_summary(paper, []).replace("\n", "  \n"), ""]
    if len(done):
        eq = done["pnl"].astype(float).cumsum()
        L += [f"Cumulative paper P&L: **${eq.iloc[-1]:+.0f}** · worst drawdown "
              f"${(eq - eq.cummax()).min():.0f}", ""]
    if len(paper):
        L += ["| Opened | Type | Strikes | Expires | Paid | Now/Exit | Status | Result |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in paper.iloc[::-1].head(10).iterrows():
            is_open = r["status"] == "open"
            val = r["last_value"] if is_open else r["exit"]
            res = f"{float(val) / float(r['entry']) - 1:+.0%} (open)" if is_open else \
                  f"${float(r['pnl']):+.0f} ({r['reason']})"
            L.append(f"| {r['entry_date']} | {r['kind']} | {describe(r['kind'], r['long'], r['short'])} | "
                     f"{r['expiration']} | ${float(r['entry']):.2f} | ${float(val):.2f} | "
                     f"{r['status']} | {res} |")
        L.append("")

    preds_path = HERE / "predictions.csv"
    if preds_path.exists():
        preds = pd.read_csv(preds_path, dtype={"date": str}).tail(20).iloc[::-1]
        L += ["## Last 20 predictions", "| Date | P(up) | Normal | Signal | SPY |",
              "|---|---|---|---|---|"]
        L += [f"| {r['date']} | {r['p_up']:.0%} | {r['base']:.0%} | {r['signal']} | "
              f"{r['spot']:.2f} |" for _, r in preds.iterrows()]
    (HERE / "STATUS.md").write_text("\n".join(L), encoding="utf-8")


# ================================================================= charts
# Static PNGs that GitHub shows inline. Colors validated for colorblind safety:
# blue/orange = two series, blue/red = gains/losses (also shown by position vs 0).
CHART = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
         "blue": "#2a78d6", "orange": "#eb6834", "red": "#e34948", "neutral": "#9a9892"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "axes.titlesize": 12, "axes.titleweight": "bold"})
    return plt, FuncFormatter


def _money(v, sign=False):
    """$1,234 / −$1,234 (and +$1,234 when sign=True)."""
    if v < 0:
        return f"−${abs(v):,.0f}"
    return f"{'+' if sign and v > 0 else ''}${v:,.0f}"


def _new_ax(title, subtitle=None, size=(8, 3.6)):
    plt, FuncFormatter = _plt()
    fig, ax = plt.subplots(figsize=size, dpi=110)
    fig.patch.set_facecolor(CHART["surface"])
    ax.set_facecolor(CHART["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CHART["grid"])
    ax.tick_params(colors=CHART["ink2"], length=0)
    ax.grid(axis="y", color=CHART["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=CHART["ink"], pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=CHART["ink2"], fontsize=9)
    return plt, fig, ax, FuncFormatter


def _save(plt, fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=CHART["surface"])
    plt.close(fig)


def chart_equity(series, path, title, subtitle=None):
    """Cumulative P&L lines. series = {label: pd.Series of per-trade P&L indexed by date}."""
    colors = [CHART["blue"], CHART["orange"]]
    plt, fig, ax, Fmt = _new_ax(title, subtitle)
    ax.axhline(0, color=CHART["neutral"], linewidth=1)
    for (label, pnl), col in zip(series.items(), colors):
        if pnl is None or len(pnl) == 0:
            continue
        eq = pnl.cumsum()
        ax.plot(eq.index, eq.values, color=col, linewidth=2, label=label)
        ax.annotate(_money(eq.iloc[-1], True), (eq.index[-1], eq.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=CHART["ink"], fontsize=9, fontweight="bold", annotation_clip=False)
    ax.yaxis.set_major_formatter(Fmt(lambda v, _: _money(v)))
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    if len(series) >= 2:
        ax.legend(frameon=False, loc="upper left", labelcolor=CHART["ink2"])
    ax.margins(x=0.02)
    _save(plt, fig, path)


def chart_signed_bars(labels, values, path, title, subtitle=None, label_values=True):
    """Bars above/below zero: gains in blue, losses in red."""
    plt, fig, ax, Fmt = _new_ax(title, subtitle)
    cols = [CHART["blue"] if v >= 0 else CHART["red"] for v in values]
    x = range(len(values))
    ax.bar(x, values, color=cols, width=0.7, edgecolor=CHART["surface"], linewidth=2)
    ax.axhline(0, color=CHART["neutral"], linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=CHART["ink2"], rotation=0 if len(labels) <= 8 else 45,
                       ha="center" if len(labels) <= 8 else "right")
    ax.yaxis.set_major_formatter(Fmt(lambda v, _: _money(v)))
    if label_values and len(values) <= 12:
        for xi, v in zip(x, values):
            ax.annotate(_money(v, True), (xi, v), xytext=(0, 4 if v >= 0 else -12),
                        textcoords="offset points", ha="center", color=CHART["ink"], fontsize=9)
    ax.margins(y=0.15)
    _save(plt, fig, path)


def chart_pct_bars(labels, values, path, title, subtitle=None, ref=0.5):
    """Share-right bars with a reference line (e.g. 50% = coin flip)."""
    subtitle = (subtitle + " · " if subtitle else "") + f"dashed line = coin flip ({ref:.0%})"
    plt, fig, ax, Fmt = _new_ax(title, subtitle)
    x = range(len(values))
    ax.bar(x, values, color=CHART["blue"], width=0.6, edgecolor=CHART["surface"], linewidth=2)
    ax.axhline(ref, color=CHART["neutral"], linewidth=1, linestyle="--")
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    for xi, v in zip(x, values):
        ax.annotate(f"{v:.0%}", (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=CHART["ink"], fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=CHART["ink2"])
    lo = math.floor((min(values + [ref]) - 0.03) * 20) / 20
    ax.set_ylim(max(0, lo), min(1, max(values + [ref]) + 0.05))
    ax.yaxis.set_major_formatter(Fmt(lambda v, _: f"{v:.0%}"))
    _save(plt, fig, path)


def safe_chart(fn, *a, **k):
    """Charts are a bonus: never let a drawing problem stop the bot."""
    try:
        fn(*a, **k)
        return True
    except Exception as e:
        print(f"Chart skipped ({getattr(fn, '__name__', 'chart')}): {e}", file=sys.stderr)
        return False


def _stats(pnl):
    pnl = pnl.astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = pnl.cumsum()
    return {"n": len(pnl), "wins": len(wins),
            "rate": f"{len(wins) / len(pnl):.0%}" if len(pnl) else "-",
            "total": f"${pnl.sum():+.2f}", "avg": f"${pnl.mean():+.2f}" if len(pnl) else "-",
            "avg_win": f"${wins.mean():+.2f}" if len(wins) else "-",
            "avg_loss": f"${losses.mean():+.2f}" if len(losses) else "-",
            "best": f"${pnl.max():+.2f}" if len(pnl) else "-",
            "worst": f"${pnl.min():+.2f}" if len(pnl) else "-",
            "dd": f"${(eq - eq.cummax()).min():.2f}" if len(pnl) else "-"}


def write_paper_journal(paper):
    """PAPER_JOURNAL.md: every practice trade the bot has taken, organized."""
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    done = paper[paper["status"] == "closed"].copy()
    op = paper[paper["status"] == "open"]
    st = _stats(done["pnl"]) if len(done) else _stats(pd.Series([], dtype=float))
    L = ["# Paper Trading Journal",
         f"_Practice trades only, no real money. Written automatically by the bot. Last update: {now}_",
         "", "Real-money trades are in **TRADE_JOURNAL.md**. Bot changes are in **CHANGELOG.md**.", "",
         "## Scoreboard", "",
         "| Closed trades | Winners | Win rate | Total P&L | Avg per trade | Avg win | Avg loss | Best | Worst | Worst drawdown | Open now |",
         "|---|---|---|---|---|---|---|---|---|---|---|",
         f"| {st['n']} | {st['wins']} | {st['rate']} | {st['total']} | {st['avg']} | {st['avg_win']} | "
         f"{st['avg_loss']} | {st['best']} | {st['worst']} | {st['dd']} | {len(op)} |", "",
         "_Judge it after 20+ closed trades. A handful of results is mostly luck._", ""]
    if len(done):
        cdir = HERE / "charts"
        dd = done.assign(d=pd.to_datetime(done["entry_date"])).sort_values("d")
        pnl_s = pd.Series(dd["pnl"].astype(float).values, index=dd["d"])
        ok1 = safe_chart(chart_equity, {"Paper P&L": pnl_s}, cdir / "paper_equity.png",
                         "Paper trading: total P&L over time", f"{len(dd)} closed practice trades")
        last = dd.tail(30)
        ok2 = safe_chart(chart_signed_bars, [d.strftime("%m-%d") for d in last["d"]],
                         list(last["pnl"].astype(float)), cdir / "paper_trades.png",
                         "Each paper trade (last 30)", "gains above the line, losses below",
                         label_values=len(last) <= 12)
        hrs = dd.assign(h=dd["entry_time"].fillna("10:30").str[:2] + ":00").groupby("h")["pnl"].sum()
        ok3 = safe_chart(chart_signed_bars, list(hrs.index), list(hrs.astype(float)),
                         cdir / "paper_by_hour.png", "Paper P&L by entry hour",
                         "which times of day are working")
        L += ["## Charts", ""]
        L += ["![Paper P&L over time](charts/paper_equity.png)", ""] if ok1 else []
        L += ["![Each paper trade](charts/paper_trades.png)", ""] if ok2 else []
        L += ["![Paper P&L by entry hour](charts/paper_by_hour.png)", ""] if ok3 else []

    L += trade_review_lines()
    L += ["## Open paper trades", ""]
    if len(op):
        L += ["| Opened | Time | Source | Contract | Expires | Paid | Now | Change | Highest |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in op.iterrows():
            L.append(f"| {r['entry_date']} | {r['entry_time'] if not pd.isna(r['entry_time']) else '-'} | "
                     f"{r['source'] if not pd.isna(r['source']) else '-'} | SPY {describe(r['kind'], r['long'], r['short'])} | {r['expiration']} | "
                     f"${float(r['entry']):.2f} | ${float(r['last_value']):.2f} | "
                     f"{float(r['last_value']) / float(r['entry']) - 1:+.0%} | ${float(r['peak']):.2f} |")
    else:
        L.append("_None._")
    L.append("")

    L += ["## Closed paper trades (newest first)", ""]
    if len(done):
        L += ["| Date | Time | Source | Contract | Expires | Paid | Exit | Highest | P&L | P&L % | How it ended |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in done.iloc[::-1].iterrows():
            e = float(r["entry"])
            L.append(f"| {r['entry_date']} | {r['entry_time'] if not pd.isna(r['entry_time']) else '-'} | "
                     f"{r['source'] if not pd.isna(r['source']) else '-'} | SPY {describe(r['kind'], r['long'], r['short'])} | {r['expiration']} | "
                     f"${e:.2f} | ${float(r['exit']):.2f} | "
                     f"${float(r['peak']) if not pd.isna(r['peak']) else e:.2f} | "
                     f"${float(r['pnl']):+.2f} | {float(r['exit']) / e - 1:+.0%} | {r['reason']} |")
        L.append("")
        done["month"] = done["entry_date"].str[:7]
        L += ["## By month", "", "| Month | Trades | Winners | Win rate | P&L |", "|---|---|---|---|---|"]
        for mth, g in done.groupby("month"):
            s_ = _stats(g["pnl"])
            L.append(f"| {mth} | {s_['n']} | {s_['wins']} | {s_['rate']} | {s_['total']} |")
        L += ["", "## By how the trade ended", "", "| Exit | Trades | Win rate | P&L | Avg |", "|---|---|---|---|---|"]
        for rsn, g in done.groupby("reason"):
            s_ = _stats(g["pnl"])
            L.append(f"| {rsn} | {s_['n']} | {s_['rate']} | {s_['total']} | {s_['avg']} |")
        done["src"] = done["source"].fillna("morning")
        L += ["", "## Morning signal vs later in the day", "", "| Source | Trades | Win rate | P&L | Avg |",
              "|---|---|---|---|---|"]
        for src, g in done.groupby("src"):
            s_ = _stats(g["pnl"])
            L.append(f"| {src} | {s_['n']} | {s_['rate']} | {s_['total']} | {s_['avg']} |")
        done["hour"] = done["entry_time"].fillna("09:45").str[:2] + ":00"
        L += ["", "## By entry hour", "", "| Entered | Trades | Win rate | P&L | Avg |", "|---|---|---|---|---|"]
        for hr, g in done.groupby("hour"):
            s_ = _stats(g["pnl"])
            L.append(f"| {hr} hour | {s_['n']} | {s_['rate']} | {s_['total']} | {s_['avg']} |")
        stren = pd.to_numeric(done["strength"], errors="coerce")
        if stren.notna().any():
            L += ["", "## Strongest signals vs the rest", "",
                  "| Signal strength | Trades | Win rate | P&L | Avg |", "|---|---|---|---|---|"]
            for nm, g in (("strong (would-be real alert)", done[stren >= real_bar()]),
                          ("normal (paper only)", done[stren < real_bar()])):
                s_ = _stats(g["pnl"]) if len(g) else _stats(pd.Series([], dtype=float))
                L.append(f"| {nm} | {s_['n']} | {s_['rate']} | {s_['total']} | {s_['avg']} |")
        L += ["", "## Calls vs puts", "", "| Type | Trades | Win rate | P&L | Avg |", "|---|---|---|---|---|"]
        for knd, g in done.groupby("kind"):
            s_ = _stats(g["pnl"])
            L.append(f"| {knd} | {s_['n']} | {s_['rate']} | {s_['total']} | {s_['avg']} |")
        L.append("")
    else:
        L += ["_None yet._", ""]
    L += ["## What the exits mean",
          "- **stop**: fell to the $-stop you set",
          "- **trail stop**: had risen, then fell back to the raised stop (profit or breakeven locked in)",
          "- **target**: hit a fixed take-profit (only if one is set)",
          "- **close**: sold at the end-of-day close-out",
          "- **missed close**: the closing run didn't happen, closed at the next morning's check"]
    (HERE / "PAPER_JOURNAL.md").write_text("\n".join(L), encoding="utf-8")


# ========================================================== notifications

def notify(title, body, important):
    n = CFG["notify"]
    if n.get("only_when_action") and not important:
        return
    topic = os.environ.get("NTFY_TOPIC")
    if n.get("ntfy") and topic:
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), method="POST",
                headers={"Title": title.encode("ascii", "ignore").decode(),
                         "Priority": "high" if important else "default",
                         "Tags": "chart_with_upwards_trend"})
            urllib.request.urlopen(req, timeout=20)
        except Exception as ex:
            print(f"ntfy failed: {ex}", file=sys.stderr)
    hook = os.environ.get("DISCORD_WEBHOOK", "").strip()
    if hook and n.get("discord", True):            # free: a Discord channel webhook
        try:
            text = f"**{title}**\n{body}"
            data = json.dumps({"content": text[:1990], "username": "SPY bot"}).encode("utf-8")
            req = urllib.request.Request(hook, data=data, method="POST",
                                         headers={"Content-Type": "application/json", "User-Agent": "spy-bot"})
            urllib.request.urlopen(req, timeout=20)
        except Exception as ex:
            print(f"Discord failed: {ex}", file=sys.stderr)
    e = n.get("email", {})
    if e.get("enabled"):
        try:
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = title, e["from"], e["to"]
            msg.set_content(body)
            with smtplib.SMTP_SSL(e["smtp_host"], e["smtp_port"]) as s:
                s.login(e["from"], os.environ["EMAIL_APP_PASSWORD"])
                s.send_message(msg)
        except Exception as ex:
            print(f"Email failed: {ex}", file=sys.stderr)


# ============================================================= dashboard data
def dashboard_data():
    """Everything the private dashboard shows, read from the repo's own files (no
    market data needed). Returns {doc_path: body}; each body stays well under the
    store's 256 KiB document limit."""
    def num(x, d=2):
        try:
            v = float(x)
            return None if not np.isfinite(v) else round(v, d)
        except (TypeError, ValueError):
            return None

    p = load_paper()
    done = p[p["status"] == "closed"].copy()
    paper_rows = [{"date": r["entry_date"], "time": r.get("entry_time"), "kind": r["kind"],
                   "strike": num(r["long"], 1), "expiration": r["expiration"], "entry": num(r["entry"], 3),
                   "exit": num(r["exit"], 3), "pnl": num(r["pnl"]), "status": r["status"],
                   "reason": r.get("reason") if isinstance(r.get("reason"), str) else None,
                   "strength": num(r.get("strength"), 4)} for _, r in p.iterrows()]
    real = load_real()
    real_rows = [{k: (num(v) if k in ("entry", "exit", "pnl") else v) for k, v in r.items()}
                 for r in real.to_dict("records")]

    def stats(pnls):
        pnls = [x for x in pnls if x is not None]
        if not pnls:
            return {"n": 0}
        a = np.array(pnls, dtype=float)
        return {"n": len(a), "wins": int((a > 0).sum()), "total": num(a.sum()), "avg": num(a.mean()),
                "best": num(a.max()), "worst": num(a.min())}

    state = json.loads((HERE / "state.json").read_text()) if (HERE / "state.json").exists() else {}
    try:
        runner = json.loads((HERE / "runner.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        runner = None
    checks = []
    ip = HERE / "predictions_intraday.csv"
    try:
        pr = pd.read_csv(ip, dtype={"date": str, "time": str}) if ip.exists() else pd.DataFrame()
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        pr = pd.DataFrame()
    if len(pr) and "date" in pr:
        keep = sorted(pr["date"].unique())[-5:]
        checks = [{"date": r["date"], "time": r["time"], "p": num(r["p_up"], 4), "base": num(r["base"], 4),
                   "signal": r["signal"], "price": num(r["price"])} for _, r in pr[pr["date"].isin(keep)].iterrows()]
    changes = []
    cl = HERE / "CHANGELOG.md"
    if cl.exists():
        for l in cl.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\| (\d{4}-\d{2}-\d{2}) \| (.*) \|$", l)
            if m:
                txt = re.sub(r"\*\*|`", "", m.group(2))
                changes.append({"date": m.group(1), "auto": txt.startswith("[automatic]"),
                                "text": txt.replace("[automatic] ", "")[:600]})
    weekly = {}
    for r in paper_rows:
        if r["status"] == "closed" and r["pnl"] is not None:
            wk = str(pd.Timestamp(r["date"]).to_period("W-FRI").end_time.date())
            w = weekly.setdefault(wk, {"week": wk, "trades": 0, "wins": 0, "pnl": 0.0})
            w["trades"] += 1
            w["wins"] += int(r["pnl"] > 0)
            w["pnl"] = round(w["pnl"] + r["pnl"], 2)
    es = earned_status()
    summary = {
        "updated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "setup": {"members": committee_names(), "ev_filter": bool(MODEL.get("meta_filter")),
                  "paper_bar": num(MODEL["edge_threshold"], 3), "real_bar": num(real_bar(), 3),
                  "exit": TRADE.get("exit_profile"), "stop": TRADE.get("stop_loss_dollars"),
                  "premium": [TRADE.get("min_premium"), TRADE.get("max_premium")],
                  "entries": [TRADE.get("first_entry_time"), TRADE.get("last_entry_time")],
                  "iv_source": MODEL.get("iv_source", "vix9d"), "iv_scale": num(MODEL.get("iv_scale", 1.0), 2)},
        "paper": stats([r["pnl"] for r in paper_rows if r["status"] == "closed"]),
        "real": stats([r.get("pnl") for r in real_rows]),
        "earned": {"text": earned_text(es), "allowed": es["allowed"], "n": es["n"],
                   "need": (es["next"][0] if es["next"] else 30), "problems": es["problems"]},
        "practice_until": CFG["notify"].get("practice_until"),
        "breaker": breaker_status(),
        "last_signal": state.get("last_signal"),
        "runner": runner,
        "account": {"value": num(ACCT.get("value")), "start": 150.0, "open_position": ACCT.get("open_position")},
        "weekly": sorted(weekly.values(), key=lambda w: w["week"]),
    }
    bt = {"report": None}
    reps = sorted(OUT.glob("backtest_2*.md"))
    if reps:
        t = reps[-1].read_text(encoding="utf-8")
        bt["report"] = reps[-1].name

        def grab(pat, cast=str):
            m = re.search(pat, t)
            return cast(m.group(1)) if m else None
        bt["period"] = grab(r"Later-in-the-day entries \(hourly data, ([^)]*)\)")
        bt["selftest"] = ("passed" if "**PASSED.**" in t else "failed" if "**FAILED**" in t else None)
        bt["sharpe"] = grab(r"\| Sharpe ratio \| ([-\d.]+)", float)
        bt["sortino"] = grab(r"\| Sortino ratio \| ([-\d.]+)", float)
        bt["max_dd"] = grab(r"\| Worst drawdown \| \$(-?[\d.]+)", float)
        bt["longest_dd"] = grab(r"\| Longest drawdown \| (\d+)", int)
        bt["pf"] = grab(r"\| Profit factor \| ([\d.]+)", float)
        bt["model_pnl"] = grab(r"The model made \*\*\$(-?[\d.]+)\*\*", float)
        bt["p_random"] = grab(r"Chance random does as well: \*\*([\d.]+)%", float)
        bt["p_direction"] = grab(r"Chance random direction does as well: \*\*([\d.]+)%", float)
        bt["verdict"] = grab(r"Verdict: \*\*([^*]+)\*\*")
        bt["late"] = grab(r"\*\*If you act an hour late\*\*[^:]*: ([^\n]+)")
        bt["pbo"] = grab(r"\(PBO\) \*\*(\d+)%", float)
        bt["dsr"] = grab(r"Deflated Sharpe of the best change[^*]*\*\*(\d+)%", float)
        bt["tuner"] = ("adopted: " + grab(r"\*\*Adopted: ([^*]+)\.\*\*")) if "**Adopted:" in t else (
            "nothing changed (overfitting check)" if "overfitting check (PBO) says" in t else "no change cleared the bar")
        bt["pricing"] = grab(r"## Option pricing check[^\n]*\n\n([^\n]+)")
        models = []
        for m in re.finditer(r"^\| \**([a-z_ +]+?)\** \(?(?:logistic|gbm)?\)?[^|]*\| ([\d.]+|-) \| (\d+) \| (\d+)% \| \$(-?[\d.]+) \| \$(-?[\d.]+) \|$",
                             t.split("## The committee")[-1].split("## Skill or luck")[0], re.M):
            models.append({"name": m.group(1).strip(), "auc": None if m.group(2) == "-" else float(m.group(2)),
                           "trades": int(m.group(3)), "win": int(m.group(4)), "pnl": float(m.group(5))})
        inuse = set(committee_names())
        bt["models"] = [dict(m_, in_use=m_["name"] in inuse) for m_ in models
                        if m_["name"] in COMMITTEE or m_["name"] == "judge"]
        info = re.findall(r"^\| ([^|]+) \| ([+-][\d.]+) \(±[\d.]+\) \| (yes|no|maybe) \|$", t, re.M)
        bt["info"] = [{"group": a.strip(), "drop": float(b), "helps": c} for a, b, c in info]
        ic = sorted(OUT.glob("backtest_intraday_*.csv"))
        try:
            tr = pd.read_csv(ic[-1]) if ic else pd.DataFrame()
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            tr = pd.DataFrame()
        if len(tr) and {"entry_date", "pnl"} <= set(tr.columns):
            daily = tr.groupby("entry_date")["pnl"].sum().cumsum()
            bt["equity"] = [[str(d), num(v)] for d, v in daily.items()]

    def clean(o):                                    # JSON-safe: no NaN/inf, no numpy types
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (dt.date, dt.datetime, pd.Timestamp)):
            return str(o)
        return o
    return clean({"bot/summary": summary, "bot/paper": {"rows": paper_rows[-400:]},
                  "bot/real": {"rows": real_rows[-400:]}, "bot/checks": {"rows": checks},
                  "bot/changes": {"rows": changes[:60]}, "bot/backtest": bt})


# ============================================================= public copy
# What can be shared: the code, paper trades, backtests, learning progress.
# What never leaves the private repo: real trades, account values, the real-money
# journal, positions, logs and state (they mention your trades).
PUBLIC_FILES = ["spy_bot.py", "requirements.txt", "events.json", "PAPER_JOURNAL.md", "paper_trades.csv",
                "predictions.csv", "predictions_intraday.csv", "option_quotes.csv", "CHANGELOG.md"]
PUBLIC_DIRS = ["charts", "reports"]
PRIVATE_WORDS = re.compile(r"real[- ]trade|real[- ]money (?:trade|result|P&L)|own idea|account (?:value|number|balance)"
                           r"|buying power|Agentic|TRADE_JOURNAL|real_trades\.csv|Robinhood account|positions? held",
                           re.IGNORECASE)
# The guard never contains anything derived from the account (not even a fingerprint:
# a 9-digit number's hash can be reversed by trying all billion numbers). It refuses
# masked account digits and any standalone 9-to-12-digit number, wherever they come from.
PRIVATE_PATTERN = re.compile("•{2,}" + r"\s*\d|(?<![\d.,$])\d{9,12}(?![\d.,])")
CHANGELOG_ROW = re.compile(r"^\| \d{4}-\d{2}-\d{2} \| .* \|$")


PUBLIC_PAGE_TEXT = [("Real and paper are kept separate", "Paper trading only (practice trades)"),
                    ("No real trades logged yet", ""),
                    ("No real trades yet. Trades you make in Robinhood are logged automatically at 4:20 PM on weekdays.", ""),
                    ("Served by the home PC. The page reloads every 2 minutes; the bot updates its data every few "
                     "minutes during market hours.", "Paper trading only. Updated after each trading day and each monthly re-test.")]


def dashboard_page(public=False):
    """The dashboard as one HTML page with the bot's data inside. public=True leaves out
    real trades, the account and anything personal (for the public repo's website)."""
    tpl = (HERE / "server" / "dashboard_template.html").read_text(encoding="utf-8")
    data = dashboard_data()
    if public:
        tpl = re.sub(r"(<!--PRIVATE-->|/\*PRIVATE\*/).*?(<!--/PRIVATE-->|/\*/PRIVATE\*/)", "", tpl, flags=re.S)
        for a, b in PUBLIC_PAGE_TEXT:
            tpl = tpl.replace(a, b)
        data["bot/summary"].update(real={"n": 0}, account=None)
        data["bot/real"] = {"rows": []}
        data["bot/changes"]["rows"] = [r for r in data["bot/changes"]["rows"] if not PRIVATE_WORDS.search(r["text"])]
    blob = json.dumps(data, default=str, allow_nan=False).replace("</", "<\\/")
    return tpl.replace("<!--BOT_DATA-->", f"<script>window.BOT_DATA = {blob};</script>")


def _private_hits(text):
    return [m.group(0) for m in PRIVATE_PATTERN.finditer(text)]


def export_public(dest):
    """Write a shareable copy of the bot into `dest` (the public repo's folder)."""
    import shutil
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in PUBLIC_FILES:
        src = HERE / f
        if not src.exists():
            continue
        if f == "CHANGELOG.md":                     # allowlist: only dated change rows, minus personal ones
            rows = [l for l in src.read_text(encoding="utf-8").splitlines()
                    if CHANGELOG_ROW.match(l) and not PRIVATE_WORDS.search(l)]
            head = ["# Bot Change Log", "Every change to the SPY bot, newest first (personal trading entries removed).",
                    "", "| Date | Change |", "|---|---|"]
            (dest / f).write_text("\n".join(head + rows) + "\n", encoding="utf-8")
        elif f.endswith(".md"):                      # drop lines that point at personal trading
            keep = [l for l in src.read_text(encoding="utf-8").splitlines() if not PRIVATE_WORDS.search(l)]
            (dest / f).write_text("\n".join(keep) + "\n", encoding="utf-8")
        else:
            shutil.copy2(src, dest / f)
    for d in PUBLIC_DIRS:
        if (HERE / d).exists():
            shutil.copytree(HERE / d, dest / d, ignore=shutil.ignore_patterns("tuner_trials.csv", "alert_*"))
    cfg = copy.deepcopy(CFG)
    cfg.pop("account", None)                         # balances, positions, results
    cfg.pop("approved", None)
    cfg.get("notify", {}).pop("email", None)
    (dest / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    p = load_paper()
    done = p[p["status"] == "closed"]
    pnl = done["pnl"].astype(float) if len(done) else pd.Series(dtype=float)
    reports = sorted((HERE / "reports").glob("backtest_*.md"))
    readme = [
        "# SPY options bot (public copy)", "",
        f"_Updated {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC. A machine-learning bot that reads SPY during the day, "
        "paper-trades short-dated SPY options, retrains daily and re-tests itself monthly. Paper trading only here: "
        "the owner's own trading is never published._", "",
        "| | |", "|---|---|",
        f"| Paper trades closed | {len(done)} |",
        f"| Paper win rate | {(pnl > 0).mean():.0%} |" if len(done) else "| Paper win rate | - |",
        f"| Paper P&L (1 contract each) | ${pnl.sum():+.2f} |" if len(done) else "| Paper P&L | - |",
        f"| Models in charge | {' + '.join(committee_names())}{' + EV filter' if MODEL.get('meta_filter') else ''} |",
        f"| Latest backtest | [{reports[-1].name}](reports/{reports[-1].name}) |" if reports else "| Latest backtest | - |",
        "", "- **[Dashboard](https://hadikhan15.github.io/spy-bot-runner/spy-bot/)**: paper trades, readings, "
        "learning and backtest health",
        "- **[PAPER_JOURNAL.md](PAPER_JOURNAL.md)**: every paper trade, with charts",
        "- **[CHANGELOG.md](CHANGELOG.md)**: every change, including the ones the bot made itself",
        "- **[reports/](reports/)**: monthly backtests with the skill-or-luck checks",
        "- **[spy_bot.py](spy_bot.py)**: the whole bot", "",
        "Not financial advice. Simulated results use estimated option prices and can differ from real fills.",
    ]
    (dest / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    try:                                              # the paper-trading dashboard (GitHub Pages)
        (dest / "index.html").write_text(dashboard_page(public=True), encoding="utf-8")
    except Exception as e:
        print(f"Public dashboard skipped: {e}", file=sys.stderr)
    leaks = {}                                        # last line of defense: scan every text file
    for f in dest.rglob("*"):
        if f.is_file() and f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            hits = _private_hits(text) + ([] if f.suffix == ".py" else PRIVATE_WORDS.findall(text))
            if hits:
                leaks[str(f.relative_to(dest))] = len(hits)
    if leaks:
        shutil.rmtree(dest)
        raise RuntimeError(f"Public copy aborted: possible personal data in {leaks}")
    print(f"Public copy written to {dest}")


# ============================================================= live watcher
SAVE_FILES = ["predictions.csv", "predictions_intraday.csv", "paper_trades.csv", "state.json",
              "STATUS.md", "PAPER_JOURNAL.md", "config.json", "CHANGELOG.md", "charts", "reports", "logs",
              "option_quotes.csv", "runner.json"]


def mark_runner(status):
    """runner.json: where the bot last ran and what it was doing (shown on the dashboard)."""
    try:
        (HERE / "runner.json").write_text(json.dumps(
            {"host": HOST, "status": status, "updated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}),
            encoding="utf-8")
    except OSError:
        pass


def git_save(msg="Live watcher save"):
    """Push the bot's memory to GitHub (inside GitHub Actions or on the home PC server)."""
    if not SYNC:
        return
    run = lambda *c: subprocess.run(list(c), cwd=HERE, capture_output=True, text=True, timeout=180)
    try:
        run("git", "config", "user.name", "spy-bot-pc" if HOST == "pc" else "spy-bot")
        run("git", "config", "user.email", "spy-bot@users.noreply.github.com")
        for f in SAVE_FILES:
            if (HERE / f).exists():
                run("git", "add", "-A", f)
        if run("git", "diff", "--cached", "--quiet").returncode != 0:
            run("git", "commit", "-m", msg + (" (home PC)" if HOST == "pc" else ""))
        run("git", "pull", "--rebase", "--autostash")
        run("git", "push")
    except subprocess.TimeoutExpired:
        print("Save to GitHub timed out; will retry at the next save.", file=sys.stderr)


def pc_active():
    """On GitHub: has the home PC server checked in within PC_FRESH_SEC? (It re-points the
    `pc-heartbeat` branch every 5 minutes while it's healthy.) Any doubt -> False, so
    GitHub runs rather than nobody running."""
    if not IN_ACTIONS:
        return False
    try:
        r = subprocess.run(["git", "fetch", "-q", "origin", HEARTBEAT_REF], cwd=HERE,
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return False
        t = subprocess.run(["git", "log", "-1", "--format=%ct", "FETCH_HEAD"], cwd=HERE,
                           capture_output=True, text=True, timeout=30)
        return -600 <= time.time() - int(t.stdout.strip()) < PC_FRESH_SEC   # allows a PC clock a bit ahead
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return False


def refresh_account():
    """Mid-day: pick up a real trade that was logged after the watcher started
    (by Claude or by you), so the watcher starts managing it within ~5 minutes."""
    if SYNC:
        subprocess.run(["git", "pull", "-q", "--rebase", "--autostash"], cwd=HERE, capture_output=True)
    try:
        fresh = json.loads(CONFIG_PATH.read_text()).get("account", {})
    except Exception:
        return
    for k in ("open_position", "value", "cash", "recent_real_results"):
        if k in fresh and fresh[k] != ACCT.get(k):
            ACCT[k] = fresh[k]
            if k == "open_position":
                print(f"Watcher: picked up open position {fresh[k]}")


def watch():
    """Runs through the trading day: every ~30 seconds it manages open trades
    (stops, trailing ladder, sell-now) with fresh prices; every 5 minutes it
    also looks for a new signal; at ~3:15 PM it runs the close-out."""
    W = {"interval_sec": 30, "scan_every_min": 5, "until": "15:50", "max_minutes": 330,
         **CFG.get("watcher", {})}
    until = dt.time.fromisoformat(W["until"])
    t_start = time.time()                            # GitHub's 6-hour limit counts from here
    now = ny_now()
    if now.time() >= until:
        print("Watcher: market day is over.")
        return
    if IN_ACTIONS and pc_active():
        print("Watcher: the home PC server is running today's watch. GitHub stands by.")
        return
    # An early start (GitHub's quiet pre-dawn timer, which runs on time far more reliably than
    # its busy daytime one) waits here for the open, so the watcher is already live at 9:31.
    if now.weekday() < 5 and now.time() < dt.time(9, 31):
        print(f"Watcher: started early ({now:%H:%M}); waiting for the open.")
        while ny_now().time() < dt.time(9, 31):
            if IN_ACTIONS and pc_active():
                print("Watcher: the home PC server came online. GitHub stands by.")
                return
            left = (dt.datetime.combine(ny_now().date(), dt.time(9, 31)) - ny_now().replace(tzinfo=None)).total_seconds()
            time.sleep(max(5, min(300, left)))
        now = ny_now()
    if not market_open_today(now.date()):
        print("Watcher: market isn't open (yet, or holiday). Exiting.")
        return
    if SYNC:
        subprocess.run(["git", "pull", "--rebase", "--autostash"], cwd=HERE, capture_output=True)
    print(f"Watcher started {now:%H:%M:%S} New York on {HOST}; checking every {W['interval_sec']}s "
          f"(new signals every {W['scan_every_min']} min) until {W['until']}.")
    mark_runner("watching")
    last_slot, last_save = None, time.time()
    # GitHub stops a job after 6 hours. Hand over before that: the runner queues a
    # fresh start every 30 minutes, and the next one carries on for the rest of the day.
    # The home PC has no such limit.
    deadline = t_start + (24 * 60 if HOST == "pc" else W["max_minutes"]) * 60
    while ny_now().time() < until:
        if time.time() > deadline:
            git_save("Live watcher: handing over to the next run")
            print(f"Watcher handing over at {ny_now():%H:%M} (job time limit); the next queued run continues.")
            return
        t0 = time.time()
        now = ny_now()
        slot = now.replace(minute=now.minute - now.minute % W["scan_every_min"], second=0, microsecond=0)
        try:
            if slot != last_slot:
                last_slot = slot
                sup = os.environ.get("SPYBOT_SUPERVISOR_FILE")
                if HOST == "pc" and sup and Path(sup).exists() and time.time() - Path(sup).stat().st_mtime > 12 * 60:
                    mark_runner("stopped: home PC server not running")   # its heartbeat stopped too,
                    git_save("Live watcher: PC server stopped")          # so GitHub takes over
                    print(f"Watcher: the PC server stopped checking in; stopping at {now:%H:%M} so GitHub can take over.")
                    return
                if IN_ACTIONS and pc_active():   # the PC came online: hand the day to it
                    mark_runner("handed over to the home PC")
                    git_save("Live watcher: home PC took over")
                    print(f"Watcher: home PC server took over at {now:%H:%M}. GitHub stands by.")
                    return
                refresh_account()            # a trade logged mid-day gets picked up here
                alert()                      # full check: signals, close-out, status pages
            else:
                alert(manage_only=True)      # fast check: open trades only
        except Exception as e:
            print(f"Watcher check failed at {now:%H:%M:%S}: {e}", file=sys.stderr)
        if time.time() - last_save > 1800:
            mark_runner("watching")
            git_save()
            last_save = time.time()
        time.sleep(max(3, W["interval_sec"] - (time.time() - t0)))
    mark_runner("finished for the day")
    git_save("Live watcher: end of day")
    print("Watcher finished for the day.")


def publish_public(repo_dir):
    """Home PC: write the public copy into a local clone of the public repo and push
    it (GitHub's runner does the same in its own publish step)."""
    import shutil
    import tempfile
    repo_dir = Path(repo_dir)
    if not (repo_dir / ".git").exists():
        print(f"Public copy skipped: {repo_dir} is not a clone of the public repo.")
        return
    run = lambda *c: subprocess.run(list(c), cwd=repo_dir, capture_output=True, text=True, timeout=180)
    run("git", "pull", "-q", "--rebase", "--autostash")
    tmp = Path(tempfile.mkdtemp()) / "public"
    export_public(tmp)                               # refuses (raises) if anything personal shows up
    shutil.rmtree(repo_dir / "spy-bot", ignore_errors=True)
    shutil.copytree(tmp, repo_dir / "spy-bot")
    shutil.rmtree(tmp.parent, ignore_errors=True)
    readme = (repo_dir / "spy-bot" / "README.md").read_text(encoding="utf-8")
    (repo_dir / "README.md").write_text(re.sub(r"\]\(([^h#][^)]*)\)", r"](spy-bot/\1)", readme), encoding="utf-8")
    run("git", "add", "-A", "spy-bot", "README.md")
    if run("git", "diff", "--cached", "--quiet").returncode != 0:
        run("git", "-c", "user.name=runner", "-c", "user.email=runner@users.noreply.github.com",
            "commit", "-q", "-m", f"Public copy {dt.date.today()}")
        run("git", "pull", "-q", "--rebase")
        r = run("git", "push", "-q")
        print("Public copy updated." if r.returncode == 0 else f"Public copy push failed: {r.stderr.strip()[:200]}")
    else:
        print("Public copy unchanged.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if (cmd in ("backtest", "alert") and IN_ACTIONS and os.environ.get("GITHUB_EVENT_NAME") == "schedule"
            and pc_active()):
        print(f"The home PC server is running and does the scheduled {cmd} itself. GitHub stands by.")
        sys.exit(0)
    if cmd == "backtest":
        mark_runner("monthly re-test")
        backtest()
        mark_runner("re-test finished")
    elif cmd == "save":
        if len(sys.argv) > 3:
            mark_runner(" ".join(sys.argv[3:]))
        git_save(sys.argv[2] if len(sys.argv) > 2 else "Bot run")
    elif cmd == "publish":
        publish_public(sys.argv[2] if len(sys.argv) > 2 else HERE.parent / "spy-bot-runner")
    elif cmd == "alert":
        alert()
    elif cmd == "watch":
        watch()
    elif cmd == "dashboard-json":
        out_ = Path(sys.argv[2] if len(sys.argv) > 2 else "dashboard_data.json")
        out_.write_text(json.dumps(dashboard_data(), default=str, allow_nan=False), encoding="utf-8")
        print(f"Dashboard data written to {out_}")
    elif cmd == "dashboard-html":                   # the home server shows this page on your network
        out_ = Path(sys.argv[2] if len(sys.argv) > 2 else HERE / "dashboard.html")
        out_.parent.mkdir(parents=True, exist_ok=True)
        tmp_ = out_.with_suffix(".tmp")
        tmp_.write_text(dashboard_page(), encoding="utf-8")
        os.replace(tmp_, out_)
        print(f"Dashboard page written to {out_}")
    elif cmd == "export-public":
        export_public(sys.argv[2] if len(sys.argv) > 2 else "public_copy")
    elif cmd == "selftest":
        bad_ = selftest()
        print("Look-ahead self-test: " + ("PASSED" if not bad_ else "FAILED: " + ", ".join(bad_)))
        sys.exit(1 if bad_ else 0)
    elif cmd == "test-notify":
        notify("SPY bot test", "If you can read this, notifications work.", important=True)
    else:
        print(__doc__)
