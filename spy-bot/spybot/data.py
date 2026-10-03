"""Market-data helpers that don't depend on settings (pure, testable)."""
import datetime as dt

import pandas as pd


def to_hourly_930(bars):
    """Regular-hours bars (any size up to 1 hour) regrouped into hourly bars starting at
    9:30, 10:30, ... 15:30, the same layout as Yahoo's hourly SPY bars (the last one is
    the 30-minute 15:30-16:00 bar). Returns (hourly regular-hours bars, extended-hours bars)."""
    t = bars.index.time
    rth = (t >= dt.time(9, 30)) & (t < dt.time(16, 0))
    hourly = bars[rth].resample("60min", offset="30min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open"])
    return hourly, bars[~rth]
