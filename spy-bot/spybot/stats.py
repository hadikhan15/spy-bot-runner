"""Statistics for judging whether results are skill or luck. Pure functions:
no settings, no data downloads, so they're easy to test and reuse."""
import math

import numpy as np
import pandas as pd


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
