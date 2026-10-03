# SPY options bot (public copy)

_Updated 2026-10-03 15:52 UTC. A machine-learning bot that reads SPY during the day, paper-trades short-dated SPY options, retrains daily and re-tests itself monthly. Paper trading only here: the owner's own trading is never published._

| | |
|---|---|
| Paper trades closed | 1 |
| Paper win rate | 0% |
| Paper P&L (1 contract each) | $-32.50 |
| Models in charge | simple + levels + flexible |
| Latest backtest | [backtest_2026-10-03.md](spy-bot/reports/backtest_2026-10-03.md) |

- **[Dashboard](https://hadikhan15.github.io/spy-bot-runner/spy-bot/)**: paper trades, readings, learning and backtest health
- **[PAPER_JOURNAL.md](spy-bot/PAPER_JOURNAL.md)**: every paper trade, with charts
- **[CHANGELOG.md](spy-bot/CHANGELOG.md)**: every change, including the ones the bot made itself
- **[reports/](spy-bot/reports/)**: monthly backtests with the skill-or-luck checks
- **[spy_bot.py](spy-bot/spy_bot.py)**: the whole bot

Not financial advice. Simulated results use estimated option prices and can differ from real fills.
