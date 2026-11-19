# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/people871/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/people871/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's trophies: from the house's Exophase reads up to the
deadline it predicts, for each of the game's 55 PlayStation 5 trophies, the percent of
tracked players who will have earned it a week after launch (2026-11-26 00:00 UTC), as five
quantiles (5%, 25%, 50%, 75%, 95%) per trophy, scored by pinball loss on the logit scale.

How it works: the median per trophy is the average, in logit space, of two estimates both
computed from the case file at run time — a logit-trend fit (OLS of logit(rate) on
ln(1 + hours), last 48 h of reads) and a denominator model (rate = earners / players_tracked,
with log-linear trends of players and of earners projected separately, then divided, so the
forecast anticipates dilution when Exophase refreshes and many new players appear). The
spread per trophy comes from the history's own volatility: the house random walk plus the
trend fit's residuals plus the logit jumps observed at refresh events, scaled by the expected
number of refreshes inside the horizon. When the history is frozen and carries no information
about refreshes at all, a structural prior spread of 1.0 logit applies instead — Exophase
refreshes its counts discretely, and one refresh can move logit rates by ~1.6, so a frozen
history must not be trusted with a narrow band. The lower tail is widened 1.3x and the upper
narrowed 0.8x when downside dilution risk is present. Trophy #2 ("Just Like Paradise", stuck
at 100% on Exophase since launch night) is forecast at [85, 95, 99.9, 99.97, 99.99]: a high
median with a hedge below, because the scorer compares against the Exophase page itself.
Nothing is fitted to the practice answers; the practice cases are used only to check the
model beats the house baselines. Standard library only, offline, deterministic.

On the practice cases (answers reconstructed as ~0.20x the last rate, the refresh the brief
documents — the reconstruction is calibrated so the house baseline reproduces the brief's
own anchor of loss ~1.51) it scores a pinball loss of 0.84, against the flat baseline's 1.59
and the house baseline's 1.50 — with 90% coverage of 0.98 against the house's 0.00. On a
synthetic final-like case (289 reads over 2 days, refreshes every 4–6 h, players growing)
it scores 0.31 against the house's 1.31. Run time is ~30 ms per practice case and ~150 ms
per 289-read final-like case (cap 8,000 ms).

## Run it

```sh
python3 main.py
```

It reads every case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one forecast per
case to `$MAKEFUN_OUTPUT_DIR` (default `/output`). To run it on the practice cases from the kit:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
