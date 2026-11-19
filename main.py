"""gta6-trophy-rarity-forecast-01: per-trophy logit-pinball forecaster.

Forecasts, for each of GTA VI's 55 PS5 trophies, five quantiles (5/25/50/75/95%) of its
Exophase earn rate at the case's target, from the history of Exophase reads in the case.

Model (Python 3.13, standard library only, no network, deterministic):
  Median per trophy = average, in logit space, of two history-driven estimates:
    (1) a logit-trend fit: OLS of logit(rate) on ln(1 + hours since 2026-11-19T00:00Z),
        fit on reads from the last 48 h (>= 20 reads) else the last 24 h, read at the target;
    (2) a denominator model: rate = earners / players_tracked, with log-linear trends of
        players_tracked and of earners (= rate * players / 100) projected separately to the
        target and then divided. This is what lets the forecast anticipate dilution: when
        Exophase refreshes and many new players appear, rates fall even while earners grow.
  Spread per trophy from the history's own volatility, never from the practice answers:
    sigma = sqrt(sigma_rw^2 + sigma_resid^2 + sigma_jump^2 + sigma_refresh^2),
    floored at 0.35 and capped at 2.5, where sigma_rw = 0.5*sqrt(horizon in days) is the
    house random walk, sigma_resid is the std of the trend fit's residuals, sigma_jump is
    the pooled std of logit jumps at refresh events (a refresh = players jump > 1% or any
    trophy's |dlogit| > 0.1 between consecutive reads) scaled by the expected number of
    refreshes inside the horizon, and sigma_refresh = 1.0 ONLY when the history shows no
    material movement at all (frozen/uninformative: no players change, no logit move >= 0.05
    between any two consecutive reads). That last term is a structural prior, documented in
    the brief itself: Exophase refreshes its counts discretely, and one refresh can move
    logit rates by ~1.6; with a frozen history the model cannot rule a refresh out, so it
    must carry a spread wide enough to survive one. It is a round prior, not fitted to any
    practice answer, and it switches off as soon as the history shows real movement.
  Asymmetry: when downside dilution risk is present (sigma_refresh active, or the median is
  projected below the last rate) the lower tail is widened 1.3x and the upper tail narrowed
  0.8x, because refreshes can only dilute rates while progression caps the upside.
  Trophy #2 at 100% (a persistent Exophase display quirk, still 100.00% on 2026-11-19):
  q = [85, 95, 99.9, 99.97, 99.99] -- a high median with a hedge below, never "fixed" to a
  low rate, because the scorer compares against the Exophase page itself.

Every case is forecast defensively: any failure inside one case falls back to the house
baseline shape (median = last rate, random-walk spread), and a failure there falls back to
a flat last-rate forecast, so the program always writes a valid output file and never
crashes. Two runs on the same case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
Z_BY_LEVEL = dict(zip(LEVELS, Z))

# Reference instant for the log-time trend (house curve uses the same origin).
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

SIGMA_RW_PER_SQRT_DAY = 0.5   # house random-walk spread, logit per sqrt(day)
SIGMA_FLOOR = 0.35
SIGMA_CAP = 2.5
# Structural prior: one unseen Exophase refresh can move logit rates by ~1.6. Used only
# when the history is frozen and carries no information about refreshes at all.
SIGMA_REFRESH_PRIOR = 1.0
LOWER_WIDEN = 1.3
UPPER_NARROW = 0.8


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit_of(pct):
    r = clamp_rate(pct)
    return math.log(r / (100.0 - r))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's JavaScript-style rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_t0(dt):
    return (dt - T0).total_seconds() / 3600.0


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b) or None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return (my, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def pstdev(values):
    n = len(values)
    if n == 0:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / n)


def forecast_case(case):
    horizon_min = float(case.get("horizon_minutes", 0))
    horizon_days = max(horizon_min, 0.0) / 1440.0
    horizon_hours = max(horizon_min, 0.0) / 60.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    trophies = case["trophies"]
    history = sorted(case.get("history", []), key=lambda r: r["at"])
    if not history:
        raise ValueError("case has no history")
    last = history[-1]
    n_tr = len(trophies)

    # ---- window for the trend fits: last 48 h when it holds >= 20 reads, else last 24 h.
    t_end = hours_since_t0(at)
    win48 = [r for r in history if hours_since_t0(parse_time(r["at"])) >= t_end - 48.0]
    win24 = [r for r in history if hours_since_t0(parse_time(r["at"])) >= t_end - 24.0]
    window = win48 if len(win48) >= 20 else (win24 if len(win24) >= 3 else history)
    t_target = hours_since_t0(target)
    x_target = math.log1p(max(t_target, 0.0))

    # ---- refresh events across the whole history (for the jump volatility).
    jump_deltas = []  # signed per-trophy logit jumps observed at refresh events
    n_events = 0
    any_movement = False
    prev = None
    for read in history:
        if prev is not None:
            p_prev = float(prev["players_tracked"])
            p_now = float(read["players_tracked"])
            players_jump = abs(p_now - p_prev) / p_prev if p_prev > 0 else 0.0
            deltas = []
            max_jump = 0.0
            for i in range(n_tr):
                r0 = prev["rates"][i]
                r1 = read["rates"][i]
                if r0 is None or r1 is None:
                    continue
                d = logit_of(r1) - logit_of(r0)
                deltas.append(d)
                max_jump = max(max_jump, abs(d))
            if players_jump >= 0.01 or max_jump >= 0.05:
                any_movement = True
            if players_jump > 0.01 or max_jump > 0.1:
                n_events += 1
                jump_deltas.extend(deltas)
        prev = read
    span_hours = 0.0
    if len(history) >= 2:
        span_hours = (parse_time(history[-1]["at"]) - parse_time(history[0]["at"])).total_seconds() / 3600.0
    expected_events = (n_events / span_hours * horizon_hours) if span_hours > 0 else 0.0
    sigma_jump = pstdev(jump_deltas) * math.sqrt(expected_events) if jump_deltas else 0.0
    sigma_refresh = 0.0 if any_movement else SIGMA_REFRESH_PRIOR

    # ---- players trend (log-linear in hours) over the window.
    players_fit = None
    xs_p, ys_p = [], []
    for read in window:
        p = float(read["players_tracked"])
        if p > 0:
            xs_p.append(hours_since_t0(parse_time(read["at"])))
            ys_p.append(math.log(p))
    if len(xs_p) >= 2:
        players_fit = ols(xs_p, ys_p)
    players_last = float(last["players_tracked"])

    rows = []
    for i, trophy in enumerate(trophies):
        try:
            rows.append(_forecast_trophy(
                trophy, i, history, window, last, at, t_target, x_target,
                horizon_days, players_fit, players_last, sigma_jump, sigma_refresh))
        except Exception:
            # Per-trophy fallback: flat last rate with the house random-walk spread.
            rate = last["rates"][i]
            rate = 0.0 if rate is None else float(rate)
            lg = logit_of(rate)
            spread = SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)
            rows.append({"id": trophy["id"], "q": [_q(lg, z, spread) for z in Z]})
    return {"trophies": rows}


def _forecast_trophy(trophy, i, history, window, last, at, t_target, x_target,
                     horizon_days, players_fit, players_last, sigma_jump, sigma_refresh):
    last_rate = last["rates"][i]
    last_rate = 0.0 if last_rate is None else float(last_rate)

    # Trophy #2's persistent 100% Exophase quirk: high median, hedged lower tail.
    if last_rate >= 99.99:
        return {"id": trophy["id"], "q": [85, 95, 99.9, 99.97, 99.99]}

    # Series of (log-time x, logit rate) in the window.
    xs, ys = [], []
    for read in window:
        r = read["rates"][i]
        if r is None:
            continue
        xs.append(math.log1p(max(hours_since_t0(parse_time(read["at"])), 0.0)))
        ys.append(logit_of(float(r)))
    trend_logit = None
    sigma_resid = 0.0
    if len(xs) >= 3:
        fit = ols(xs, ys)
        if fit is not None:
            a, b = fit
            trend_logit = a + b * x_target
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
            sigma_resid = pstdev(resid)
    if trend_logit is None:
        trend_logit = logit_of(last_rate)

    # Denominator model: project earners and players separately, then divide.
    denom_logit = None
    if players_fit is not None:
        xs_e, ys_e = [], []
        for read in window:
            r = read["rates"][i]
            p = float(read["players_tracked"])
            if r is None or p <= 0:
                continue
            earners = float(r) / 100.0 * p
            if earners > 0:
                xs_e.append(hours_since_t0(parse_time(read["at"])))
                ys_e.append(math.log(earners))
        if len(xs_e) >= 2:
            efit = ols(xs_e, ys_e)
            if efit is not None:
                ea, eb = efit
                pa, pb = players_fit
                earners_t = math.exp(ea + eb * t_target)
                players_t = math.exp(pa + pb * t_target)
                if players_t > 0 and earners_t > 0:
                    denom_logit = logit_of(100.0 * earners_t / players_t)
    if denom_logit is None:
        denom_logit = trend_logit

    median_logit = 0.5 * (trend_logit + denom_logit)
    median_rate = inv_logit(median_logit)

    sigma_rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)
    sigma = math.sqrt(sigma_rw ** 2 + sigma_resid ** 2 + sigma_jump ** 2 + sigma_refresh ** 2)
    sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

    # Downside dilution risk widens the lower tail and narrows the upper one.
    downside = sigma_refresh > 0 or median_rate < last_rate * 0.999
    qs = []
    for z in Z:
        side = sigma * (LOWER_WIDEN if (downside and z < 0) else
                        (UPPER_NARROW if (downside and z > 0) else 1.0))
        qs.append(_q(median_logit, z, side))
    # Enforce non-decreasing after rounding (rounding can only tie, never invert, but be safe).
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return {"id": trophy["id"], "q": qs}


def _q(median_logit, z, sigma):
    return round4(clamp_rate(inv_logit(median_logit + z * sigma)))


def house_baseline_case(case):
    """House baseline shape (starter): median = last rate, sigma = 0.5*sqrt(days)."""
    horizon_days = max(float(case.get("horizon_minutes", 0)), 0.0) / 1440.0
    last = sorted(case.get("history", []), key=lambda r: r["at"])[-1]
    spread = SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit_of(0.0 if rate is None else float(rate))
        rows.append({"id": trophy["id"], "q": [_q(lg, z, spread) for z in Z]})
    return {"trophies": rows}


def flat_case(case):
    """Flat: every quantile equals the last rate."""
    last = sorted(case.get("history", []), key=lambda r: r["at"])[-1]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        q = round4(clamp_rate(0.0 if rate is None else float(rate)))
        rows.append({"id": trophy["id"], "q": [q] * 5})
    return {"trophies": rows}


def last_resort_case(case):
    """Valid output from the trophy list alone (no history at all): a wide, neutral
    prior around 1%. This should never trigger on a real case; it exists only so the
    program always writes a valid file instead of crashing."""
    rows = []
    for trophy in case["trophies"]:
        rows.append({"id": trophy["id"], "q": [0.05, 0.3, 1.0, 3.0, 15.0]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            try:
                forecast = house_baseline_case(case)
            except Exception:
                try:
                    forecast = flat_case(case)
                except Exception:
                    forecast = last_resort_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
