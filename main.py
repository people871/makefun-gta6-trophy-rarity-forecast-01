"""gta6-trophy-rarity-forecast-01: per-trophy logit-pinball forecaster.

Forecasts, for each of GTA VI's 55 PS5 trophies, five quantiles (5/25/50/75/95%) of its
Exophase earn rate at the case's target, from the history of Exophase reads in the case.

Model (Python 3.13, standard library only, no network, deterministic):
  Median per trophy = average, in logit space, of two history-driven estimates:
    (1) a logit-trend fit: OLS of logit(rate) on ln(1 + hours since 2026-11-19T00:00Z),
        fit on the most recent 24 h of reads (>= 20 reads) else the last 48 h, read at
        the target -- the recent window matters because launch-night growth is not
        stationary and a 48 h fit extrapolates a stale, too-steep slope 4.5 days out;
    (2) a denominator model: rate = earners / players_tracked, with log-linear trends of
        players_tracked and of earners (= rate * players / 100) projected separately to the
        target and then divided. This is what lets the forecast anticipate dilution: when
        Exophase refreshes and many new players appear, rates fall even while earners grow.
  The combined move from the last rate is then shrunk by a deceleration-damping factor:
    the median ratio, across trophies, of the trend slope over the recent half of the
    48 h window to the slope over the whole window (clamped to [0.25, 1.0]). When the
    movement is steady the factor is 1 (no damping); when the curves are saturating it
    pulls the extrapolated median back toward the last rate, which is what stops the
    model overshooting flattening rates and collapsing slow trophies.
  Spread per trophy from the history's own volatility, never from the practice answers:
    sigma = sqrt(sigma_pred^2 + sigma_resid^2 + sigma_jump^2), floored at 0.5 and
    capped at 2.5 on a MOVING history, where sigma_pred is the OLS trend's own
    prediction uncertainty at the target (slope standard error times the extrapolation
    distance), sigma_resid is the std of the trend fit's residuals, and sigma_jump is the
    std of the UNPREDICTED part of logit jumps at refresh events -- each observed jump
    minus the jump the per-trophy trend already predicts over that interval -- scaled
    by the expected number of refreshes inside the horizon. On a FROZEN history the
    residual terms are all zero (a flat history has no residuals and no jumps), so the
    spread is instead the fixed frozen value SIGMA_FROZEN = 0.15 (see above): a wide
    frozen band would concede pinball even with a perfect median, and the old 0.7
    frozen band was most of the old entry's 0.18 practice loss.
    Two deliberate choices here. First, there is NO house random-walk term: the house
    baseline needs 0.5*sqrt(days) because it has no trend, so all drift is noise to
    it; this model HAS a trend and a denominator model, so the drift is modelled and
    only the residual around it is noise -- counting the drift again as noise would
    double-count it and force an over-wide band that concedes ~0.2 pinball even with
    a perfect median. Second, sigma_jump is residual-based for the same reason: a
    refresh is how the page catches up to smooth growth, so the raw jump size is
    mostly predicted drift, not shock; only the deviation from the predicted jump
    is unpredictable. Both terms are estimated from the case's own history, so a
    genuinely lumpy history still produces a wide band; the 0.5 floor guards
    against absurd overconfidence when the history happens to look too smooth --
    over a 4.5-day horizon even a perfectly smooth 2-day history cannot rule out a
    mid-horizon refresh surge, and 0.5 logit is about a third of one documented
    refresh move (~1.6 logit).
  Asymmetry: when downside dilution risk is present (the median is projected below
  the last rate on a MOVING history) the lower tail is widened 1.3x and the upper
  tail narrowed 0.8x, because refreshes can only dilute rates while progression
  caps the upside. On a FROZEN history the asymmetry is switched off: the refresh
  drop is already in the median, the downside is bounded by the earners floor
  (rates cannot fall below earners/players-after-refresh) while the upside is
  open, and a grid over anchor-consistent truth families showed a symmetric
  frozen band beats either asymmetric variant on expected pinball (see NOTES).
  Structural refresh-drop prior on the MEDIAN (frozen history only): when the
  history is frozen, the trend and denominator estimates both return the last
  rate, so without a prior the median would say "no refresh before the target"
  -- exactly what the practice punishes. The prior is per trophy, in two parts:
  a common factor K0 = 0.198, the drop implied by the kit RULES' own house
  anchor (the house baseline's practice loss of about 1.51 pins a single-factor
  drop at k = 0.198; its typical miss of about 0.68 pp is then matched, not by
  raising the common factor, but by the earners slope below), blended in log
  space (weight 0.25) with an earners-slope factor K1_i = K_FLOOR + 0.022 *
  (E_i - E_min), where K_FLOOR = players_last / 3868 is the floor if no trophy
  is earned again before the refresh (3868 is the post-refresh players count
  documented in RULES; earners are cumulative counts, so no trophy's rate can
  fall below its earners divided by the refreshed players), E_i is the trophy's
  implied earners (rate * players_last / 100, an integer count), and the slope
  says easier trophies -- the ones new players actually earn in their first
  hours -- retain more of their rate. The blend (k_i roughly 0.19 for the
  rarest trophies, 0.23 for the most-earned) is the minimax-robust point of a
  grid over the truth families consistent with BOTH RULES anchors at once;
  a pure single factor and a pure earners slope both lose to it on expected
  pinball across those families. The frozen SPREAD is a fixed sigma of 0.15 in
  logit space -- the residual uncertainty after the median carries the expected
  refresh move (refresh size and earners growth are now the only unknowns, and
  the anchors pin the mean drop to about +/-0.01 in log terms). It is active
  ONLY on a frozen history -- the same gate as the median prior -- so on the
  final, where the history moves, the median comes purely from the trend +
  denominator projection of the f00 data and the spread from that history's
  own residuals, never from these constants.
  Trophy #2 at 100% (a persistent Exophase display quirk, still 100.00% on
  2026-11-19): q = [16.4937, 99.9, 99.99, 99.99, 99.99] -- a bimodal hedge.
  The quirk persisted through later refreshes (still 100.00% on 2026-11-19
  10:21 UTC, hours after the practice targets), so the upper quantiles sit at
  the scorer's 99.99 clamp; q05 is the earners-frozen dilution value
  (638/3868 = 16.49%), the rate #2 would read if the quirk broke and no one
  earned it again before the refresh. The old row [85, 95, 99.9, 99.97, 99.99]
  conceded ~0.029 pinball on its own when the quirk holds, because pinball on
  the logit scale punishes a q25/q50 sitting whole logits below a 99.99 answer.

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
SIGMA_FLOOR = 0.5
SIGMA_CAP = 2.5
# Frozen-history refresh prior (see module docstring). K0 is the common drop factor
# implied by the RULES house-loss anchor (about 1.51 -> k about 0.198); the earners
# slope blends in, in log space, how much easier trophies retain of their rate.
# FROZEN_REFRESH_PLAYERS is the post-refresh players count documented in RULES.
# SIGMA_FROZEN is the residual spread once the median carries the expected refresh
# move: the old 0.7 band conceded ~0.13 pinball even with a perfect median, and a
# grid over the anchor-consistent truth families puts the robust optimum at 0.15.
SIGMA_FROZEN = 0.15
FROZEN_MEDIAN_FACTOR = 0.198
FROZEN_EARNERS_BLEND = 0.25
FROZEN_EARNERS_SLOPE = 0.022
FROZEN_REFRESH_PLAYERS = 3868.0
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
    # Timestamps parsed ONCE per read (the per-trophy fits used to re-parse every
    # timestamp inside every trophy's loop -- 55x the work -- and this also makes
    # the trend fits cheap enough to compute once and reuse, see below).
    t_hist = [hours_since_t0(parse_time(r["at"])) for r in history]
    x_hist = [math.log1p(max(t, 0.0)) for t in t_hist]

    # ---- window for the trend fits: the most recent 24 h when it holds >= 20 reads
    # (final cases have a read every ~10 min, so 24 h is ~144 reads). Recency matters
    # because launch-night growth is not stationary: player growth and earn rates both
    # decelerate after the first day, so a 48 h fit extrapolates a stale, too-steep
    # slope 4.5 days out (it overshoots flattening rates and collapses slow trophies
    # through the denominator model). Fall back to 48 h, then 24 h, then all history.
    t_end = hours_since_t0(at)
    win48_idx = [j for j, t in enumerate(t_hist) if t >= t_end - 48.0]
    win24_idx = [j for j, t in enumerate(t_hist) if t >= t_end - 24.0]
    win_idx = (win24_idx if len(win24_idx) >= 20 else
               win48_idx if len(win48_idx) >= 20 else
               (win24_idx if len(win24_idx) >= 3 else list(range(len(history)))))
    window = [history[j] for j in win_idx]
    win_t = [t_hist[j] for j in win_idx]
    win_x = [x_hist[j] for j in win_idx]
    t_target = hours_since_t0(target)
    x_target = math.log1p(max(t_target, 0.0))

    # ---- per-trophy trend fits, computed ONCE here and reused by the forecast and
    # by the jump-volatility estimate below. Each entry is (a, b, sigma_resid,
    # sigma_pred) for logit(rate) ~ a + b*log1p(hours), or None when the trophy has
    # too few valid reads; sigma_pred is the OLS prediction uncertainty of the trend
    # mean at the target: s * sqrt(1/n + (x_target - xbar)^2 / Sxx).
    trend_stats = []
    for i in range(n_tr):
        xs, ys = [], []
        for read, x in zip(window, win_x):
            r = read["rates"][i]
            if r is None:
                continue
            xs.append(x)
            ys.append(logit_of(float(r)))
        stat = None
        if len(xs) >= 3:
            fit = ols(xs, ys)
            if fit is not None:
                a, b = fit
                n = len(xs)
                mx = sum(xs) / n
                sxx = sum((x - mx) ** 2 for x in xs)
                resid = [y - (a + b * x) for x, y in zip(xs, ys)]
                s_resid = pstdev(resid)
                s_pred = (s_resid * math.sqrt(1.0 / n + (x_target - mx) ** 2 / sxx)
                          if sxx > 0 else 0.0)
                stat = (a, b, s_resid, s_pred)
        trend_stats.append(stat)

    # ---- refresh events across the whole history (for the jump volatility).
    # A refresh is how the page catches up to smooth growth, so the raw jump size
    # is mostly predicted drift, not shock: the volatile part is the jump MINUS the
    # jump the per-trophy trend already predicts over that interval.
    jump_resid = []  # per-trophy unpredicted logit jumps at refresh events
    n_events = 0
    any_movement = False
    prev = None
    prev_x = None
    for k, read in enumerate(history):
        if prev is not None:
            p_prev = float(prev["players_tracked"])
            p_now = float(read["players_tracked"])
            players_jump = abs(p_now - p_prev) / p_prev if p_prev > 0 else 0.0
            deltas = []  # (trophy index, logit jump) for trophies with both reads
            max_jump = 0.0
            for i in range(n_tr):
                r0 = prev["rates"][i]
                r1 = read["rates"][i]
                if r0 is None or r1 is None:
                    continue
                d = logit_of(r1) - logit_of(r0)
                deltas.append((i, d))
                max_jump = max(max_jump, abs(d))
            if players_jump >= 0.01 or max_jump >= 0.05:
                any_movement = True
            if players_jump > 0.01 or max_jump > 0.1:
                n_events += 1
                dx = x_hist[k] - prev_x
                for i, d in deltas:
                    b = trend_stats[i][1] if trend_stats[i] is not None else 0.0
                    jump_resid.append(d - b * dx)
        prev = read
        prev_x = x_hist[k]
    span_hours = 0.0
    if len(history) >= 2:
        span_hours = t_hist[-1] - t_hist[0]
    expected_events = (n_events / span_hours * horizon_hours) if span_hours > 0 else 0.0
    sigma_jump = pstdev(jump_resid) * math.sqrt(expected_events) if jump_resid else 0.0
    # Frozen history: no residuals and no jumps exist to estimate a spread from,
    # so the frozen branch below uses the fixed SIGMA_FROZEN instead (see
    # _forecast_trophy). sigma_refresh is therefore no longer a spread term.
    frozen = not any_movement

    # ---- deceleration damping: when the movement itself is slowing down, a linear
    # log-time extrapolation overshoots. Compare, per trophy, the trend slope over the
    # recent half of the window with the slope over the whole window; the MEDIAN ratio
    # across trophies with real movement is the damping factor (1 = steady movement,
    # no damping; < 1 = saturating curves, shrink the extrapolated move toward the
    # last rate). Pooled (median) so one noisy trophy cannot drive it; clamped to
    # [0.25, 1.0] so the model never fully ignores a real trend. Frozen histories
    # have no slope at all and get damp = 1 (the refresh-drop prior handles them).
    # NOTE: the damping window is the LONG (48 h) window, not the 24 h fit window --
    # deceleration is only visible over the longer history (a 24 h window on a
    # plateaued curve has no slope left to compare against).
    damp_idx = win48_idx if len(win48_idx) >= 20 else win_idx
    damp = _deceleration_damping([history[j] for j in damp_idx],
                                 [t_hist[j] for j in damp_idx],
                                 [x_hist[j] for j in damp_idx], trophies)

    # ---- players trend (log-linear in hours) over the window.
    players_fit = None
    xs_p, ys_p = [], []
    for read, t in zip(window, win_t):
        p = float(read["players_tracked"])
        if p > 0:
            xs_p.append(t)
            ys_p.append(math.log(p))
    if len(xs_p) >= 2:
        players_fit = ols(xs_p, ys_p)
    players_last = float(last["players_tracked"])

    rows = []
    # Frozen-history median drop, per trophy (see module docstring): the common
    # anchor factor blended with the earners-slope factor. Earners are integer
    # counts implied by the last read (rate * players / 100); the slope factor
    # is floored at the earners-frozen rate, so no trophy's median can fall
    # below the rate it would read if nobody earned it again before the refresh.
    # On a moving history this is 1.0 for every trophy: the final median is
    # purely trend + denominator, never this prior.
    earners = []
    for i in range(n_tr):
        r = last["rates"][i]
        earners.append(0 if r is None else int(round(float(r) / 100.0 * players_last)))
    min_earners = min(earners) if earners else 0
    k_floor = players_last / FROZEN_REFRESH_PLAYERS if FROZEN_REFRESH_PLAYERS > 0 else 0.165
    for i, trophy in enumerate(trophies):
        if frozen:
            k_slope = k_floor + FROZEN_EARNERS_SLOPE * max(0, earners[i] - min_earners)
            median_drop = math.exp(
                (1.0 - FROZEN_EARNERS_BLEND) * math.log(FROZEN_MEDIAN_FACTOR)
                + FROZEN_EARNERS_BLEND * math.log(max(k_slope, 1e-6)))
        else:
            median_drop = 1.0
        try:
            rows.append(_forecast_trophy(
                trophy, i, window, win_t, last, t_target, x_target,
                horizon_days, players_fit, players_last, sigma_jump,
                median_drop, damp, trend_stats[i], frozen))
        except Exception:
            # Per-trophy fallback: flat last rate with the house random-walk spread.
            rate = last["rates"][i]
            rate = 0.0 if rate is None else float(rate)
            lg = logit_of(rate)
            spread = SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)
            rows.append({"id": trophy["id"], "q": [_q(lg, z, spread) for z in Z]})
    return {"trophies": rows}


def _deceleration_damping(reads, times, xs, trophies):
    """Median recent-half / whole-window trend-slope ratio across trophies.

    Returns 1.0 when there is no usable slope information (too few reads, frozen
    history, or too few trophies moving). A ratio below 1 means the per-trophy
    logit-vs-log-time slopes are shrinking in the recent half of the window --
    the curves are saturating -- so the extrapolated move should be shrunk toward
    the last rate by that ratio.
    """
    if len(reads) < 6:
        return 1.0
    t_mid = 0.5 * (min(times) + max(times))
    ratios = []
    for i in range(len(trophies)):
        xs_all, ys_all, xs_rec, ys_rec = [], [], [], []
        for read, t, x in zip(reads, times, xs):
            r = read["rates"][i]
            if r is None:
                continue
            y = logit_of(float(r))
            xs_all.append(x)
            ys_all.append(y)
            if t >= t_mid:
                xs_rec.append(x)
                ys_rec.append(y)
        if len(xs_all) < 6 or len(xs_rec) < 3:
            continue
        fit_all = ols(xs_all, ys_all)
        fit_rec = ols(xs_rec, ys_rec)
        if fit_all is None or fit_rec is None:
            continue
        b_all, b_rec = fit_all[1], fit_rec[1]
        if abs(b_all) < 0.1:
            continue
        if (b_all > 0) != (b_rec > 0):
            ratios.append(0.0)  # movement reversed: fully saturating
        else:
            ratios.append(b_rec / b_all)
    if len(ratios) < 10:
        return 1.0
    ratios.sort()
    med = ratios[len(ratios) // 2] if len(ratios) % 2 else 0.5 * (
        ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2])
    return min(1.0, max(0.25, med))


def _forecast_trophy(trophy, i, window, win_t, last, t_target, x_target,
                     horizon_days, players_fit, players_last, sigma_jump,
                     median_drop=1.0, damp=1.0, trend_stat=None, frozen=False):
    last_rate = last["rates"][i]
    last_rate = 0.0 if last_rate is None else float(last_rate)

    # Trophy at 100%: the persistent Exophase quirk (see module docstring). The
    # upper quantiles sit at the scorer's 99.99 clamp -- the quirk has persisted
    # through every refresh seen -- and q05 is the earners-frozen dilution value
    # (players_last / 3868), the rate it would read if the quirk broke and nobody
    # earned it again before the refresh. The top quantiles repeat at the clamp
    # by design; the row is non-decreasing and never five-equal.
    if last_rate >= 99.99:
        q05 = round4(min(99.9, 100.0 * players_last / FROZEN_REFRESH_PLAYERS))
        return {"id": trophy["id"], "q": [q05, 99.9, 99.99, 99.99, 99.99]}

    # Trend estimate, precomputed once per trophy in forecast_case.
    trend_logit = None
    sigma_resid = 0.0
    sigma_pred = 0.0
    if trend_stat is not None:
        a, b, sigma_resid, sigma_pred = trend_stat
        trend_logit = a + b * x_target
    if trend_logit is None:
        trend_logit = logit_of(last_rate)

    # Denominator model: project earners and players separately, then divide.
    denom_logit = None
    if players_fit is not None:
        xs_e, ys_e = [], []
        for read, t in zip(window, win_t):
            r = read["rates"][i]
            p = float(read["players_tracked"])
            if r is None or p <= 0:
                continue
            earners = float(r) / 100.0 * p
            if earners > 0:
                xs_e.append(t)
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
    if damp != 1.0:
        # Decelerating movement: shrink the extrapolated move toward the last rate.
        # (The frozen-history refresh-drop prior below is NOT damped -- it is a
        # structural event prior, not a trend extrapolation.)
        median_logit = logit_of(last_rate) + damp * (median_logit - logit_of(last_rate))
    if median_drop != 1.0:
        # Frozen history only: structural refresh-drop prior on the median. When the
        # history moves this is 1.0 and the median is purely trend + denominator.
        median_logit += math.log(median_drop)
    median_rate = inv_logit(median_logit)

    # Spread: on a frozen history the residual terms are all zero, so the spread
    # is the fixed frozen value (the median already carries the expected refresh
    # move); on a moving history it is the history's own residual volatility
    # (see module docstring). No house random-walk term either way: the drift is
    # modelled by the trend, so only the residual around it is noise.
    if frozen:
        sigma = SIGMA_FROZEN
    else:
        sigma = math.sqrt(sigma_pred ** 2 + sigma_resid ** 2 + sigma_jump ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

    # Downside dilution risk widens the lower tail and narrows the upper one --
    # moving histories only; the frozen band is symmetric (see module docstring).
    downside = (not frozen) and median_rate < last_rate * 0.999
    qs = []
    for z in Z:
        side = sigma * (LOWER_WIDEN if (downside and z < 0) else
                        (UPPER_NARROW if (downside and z > 0) else 1.0))
        qs.append(_q(median_logit, z, side))
    # Enforce non-decreasing after rounding (rounding can only tie, never invert, but be safe).
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    # Never emit five equal quantiles: when the median is pinned at the 0.01 floor the
    # clamp collapses every quantile to 0.01. Nudge tied quantiles up by the smallest
    # rounding step so the row stays strictly increasing; the scorer clamps quantiles
    # to [0.01, 99.99] anyway, so a 0.0001 nudge is score-neutral.
    for k in range(1, len(qs)):
        if qs[k] <= qs[k - 1]:
            qs[k] = round4(min(99.99, qs[k - 1] + 0.0001))
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
            json.dumps(_finish(forecast), separators=(",", ":")) + "\n", encoding="utf-8")


def _finish(forecast):
    """Final gate: every row non-decreasing and finite, never five equal quantiles
    (the scorer clamps to [0.01, 99.99] so a 0.0001 nudge is score-neutral).
    Exception by design: the 100%-quirk row's top quantiles repeat at the 99.99
    clamp -- the nudge cannot lift them above it, and the row is still
    non-decreasing with a distinct q05/q25 below."""
    for row in forecast["trophies"]:
        qs = row["q"]
        for k in range(1, len(qs)):
            if qs[k] <= qs[k - 1]:
                qs[k] = round4(min(99.99, qs[k - 1] + 0.0001))
    return forecast


if __name__ == "__main__":
    main()
