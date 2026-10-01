"""Age weighting and the matchup-level strength adjustment.

    w      = age_weight(line_age)                      in [0, max_weight];  0 up to start_age
    delta  = shift(home_move) - shift(away_move)       game log-odds units, home perspective
    adj    = softcap(w * delta)                        final log-odds shift actually applied
    p_home = link^-1( link(anchor_home) + adj )

Because the shift lives in log-odds space and is *antisymmetric* between the two teams,
P(home) + P(away) == 1 by construction, probabilities can never leave (0, 1), and the effect
shrinks automatically near 0% / 100%.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import AdjustCfg, AgingCfg, StrengthCfg
from .probability import sigmoid
from .strength import strength_shift


def age_weight(age_weeks: float, a: AgingCfg) -> float:
    """Influence of the strength-movement signal for a Vegas line of the given age.

    hyperbolic (default): w = max * x / (x + h),  x = age - start.  This is the Kalman gain of
        a random-walk strength model: stale-line variance grows ~linearly with age while the
        Super Bowl signal has fixed noise, so the optimal blend weight saturates.
    """
    x = age_weeks - a.start_age_weeks
    if x <= 0:
        return 0.0
    c = a.curve
    if c == "hyperbolic":
        raw = x / (x + a.half_weight_weeks)
    elif c == "linear":
        raw = min(x / a.full_weight_weeks, 1.0)
    elif c == "exponential":
        raw = 1.0 - math.exp(-a.rate * x)
    elif c == "logistic":
        k, mid = a.logistic_steepness, a.logistic_midpoint_weeks
        s, s0 = sigmoid(k * (age_weeks - mid)), sigmoid(k * (a.start_age_weeks - mid))
        raw = max((s - s0) / (1.0 - s0), 0.0)  # rescaled so it is exactly 0 at start_age
    elif c == "step":
        w = 0.0
        for step in sorted(a.steps, key=lambda d: d["age"]):
            if age_weeks >= step["age"]:
                w = float(step["weight"])
        return min(max(w, 0.0), a.max_weight)
    else:  # pragma: no cover - guarded by Config.validate
        raise ValueError(c)
    return a.max_weight * raw


def cap_adjustment(x: float, acfg: AdjustCfg) -> float:
    cap = acfg.cap
    if cap <= 0:
        return 0.0
    if acfg.cap_mode == "clip":
        return max(-cap, min(cap, x))
    return cap * math.tanh(x / cap)  # smooth: ~identity for small x, saturates at +/-cap


@dataclass
class Adjustment:
    home_shift: float
    away_shift: float
    delta: float      # unweighted matchup shift (home perspective)
    adj_logit: float  # weighted + capped, the value applied to the anchor


def compute_adjustment(move_home: float, move_away: float, weight: float,
                       scfg: StrengthCfg, acfg: AdjustCfg) -> Adjustment:
    hs, as_ = strength_shift(move_home, scfg), strength_shift(move_away, scfg)
    delta = hs - as_
    adj = cap_adjustment(weight * delta, acfg) if weight > 0 else 0.0
    return Adjustment(hs, as_, delta, adj)
