"""Probability math: links (logit / probit), decimal-odds validation and de-vigging."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .config import DevigCfg
from .errors import OddsError

EPS = 1e-9
_ND = NormalDist()


def clip_prob(p: float, eps: float = EPS) -> float:
    return min(max(float(p), eps), 1.0 - eps)


def logit(p: float) -> float:
    p = clip_prob(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def probit(p: float) -> float:
    return _ND.inv_cdf(clip_prob(p))


def normal_cdf(x: float) -> float:
    return _ND.cdf(x)


# name -> (to_link, from_link, scale).  ``scale`` converts a shift expressed in
# LOGIT units (the unit all config parameters use) into the link's own units, so that
# switching the link does not silently change how aggressive the adjustment is.
LINKS = {
    "logit": (logit, sigmoid, 1.0),
    "probit": (probit, normal_cdf, 1.0 / 1.702),
}


def shift_probability(p: float, adj_logit_units: float, link: str = "logit") -> float:
    """Shift probability ``p`` by ``adj`` (logit units) in link space; always inside (0, 1)."""
    if adj_logit_units == 0.0:
        return p
    to_link, from_link, scale = LINKS[link]
    return clip_prob(from_link(to_link(p) + adj_logit_units * scale))


# ---------------------------------------------------------------------------------------
# Decimal odds -> fair probabilities
# ---------------------------------------------------------------------------------------
def parse_decimal_odds(x, cfg: DevigCfg, max_odds: float | None = None) -> float:
    if x is None or (isinstance(x, str) and not x.strip()):
        raise OddsError("missing_odds", "odds are missing")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise OddsError("invalid_odds", f"odds {x!r} are not numeric") from None
    if not math.isfinite(v):
        raise OddsError("missing_odds" if math.isnan(v) else "invalid_odds", f"odds {x!r} not finite")
    if v < cfg.min_decimal_odds:
        raise OddsError("invalid_odds", f"decimal odds {v} below minimum {cfg.min_decimal_odds}")
    hi = cfg.max_decimal_odds if max_odds is None else max_odds
    if v > hi:
        raise OddsError("invalid_odds", f"decimal odds {v} above maximum {hi}")
    return v


def devig_multiway(q: np.ndarray, method: str = "proportional") -> np.ndarray:
    """Turn raw implied probabilities ``q`` (sum >= 1) into fair probabilities summing to 1.

    proportional : p_i = q_i / sum(q)                    (simple normalisation)
    power        : p_i = q_i**k with k >= 1 s.t. sum = 1 (shrinks longshot over-pricing)
    """
    q = np.asarray(q, dtype=float)
    s = q.sum()
    if method == "proportional" or s <= 1.0 + 1e-12:
        return q / s
    if method == "power":
        f = lambda k: float(np.sum(q ** k)) - 1.0  # noqa: E731
        lo, hi = 1.0, 2.0
        while f(hi) > 0 and hi < 1e4:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        p = q ** (0.5 * (lo + hi))
        return p / p.sum()
    raise ValueError(f"unknown de-vig method {method!r}")


def devig_two_way(odds_home, odds_away, cfg: DevigCfg) -> tuple[float, float]:
    """Fair (home_prob, overround) from two decimal odds.  Raises ``OddsError`` on bad input.

    Returns ``(p_home, overround)``; ``p_away`` is by construction ``1 - p_home``.
    """
    oh = parse_decimal_odds(odds_home, cfg)
    oa = parse_decimal_odds(odds_away, cfg)
    q = np.array([1.0 / oh, 1.0 / oa])
    s = float(q.sum())
    if s < cfg.min_game_overround - 1e-12:
        raise OddsError("overround_too_low", f"implied total {s:.4f} < 1: arbitrage or data error")
    if s > cfg.max_game_overround:
        raise OddsError("overround_too_high", f"implied total {s:.4f} > {cfg.max_game_overround}")
    p = devig_multiway(q, cfg.game_method)
    p_home = float(p[0])
    if not (0.0 < p_home < 1.0):
        raise OddsError("invalid_odds", "de-vigged probability outside (0, 1)")
    return p_home, s
