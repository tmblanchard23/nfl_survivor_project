"""Gating of rating movement: which team-strength changes are 'meaningful'.

Movement is measured in rating (logit) units, which are directly game-log-odds units, so the
default sensitivity is 1.0.  The gate is applied per team, with separate up/down thresholds."""
from __future__ import annotations

import numpy as np

from .config import StrengthCfg


def strength_shift(m: float, cfg: StrengthCfg) -> float:
    """Gate + scale one team's centered rating movement into a *strength shift* (game
    log-odds units).  Movement inside the dead-zone (|m| <= threshold) contributes exactly 0.

    gating = "soft": only the part of the move BEYOND the threshold counts (continuous).
    gating = "hard": the full move counts once the threshold is exceeded (has a jump).
    """
    if m > 0:
        thr, sens, mag = cfg.threshold_up, cfg.sensitivity_up, m
    elif m < 0:
        thr, sens, mag = cfg.threshold_down, cfg.sensitivity_down, -m
    else:
        return 0.0
    if mag <= thr:
        return 0.0
    eff = mag - thr if cfg.gating == "soft" else mag
    return float(np.sign(m) * sens * eff)
