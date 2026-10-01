import math

import numpy as np
import pytest

from nfl_prob_grid.adjustment import age_weight, cap_adjustment, compute_adjustment
from nfl_prob_grid.config import AdjustCfg, AgingCfg, DevigCfg, StrengthCfg
from nfl_prob_grid.errors import OddsError
from nfl_prob_grid.probability import devig_multiway, devig_two_way, shift_probability
from nfl_prob_grid.strength import strength_shift
from nfl_prob_grid.teams import TeamNormalizer

DV = DevigCfg()


# ---------------------------------------------------------------- de-vig
def test_devig_symmetric_and_sums_to_one():
    p, over = devig_two_way(1.91, 1.91, DV)
    assert p == pytest.approx(0.5) and over == pytest.approx(2 / 1.91)


def test_devig_known_value():
    p, _ = devig_two_way(1.55, 2.50, DV)
    q = np.array([1 / 1.55, 1 / 2.5])
    assert p == pytest.approx(q[0] / q.sum())


def test_power_devig_sums_to_one_and_favours_favourite_more():
    q = np.array([0.60, 0.50])
    prop, power = devig_multiway(q, "proportional"), devig_multiway(q, "power")
    assert power.sum() == pytest.approx(1.0) and prop.sum() == pytest.approx(1.0)
    assert power[0] > prop[0]


@pytest.mark.parametrize("bad, code", [
    ((1.0, 2.0), "invalid_odds"), ((0.5, 2.0), "invalid_odds"), (("abc", 2.0), "invalid_odds"),
    ((None, 2.0), "missing_odds"), (("", 2.0), "missing_odds"), ((float("nan"), 2.0), "missing_odds"),
    ((5000, 1.5), "invalid_odds"), ((2.1, 2.1), "overround_too_low"), ((1.3, 1.3), "overround_too_high"),
])
def test_bad_odds_rejected_with_code(bad, code):
    with pytest.raises(OddsError) as e:
        devig_two_way(*bad, DV)
    assert e.value.code == code


# ---------------------------------------------------------------- link / bounds
@pytest.mark.parametrize("link", ["logit", "probit"])
def test_shift_bounded_and_antisymmetric(link):
    assert 0 < shift_probability(0.9999, 6.0, link) < 1
    assert 0 < shift_probability(0.0001, -6.0, link) < 1
    for p in (0.1, 0.5, 0.83):
        assert shift_probability(1 - p, -0.3, link) == pytest.approx(1 - shift_probability(p, 0.3, link))


def test_effect_shrinks_near_boundaries():
    move_mid = shift_probability(0.5, 0.3) - 0.5
    move_fav = shift_probability(0.95, 0.3) - 0.95
    assert 0 < move_fav < move_mid


# ---------------------------------------------------------------- aging curves
@pytest.mark.parametrize("curve", ["hyperbolic", "linear", "exponential", "logistic", "step"])
def test_aging_curve_properties(curve):
    a = AgingCfg(curve=curve)
    assert age_weight(0, a) == 0 and age_weight(a.start_age_weeks, a) == 0
    ws = [age_weight(x / 2, a) for x in range(0, 60)]
    assert all(0 <= w <= a.max_weight + 1e-12 for w in ws)
    assert all(b >= a_ - 1e-12 for a_, b in zip(ws, ws[1:]))          # non-decreasing
    assert age_weight(a.start_age_weeks + 0.01, a) < 0.05 or curve == "step"   # continuous at start


# ---------------------------------------------------------------- gating
def test_movement_inside_deadzone_is_ignored_both_directions():
    s = StrengthCfg(threshold_up=0.12)
    assert strength_shift(0.11, s) == 0 and strength_shift(-0.11, s) == 0
    d = StrengthCfg()                                                    # backtested default gate
    assert strength_shift(0.019, d) == 0 and strength_shift(0.05, d) == pytest.approx(0.03)


def test_soft_vs_hard_gating():
    soft, hard = StrengthCfg(gating="soft", threshold_up=0.12), StrengthCfg(gating="hard", threshold_up=0.12)
    assert strength_shift(0.13, soft) == pytest.approx(0.01)            # continuous: only the excess counts
    assert strength_shift(0.13, hard) == pytest.approx(0.13)            # jumps at the threshold


def test_asymmetric_thresholds():
    s = StrengthCfg(symmetric=False, threshold_up=0.4, threshold_down=0.1)
    assert strength_shift(0.3, s) == 0 and strength_shift(-0.3, s) < 0


# ---------------------------------------------------------------- spec cases A-E
S, A = StrengthCfg(), AdjustCfg()


def adj(h, a, w=0.5):
    return compute_adjustment(h, a, w, S, A).adj_logit


def test_case_A_B_single_team_moves():
    assert adj(0.8, 0.0) > 0 and adj(-0.8, 0.0) < 0
    assert adj(0.0, 0.8) < 0 and adj(0.0, -0.8) > 0          # mirrored for the away team


def test_case_C_both_improve_relative_movement_decides():
    assert adj(0.8, 0.8) == 0
    assert adj(1.0, 0.5) > 0 and adj(0.5, 1.0) < 0


def test_case_D_opposite_moves_larger_than_one_sided():
    assert adj(0.8, -0.8) > adj(0.8, 0.0) > 0


def test_case_E_both_decline_relative_magnitude_decides():
    assert adj(-0.8, -0.8) == 0
    assert adj(-1.0, -0.5) < 0 and adj(-0.5, -1.0) > 0


def test_cap_bounds_extreme_input_and_zero_weight_gives_zero():
    for mode in ("tanh", "clip"):
        c = AdjustCfg(cap=0.3, cap_mode=mode)
        assert abs(cap_adjustment(50.0, c)) <= 0.3 and abs(cap_adjustment(-50.0, c)) <= 0.3
    assert compute_adjustment(5, -5, 0.0, S, A).adj_logit == 0.0


def test_team_normalizer():
    nz = TeamNormalizer({"Niners": "SF"})
    assert [nz(x) for x in ["Kansas City Chiefs", "chiefs", "KC", "Niners", "Tampa Bay", "St. Louis Rams"]] \
        == ["KC", "KC", "KC", "SF", "TB", "LAR"]
    assert nz("New York") is None and nz("Los Angeles") is None and nz("Nonsense FC") is None


def test_backtested_defaults_are_pinned():
    """Guard the real-data calibration (backtest/BACKTEST_RESULTS.md) against accidental edits."""
    from pathlib import Path
    from nfl_prob_grid import load_config
    c = load_config(Path(__file__).resolve().parents[1] / "config.toml")
    assert (c.aging.curve, c.aging.start_age_weeks, c.aging.half_weight_weeks, c.aging.max_weight) == ("hyperbolic", 3.0, 0.5, 1.0)
    assert (c.strength.threshold_up, c.strength.gating, c.strength.sensitivity_up) == (0.02, "soft", 1.0)
    assert (c.adjust.cap, c.adjust.cap_mode) == (1.0, "tanh")
    assert (c.rating.prior_sd, c.rating.process_sd_per_week, c.rating.fresh_line_sd) == (0.15, 0.08, 0.15)
    w = [age_weight(a, c.aging) for a in (3.0, 3.7, 5.0, 7.0, 10.0)]
    assert w[0] == 0 and [round(x, 2) for x in w[1:]] == [0.58, 0.8, 0.89, 0.93]
