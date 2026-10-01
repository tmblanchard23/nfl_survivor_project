import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from nfl_prob_grid.config import RatingCfg  # noqa: E402
from nfl_prob_grid.errors import InputError  # noqa: E402
from nfl_prob_grid.ingest import ingest_win_totals  # noqa: E402
from nfl_prob_grid.ratings import (N, TEAMS, RatingHistory, advance_ratings, build_baseline,  # noqa: E402
                                   expected_wins, kalman_update, normalize_win_totals,
                                   solve_win_total_ratings, team_idx)
from nfl_prob_grid.teams import TeamNormalizer  # noqa: E402

T = pd.Timestamp


@pytest.fixture()
def sched(init_state):
    m, _ = init_state
    return team_idx(m.home_team), team_idx(m.away_team), m


# ---------------------------------------------------------------- win totals -> ratings
def test_solve_recovers_ratings_from_their_own_expected_wins(sched):
    hi, ai, _ = sched
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.6, N); r -= r.mean()
    hfa = np.full(len(hi), 0.25)
    target, _ = expected_wins(r, hi, ai, hfa)
    got = solve_win_total_ratings(target, hi, ai, hfa)
    assert np.allclose(got, r, atol=1e-7)
    assert np.allclose(expected_wins(got, hi, ai, hfa)[0], target, atol=1e-8)


def test_solve_is_opponent_and_home_adjusted(sched):
    """Same win total, tougher schedule => higher rating (naive win% -> logit would ignore this)."""
    hi, ai, _ = sched
    r_true = np.random.default_rng(1).normal(0, 0.6, N); r_true -= r_true.mean()
    hfa = np.full(len(hi), 0.25)
    target, _ = expected_wins(r_true, hi, ai, hfa)
    naive = np.log((target / 17) / (1 - target / 17)); naive -= naive.mean()
    solved = solve_win_total_ratings(target, hi, ai, hfa)
    assert np.abs(solved - r_true).max() < 1e-6 and np.abs(naive - r_true).max() > 0.02


def test_normalization_removes_vig_so_totals_sum_to_games():
    t = np.full(32, 8.5) + 0.2
    for mode in ("additive", "proportional"):
        out, _ = normalize_win_totals(t, 272, mode)
        assert out.sum() == pytest.approx(272)
    assert normalize_win_totals(t, 272, "none")[0].sum() == pytest.approx(t.sum())


def test_neutral_games_have_no_home_field(sched):
    hi, ai, _ = sched
    r = np.zeros(N)
    hfa = np.full(len(hi), 0.3)
    hfa[0] = 0.0
    e, p = expected_wins(r, hi, ai, hfa)
    assert p[0] == pytest.approx(0.5) and p[1] > 0.5


# ---------------------------------------------------------------- win-total table
def _wt(demo_df, **edits):
    df = demo_df.copy()
    for k, v in edits.items():
        df[k] = v
    return df


def test_win_total_table_validation(cfg, demo):
    raw = pd.read_csv(demo["dir"] / "win_totals.csv", dtype=str, keep_default_na=False)
    nz = TeamNormalizer()
    ok, issues, ts = ingest_win_totals(raw, cfg, nz, n_games=272)
    assert len(ok) == 32 and ts is None
    for bad in (raw.iloc[:-2], pd.concat([raw, raw.iloc[[0]]]), _wt(raw, **{"Win Total": "abc"}),
                _wt(raw, **{"Win Total": "30"})):
        with pytest.raises(InputError):
            ingest_win_totals(bad, cfg, nz, n_games=272)
    unk = raw.copy(); unk.loc[0, "Team"] = "Gotham Rogues"
    with pytest.raises(InputError, match="unknown team"):
        ingest_win_totals(unk, cfg, nz, n_games=272)


def test_over_under_price_shades_mean_in_the_right_direction(cfg, demo):
    raw = pd.read_csv(demo["dir"] / "win_totals.csv", dtype=str, keep_default_na=False)
    nz = TeamNormalizer()
    raw.loc[0, ["Over Odds", "Under Odds"]] = ["1.70", "2.15"]     # over favoured -> mean above the line
    raw.loc[1, ["Over Odds", "Under Odds"]] = ["2.15", "1.70"]     # under favoured -> mean below
    raw.loc[2, ["Over Odds", "Under Odds"]] = ["1.91", "1.91"]     # even -> no shade
    df, _, _ = ingest_win_totals(raw, cfg, nz, n_games=272)
    d = df.set_index("team")
    t0, t1, t2 = (nz(raw.loc[i, "Team"]) for i in range(3))
    assert d.at[t0, "expected_wins"] > d.at[t0, "win_total"]
    assert d.at[t1, "expected_wins"] < d.at[t1, "win_total"]
    assert d.at[t2, "expected_wins"] == pytest.approx(d.at[t2, "win_total"], abs=1e-9)
    cfg.rating.use_over_under_price = False
    df2, _, _ = ingest_win_totals(raw, cfg, nz, n_games=272)
    assert (df2["price_shift"] == 0).all()


# ---------------------------------------------------------------- Kalman properties
def test_pair_observation_splits_change_by_prior_variance():
    r, cov = np.zeros(N), 0.02 * np.eye(N)
    hi, ai = np.array([0]), np.array([1])
    r1, cov1, _ = kalman_update(r, cov, hi, ai, np.array([0.4]), np.array([0.0225]))
    assert r1[0] > 0 > r1[1] and r1[0] == pytest.approx(-r1[1])          # equal priors -> equal split
    assert abs(r1[2]) < 1e-12 and r1.sum() == pytest.approx(0, abs=1e-12)
    cov2 = 0.02 * np.eye(N); cov2[0, 0] = 0.08                            # team 0 far more uncertain
    r2, _, _ = kalman_update(r, cov2, hi, ai, np.array([0.4]), np.array([0.0225]))
    assert r2[0] > 3 * abs(r2[1])                                         # ... so it takes most of the change
    assert cov1[0, 0] < 0.02 and cov1[0, 1] > 0     # variance shrinks; errors become POSITIVELY correlated:
    #   a difference is known, the sum is not => a single game cannot say WHO moved
    assert np.linalg.eigvalsh(cov1).min() > -1e-12 and np.allclose(cov1, cov1.T)


def test_evidence_accumulates_across_games_to_identify_who_moved():
    """A appears to beat every opponent by more than expected; opponents are otherwise unremarkable."""
    r, cov = np.zeros(N), 0.02 * np.eye(N)
    hi, ai = np.array([0, 0, 0]), np.array([1, 2, 3])
    single, _, _ = kalman_update(r, cov, hi[:1], ai[:1], np.array([0.4]), np.array([0.0225]))
    multi, _, _ = kalman_update(r, cov, hi, ai, np.array([0.4] * 3), np.array([0.0225] * 3))
    assert multi[0] > single[0] and abs(multi[1]) < abs(single[1])       # more evidence -> attributed to A


def test_advance_without_new_lines_grows_uncertainty_but_not_means(init_state, cfg):
    _, hist = init_state
    h = hist.copy()
    before = h.ratings(0).copy()
    sid, stats = advance_ratings(h, [], T("2026-10-01T00:00:00Z"), cfg.rating, 5)
    assert sid == 1 and stats["n_obs"] == 0 and stats["max_abs_rating_change"] == 0
    assert np.allclose(h.ratings(1).to_numpy(), before.to_numpy())
    sd = h.df.groupby("snapshot_id")["sd"].mean()
    assert sd[1] > sd[0]                                                  # process noise inflated uncertainty


def test_lookahead_lines_move_ratings_less(init_state, cfg):
    _, hist = init_state
    base = hist.ratings(0)
    a, b = TEAMS[0], TEAMS[1]
    p_hi = float(1 / (1 + np.exp(-(base[a] - base[b] + hist.hfa + 0.6))))
    moves = {}
    for week in (6, 16):
        h = hist.copy()
        advance_ratings(h, [{"home": a, "away": b, "p_home": p_hi, "week": week, "neutral": False}],
                        T("2026-10-13T15:00:00Z"), cfg.rating, 6)
        moves[week] = h.ratings(1)[a] - base[a]
    assert moves[6] > moves[16] > 0


def test_neutral_flag_removes_home_field_from_the_observation(init_state, cfg):
    _, hist = init_state
    a, b = TEAMS[2], TEAMS[3]
    p = float(1 / (1 + np.exp(-(hist.ratings(0)[a] - hist.ratings(0)[b]))))   # no hfa in this line
    out = {}
    for neutral in (True, False):
        h = hist.copy()
        advance_ratings(h, [{"home": a, "away": b, "p_home": p, "week": 6, "neutral": neutral}],
                        T("2026-10-13T15:00:00Z"), cfg.rating, 6)
        out[neutral] = h.ratings(1)[a] - hist.ratings(0)[a]
    assert abs(out[True]) < 1e-9 and out[False] < -0.02                       # mis-specified hfa => fake movement


# ---------------------------------------------------------------- baseline construction
def _baseline(cfg_r, init_state, demo):
    from make_demo_data import INIT_AS_OF  # noqa: F401
    m, _ = init_state
    nz = TeamNormalizer()
    raw = pd.read_csv(demo["dir"] / "win_totals.csv", dtype=str, keep_default_na=False)
    wins, _, _ = ingest_win_totals(raw, __import__("nfl_prob_grid").load_config(
        Path(__file__).resolve().parents[1] / "config.toml"), nz, n_games=len(m))
    hi, ai = team_idx(m.home_team), team_idx(m.away_team)
    pre = np.log(m.home_win_prob / (1 - m.home_win_prob)).to_numpy()
    return build_baseline(wins, hi, ai, pre, np.zeros(len(m), bool), cfg_r)


def test_pure_win_total_baseline_is_exactly_the_win_total_solve(init_state, demo):
    b = _baseline(RatingCfg(reconcile_preseason_lines=False), init_state, demo)
    assert np.allclose(b.ratings, b.anchor["win_total_rating"]) and b.diagnostics["win_total_weight_in_baseline"] == 1.0
    assert np.allclose(np.diag(b.cov), 0.15 ** 2)


def test_win_total_weight_is_monotone_in_preseason_line_noise(init_state, demo):
    w = [_baseline(RatingCfg(preseason_line_sd=s), init_state, demo).diagnostics["win_total_weight_in_baseline"]
         for s in (0.1, 0.3, 0.6, 1.5, 1e6)]
    assert all(a < b for a, b in zip(w, w[1:])) and w[0] < 0.1 and w[-1] > 0.999


def test_baseline_diagnostics_and_yardstick_independent_of_line_noise(init_state, demo):
    a = _baseline(RatingCfg(preseason_line_sd=0.3), init_state, demo)
    b = _baseline(RatingCfg(preseason_line_sd=1e6), init_state, demo)
    assert a.diagnostics["rms_gap_win_totals_vs_preseason_lines_logit"] == b.diagnostics["rms_gap_win_totals_vs_preseason_lines_logit"]
    assert abs(a.ratings.mean()) < 1e-12 and a.diagnostics["hfa_logit"] == pytest.approx(0.25, abs=0.05)


# ---------------------------------------------------------------- history
def test_history_roundtrip_is_exact(init_state, tmp_path):
    _, hist = init_state
    h = hist.copy()
    advance_ratings(h, [{"home": TEAMS[0], "away": TEAMS[1], "p_home": 0.62, "week": 3, "neutral": False}],
                    T("2026-09-15T15:00:00Z"), RatingCfg(), 2)
    h.save(tmp_path / "h.csv", tmp_path / "s.json")
    g = RatingHistory.load(tmp_path / "h.csv", tmp_path / "s.json")
    assert (g.df["rating"].to_numpy() == h.df["rating"].to_numpy()).all()
    assert (g.cov == h.cov).all() and g.hfa == h.hfa and g.latest_id() == 1
    assert g.anchor is not None and list(g.anchor["team"]) == TEAMS


def test_movement_is_centred_on_the_league_median(init_state, cfg):
    _, hist = init_state
    h = hist.copy()
    obs = [{"home": TEAMS[i], "away": TEAMS[i + 1], "p_home": 0.7, "week": 3, "neutral": False} for i in range(0, 30, 2)]
    advance_ratings(h, obs, T("2026-09-15T15:00:00Z"), cfg.rating, 2)
    mv = h.movement(0, 1, cfg.strength)
    assert abs(mv.median()) < 1e-12


def test_hfa_estimate_is_not_fooled_by_schedule_imbalance(init_state, demo):
    """Demo schedule (seed 11) has mean(r_home - r_away) = -0.15; the naive mean of home logits
    would read 0.10.  The regression intercept recovers the true 0.25."""
    m, _ = init_state
    hi, ai = team_idx(m.home_team), team_idx(m.away_team)
    y = np.log(m.home_win_prob / (1 - m.home_win_prob)).to_numpy()
    from nfl_prob_grid.ratings import estimate_hfa
    assert abs(y.mean() - 0.25) > 0.1                                   # naive estimate is biased here
    assert estimate_hfa(y, hi, ai, np.zeros(len(y), bool)) == pytest.approx(0.25, abs=0.04)
    neutral = np.zeros(len(y), bool); neutral[:10] = True               # neutral games carry no hfa
    assert estimate_hfa(y, hi, ai, neutral) == pytest.approx(0.25, abs=0.06)
