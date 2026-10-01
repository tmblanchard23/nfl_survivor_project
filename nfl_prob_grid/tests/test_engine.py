import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import (INIT_AS_OF, implied_lines, line_rows, pick_game, run_as_of,  # noqa: E402
                      team_playing_in)

from nfl_prob_grid import (apply_init, apply_refresh, build_grid, regenerate_grid, run_init,  # noqa: E402
                           run_refresh)
from nfl_prob_grid.config import load_config  # noqa: E402
from nfl_prob_grid.errors import InputError, StateError, ValidationError  # noqa: E402
from nfl_prob_grid.master import StateStore  # noqa: E402
from nfl_prob_grid.probability import devig_two_way, shift_probability  # noqa: E402
from nfl_prob_grid.tabular import read_table  # noqa: E402

T = pd.Timestamp
ROOT = Path(__file__).resolve().parents[1]
D2, AS_OF2 = "2026-09-15 13:00:00", T("2026-09-15T15:00:00Z")        # preseason lines ~3.7 weeks old
LATE, AS_OF_LATE = "2026-10-13 13:00:00", T("2026-10-13T15:00:00Z")  # preseason lines ~7.5 weeks old


def refresh(prev, hist, cfg, lines=None, week=2, as_of=AS_OF2, version=2, **kw):
    return apply_refresh(prev, hist, lines, cfg, version=version, current_week=week, as_of=as_of, **kw)


def row_of(master, gid):
    return master.set_index("game_id").loc[gid]


def two_teams_not_facing(m, weeks):
    cands = sorted(set.intersection(*[set(m[m.week == w].home_team) | set(m[m.week == w].away_team) for w in weeks]))
    for x in cands:
        for y in cands:
            if x != y and not ((m.week.isin(weeks)) & (((m.home_team == x) & (m.away_team == y)) |
                                                       ((m.home_team == y) & (m.away_team == x)))).any():
                return x, y
    raise AssertionError("no suitable pair")


# ============================================================ master table, baseline, grid
def test_init_builds_complete_consistent_master_and_baseline(init_state, cfg):
    m, hist = init_state
    assert len(m) == 272 and m["game_id"].is_unique
    assert (m["home_win_prob"] + m["away_win_prob"]).sub(1).abs().max() < 1e-12
    assert (m["prob_source"] == "VEGAS_PRESEASON").all() and not m["is_frozen"].any()
    assert (m["home_win_prob"] == m["original_vegas_home_prob"]).all()
    assert (m["anchor_rating_snapshot_id"] == 0).all()
    grid = build_grid(m, cfg)
    assert grid.shape == (32, 18) and (grid.notna().sum(axis=1) == 17).all()
    assert hist.baseline_id() == 0 and len(hist.snapshots()) == 1 and abs(hist.ratings(0).mean()) < 1e-9
    assert len(hist.anchor) == 32 and hist.hfa == pytest.approx(0.25, abs=0.05)


def test_grid_is_a_pure_view_of_master(init_state, cfg):
    m, _ = init_state
    g = build_grid(m, cfg)
    for r in m.itertuples():
        assert g.at[r.home_team, f"Week {r.week}"] == pytest.approx(r.home_win_prob, abs=1e-6)
        assert g.at[r.away_team, f"Week {r.week}"] == pytest.approx(r.away_win_prob, abs=1e-6)


def test_neutral_site_games_are_honoured_and_unknown_ids_warned(demo, cfg):
    d = demo["dir"]
    m0 = apply_init(read_table(str(d / "initial_lines.csv")), read_table(str(d / "win_totals.csv")), cfg,
                    as_of=T(INIT_AS_OF), current_week=1).master
    cfg.rating.neutral_site_games = [m0.game_id.iloc[0], "2026-W99-XXX@YYY"]
    res = apply_init(read_table(str(d / "initial_lines.csv")), read_table(str(d / "win_totals.csv")), cfg,
                     as_of=T(INIT_AS_OF), current_week=1)
    assert res.report.details["baseline"]["neutral_games"] == 1
    assert any(i.code == "unknown_neutral_game" for i in res.report.issues)


# ============================================================ newer / older / partial refreshes
def test_only_genuinely_newer_lines_replace_probability(init_state, cfg):
    m, h = init_state
    g = pick_game(m, min_week=3)
    res = refresh(m, h, cfg, line_rows([(g.week, g.away_team, 2.10, g.home_team, 1.75)], D2))
    new = row_of(res.master, g.game_id)
    p, _ = devig_two_way(1.75, 2.10, cfg.devig)
    assert new.latest_vegas_home_prob == pytest.approx(p) and new.home_win_prob == pytest.approx(p)
    assert new.prob_source == "VEGAS_FRESH" and new.refresh_count == 1
    assert new.last_vegas_refresh_ts == T(D2, tz="UTC")
    assert new.original_vegas_home_prob == g.original_vegas_home_prob        # baseline preserved
    assert new.anchor_rating_snapshot_id == 1 and res.hist.latest_id() == 1  # line is inside snapshot 1
    others = res.master[res.master.game_id != g.game_id]
    assert (others.anchor_rating_snapshot_id == 0).all()
    assert res.report.counts["lines_refreshed"] == 1 and res.report.details["ratings"]["n_obs"] == 1


def test_same_and_older_timestamps_are_ignored(init_state, cfg):
    m, h = init_state
    g = pick_game(m, min_week=3)
    r1 = refresh(m, h, cfg, line_rows([(g.week, g.away_team, 2.10, g.home_team, 1.75)], D2))
    p1 = row_of(r1.master, g.game_id).home_win_prob
    args = (r1.master, r1.hist, cfg)
    same = refresh(*args, line_rows([(g.week, g.away_team, 3.0, g.home_team, 1.4)], D2), version=3)
    older = refresh(*args, line_rows([(g.week, g.away_team, 3.0, g.home_team, 1.4)], "2026-09-10 00:00:00"), version=3)
    for r, key in ((same, "ignored_same_timestamp"), (older, "ignored_older")):
        assert r.report.counts[key] == 1 and row_of(r.master, g.game_id).home_win_prob == p1
        assert r.report.details["ratings"]["n_obs"] == 0                       # ignored lines teach the filter nothing
    assert any(i.code == "same_timestamp_different_odds" for i in same.report.issues)
    assert any(i.code == "incoming_older_than_existing" for i in older.report.issues)


def test_irregular_table_sizes_and_scattered_weeks(init_state, cfg):
    m, h = init_state
    sched = m[m.week >= 4].sample(7, random_state=1)
    rows = [(r.week, r.away_team, 1.95, r.home_team, 1.95) for r in sched.itertuples()]
    res = refresh(m, h, cfg, line_rows(rows, D2))
    assert res.report.counts["lines_refreshed"] == 7 and res.report.details["ratings"]["n_obs"] == 7
    got = res.master.set_index("game_id").loc[sched["game_id"]]
    assert np.allclose(got["home_win_prob"], 0.5) and (got["prob_source"] == "VEGAS_FRESH").all()


# ============================================================ matching + row validation
def test_matching_is_independent_of_row_order_and_team_name_style(init_state, cfg):
    m, h = init_state
    gs = m[m.week == 6].head(3)
    rows = [(r.week, r.away_team, 2.0, r.home_team, 1.9) for r in gs.itertuples()]
    a = refresh(m, h, cfg, line_rows(rows, D2)).master
    b = refresh(m, h, cfg, line_rows(rows[::-1], D2)).master
    pd.testing.assert_frame_equal(a, b)
    from make_demo_data import CITY_ONLY, NICK
    g = m[(m.week == 6) & ~m.home_team.isin(['NYG', 'NYJ', 'LAR', 'LAC'])].iloc[0]
    styled = line_rows([(g.week, NICK[g.away_team], 2.0, CITY_ONLY[g.home_team], 1.9)], D2)
    res = refresh(m, h, cfg, styled)
    assert res.report.counts["lines_refreshed"] == 1 and row_of(res.master, g.game_id).refresh_count == 1


def test_bad_rows_are_rejected_individually_and_do_not_stop_good_rows(init_state, cfg):
    m, h = init_state
    good = pick_game(m, min_week=6)
    rows = pd.concat([line_rows([(good.week, good.away_team, 2.1, good.home_team, 1.75)], D2),
                      line_rows([(6, "Nonsense FC", 2.0, "KC", 1.9)], D2),
                      line_rows([(6, "KC", "abc", "DEN", 1.9)], D2),
                      line_rows([(6, "KC", 2.0, "DEN", "")], D2),
                      line_rows([(99, "KC", 2.0, "DEN", 1.9)], D2),
                      line_rows([(6, "KC", 2.0, "DEN", 1.9)], "not a date")], ignore_index=True)
    cfg.validation.max_reject_fraction = 0.9
    res = refresh(m, h, cfg, rows)
    codes = {r["code"] for r in res.report.rejected_rows}
    assert {"unknown_team", "invalid_odds", "missing_odds", "unexpected_week", "bad_timestamp"} <= codes
    assert res.report.counts["lines_refreshed"] == 1 and res.report.details["ratings"]["n_obs"] == 1


def test_unmatched_and_unexpected_week_games_reported(init_state, cfg):
    m, h = init_state
    g = pick_game(m, min_week=6)
    cfg.validation.max_reject_fraction = 1.0
    res = refresh(m, h, cfg, line_rows([(3 if g.week != 3 else 4, g.away_team, 2.0, g.home_team, 1.9)], D2))
    assert res.report.rejected_rows[0]["code"] in ("unexpected_week", "unmatched_game")
    assert res.report.counts["lines_refreshed"] == 0


def test_swapped_home_away_is_resolved_with_flipped_odds(init_state, cfg):
    m, h = init_state
    g = pick_game(m, min_week=6)
    res = refresh(m, h, cfg, line_rows([(g.week, g.home_team, 1.70, g.away_team, 2.20)], D2))
    new = row_of(res.master, g.game_id)
    p, _ = devig_two_way(1.70, 2.20, cfg.devig)
    assert new.latest_vegas_home_prob == pytest.approx(p) and new.latest_home_odds == 1.70
    assert new.latest_away_odds == 2.20 and any(i.code == "swapped_home_away" for i in res.report.issues)


def test_duplicate_rows_identical_conflicting_and_superseded(init_state, cfg):
    m, h = init_state
    g, g2 = m[m.week == 7].iloc[0], m[m.week == 7].iloc[1]
    same = pd.concat([line_rows([(g.week, g.away_team, 2.0, g.home_team, 1.9)], D2)] * 2, ignore_index=True)
    r1 = refresh(m, h, cfg, same)
    assert any(i.code == "duplicate_timestamp" for i in r1.report.issues)
    assert row_of(r1.master, g.game_id).refresh_count == 1 and r1.report.details["ratings"]["n_obs"] == 1
    conflict = pd.concat([line_rows([(g2.week, g2.away_team, 2.0, g2.home_team, 1.9)], D2),
                          line_rows([(g2.week, g2.away_team, 3.0, g2.home_team, 1.4)], D2)], ignore_index=True)
    cfg.validation.max_reject_fraction = 1.0
    r2 = refresh(m, h, cfg, conflict)
    assert row_of(r2.master, g2.game_id).refresh_count == 0
    assert any(i.code == "conflicting_duplicates" for i in r2.report.issues if i.level == "error")
    two_ts = pd.concat([line_rows([(g.week, g.away_team, 3.0, g.home_team, 1.4)], "2026-09-14 09:00:00"),
                        line_rows([(g.week, g.away_team, 2.0, g.home_team, 1.9)], D2)], ignore_index=True)
    assert row_of(refresh(m, h, cfg, two_ts).master, g.game_id).latest_home_odds == 1.9


def test_too_many_rejections_abort_without_result(init_state, cfg):
    m, h = init_state
    with pytest.raises(ValidationError, match="too_many_rejected_rows"):
        refresh(m, h, cfg, line_rows([(5, "Nope", 2.0, "Zip", 1.9)] * 4, D2))


# ============================================================ freezing
def test_frozen_weeks_never_change(init_state, cfg):
    m, h = init_state
    week1 = m[m.week == 1].iloc[0]
    x = team_playing_in(m, [3, 4])
    lines = pd.concat([line_rows([(week1.week, week1.away_team, 1.2, week1.home_team, 4.5)], D2),
                       implied_lines(m, h, [3, 4], D2, {x: 1.0})], ignore_index=True)
    res = refresh(m, h, cfg, lines, week=3)
    frozen = res.master[res.master.week < 3].set_index("game_id")
    before = m[m.week < 3].set_index("game_id").loc[frozen.index]
    for col in ("home_win_prob", "away_win_prob", "prob_source", "latest_vegas_home_prob", "adj_logit"):
        assert (frozen[col] == before[col]).all(), col
    assert frozen["is_frozen"].all() and (frozen["frozen_at_version"] == 2).all()
    assert res.report.counts["ignored_frozen_week"] == 1 and res.report.counts["newly_frozen"] == 32


def test_cannot_unfreeze_by_lowering_current_week(init_state, cfg):
    m, h = init_state
    with pytest.raises(ValidationError, match="un-freeze"):
        refresh(m, h, cfg, week=2, prev_current_week=3)


# ============================================================ ratings learn from fresh lines only
def test_ratings_move_toward_the_evidence(init_state, cfg):
    m, h = init_state
    x = team_playing_in(m, [2, 3])
    res = refresh(m, h, cfg, implied_lines(m, h, [2, 3], D2, {x: 0.8}))
    r0, r1 = res.hist.ratings(0), res.hist.ratings(1)
    assert r1[x] - r0[x] > 0.15
    opp = set(m[(m.week.isin([2, 3])) & ((m.home_team == x) | (m.away_team == x))][["home_team", "away_team"]].to_numpy().ravel()) - {x}
    assert all(r1[o] < r0[o] for o in opp)                                    # the other side of each game moves down
    assert res.report.details["ratings"]["n_obs"] == len(m[m.week.isin([2, 3])])


def test_no_new_information_means_no_rating_change_and_no_adjustment(init_state, cfg):
    m, h = init_state
    res = refresh(m, h, cfg, implied_lines(m, h, [6, 7], LATE), week=6, as_of=AS_OF_LATE)
    assert res.report.details["ratings"]["max_abs_rating_change"] < 2e-3
    assert res.report.counts["stale_adjusted"] == 0


def test_no_adjustment_until_line_is_older_than_start_age(init_state, cfg):
    m, h = init_state
    x = team_playing_in(m, [1, 2])
    res = refresh(m, h, cfg, implied_lines(m, h, [1, 2], "2026-09-08 13:00:00", {x: 1.0}), week=1,
                  as_of=T("2026-09-08T15:00:00Z"))               # preseason lines only ~2.7 weeks old
    assert res.hist.ratings(1)[x] - res.hist.ratings(0)[x] > 0.3   # the signal is there ...
    assert res.report.counts["stale_adjusted"] == 0                # ... but the line is too young to touch
    assert (res.master.loc[~res.master.is_frozen, "age_weight"] == 0).all()


def test_stale_adjustment_direction_and_bounds(init_state, cfg):
    m, h = init_state
    x = team_playing_in(m, [6, 7])
    res = refresh(m, h, cfg, implied_lines(m, h, [6, 7], LATE, {x: 1.0}), week=6, as_of=AS_OF_LATE)
    new = res.master
    xs = new[(new.week >= 10) & ((new.home_team == x) | (new.away_team == x))]
    assert len(xs) >= 5 and xs.adj_active.all()
    for g in xs.itertuples():
        d = g.home_win_prob - g.latest_vegas_home_prob
        assert d > 0 if g.home_team == x else d < 0
    assert 0 < new["home_win_prob"].min() and new["home_win_prob"].max() < 1
    assert (new["home_win_prob"] + new["away_win_prob"]).sub(1).abs().max() < 1e-12
    assert new["adj_logit"].abs().max() <= cfg.adjust.cap + 1e-12
    a = xs.iloc[0]
    assert a.prob_source == "VEGAS_STALE_ADJUSTED" and a.rating_ref_snapshot_id == 0 and a.rating_now_snapshot_id == 1
    assert a.home_win_prob == pytest.approx(shift_probability(a.latest_vegas_home_prob, a.adj_logit))
    fresh = new[new.week.isin([6, 7])]
    assert (fresh.prob_source == "VEGAS_FRESH").all() and not fresh.adj_active.any()   # fresh lines untouched


def test_small_evidence_below_threshold_changes_nothing(init_state, cfg):
    m, h = init_state
    cfg.strength.threshold_up = 0.12                    # gate behaviour, independent of the default value
    x = team_playing_in(m, [6, 7])
    res = refresh(m, h, cfg, implied_lines(m, h, [6, 7], LATE, {x: 0.05}), week=6, as_of=AS_OF_LATE)
    assert res.report.counts["stale_adjusted"] == 0


def test_line_time_reference_prevents_double_counting_but_preseason_reference_does_not(init_state, cfg):
    m, h = init_state
    x = team_playing_in(m, [2, 3])
    g = pick_game(m, min_week=10, team=x)
    p_home = 1 / (1 + np.exp(-(np.log(g.latest_vegas_home_prob / (1 - g.latest_vegas_home_prob)) + (1.0 if g.home_team == x else -1.0))))
    oh, oa = 1 / (p_home * 1.045), 1 / ((1 - p_home) * 1.045)
    lines = pd.concat([implied_lines(m, h, [2, 3], D2, {x: 1.0}),
                       line_rows([(g.week, g.away_team, oa, g.home_team, oh)], D2)], ignore_index=True)
    r1 = refresh(m, h, cfg, lines)                                           # X's surge is INSIDE g's fresh line
    assert row_of(r1.master, g.game_id).anchor_rating_snapshot_id == 1
    later = T("2026-10-20T15:00:00Z")                                         # g's line is now ~5 weeks old
    res_lt = refresh(r1.master, r1.hist, cfg, None, version=3, as_of=later)
    assert not row_of(res_lt.master, g.game_id).adj_active                    # nothing new since the line
    cfg2 = load_config(ROOT / "config.toml", strength__reference="preseason")
    res_ps = refresh(r1.master, r1.hist, cfg2, None, version=3, as_of=later)
    assert row_of(res_ps.master, g.game_id).adj_active                        # literal-baseline mode double counts


def test_fresh_lines_are_not_overwritten_and_adjustments_are_anchored_not_stacked(init_state, cfg):
    """Going through an intermediate run must give the SAME numbers as skipping it."""
    m, h = init_state
    x = team_playing_in(m, [6, 7])
    a1 = refresh(m, h, cfg, implied_lines(m, h, [6, 7], LATE, {x: 1.0}), week=6, as_of=AS_OF_LATE)
    t_mid, t2 = T("2026-10-16T15:00:00Z"), T("2026-10-20T15:00:00Z")
    direct = refresh(a1.master, a1.hist, cfg, None, week=6, as_of=t2, version=3)
    mid = refresh(a1.master, a1.hist, cfg, None, week=6, as_of=t_mid, version=3)
    via = refresh(mid.master, mid.hist, cfg, None, week=6, as_of=t2, version=4)
    pd.testing.assert_series_equal(direct.master["home_win_prob"], via.master["home_win_prob"])
    assert direct.master["adj_active"].sum() > 0
    a = direct.master[direct.master.adj_active].iloc[0]                        # stored fields fully explain the number
    from nfl_prob_grid.adjustment import cap_adjustment
    from nfl_prob_grid.strength import strength_shift
    delta = strength_shift(a.home_rating_move, cfg.strength) - strength_shift(a.away_rating_move, cfg.strength)
    assert a.adj_logit == pytest.approx(cap_adjustment(a.age_weight * delta, cfg.adjust))


def test_rerunning_same_inputs_is_idempotent(init_state, cfg):
    m, h = init_state
    x = team_playing_in(m, [2, 3])
    lines = implied_lines(m, h, [2, 3], D2, {x: 0.6})
    a = refresh(m, h, cfg, lines)
    b = refresh(a.master, a.hist, cfg, lines, version=3)
    pd.testing.assert_series_equal(a.master["home_win_prob"], b.master["home_win_prob"])
    assert b.report.counts.get("lines_refreshed", 0) == 0 and b.report.details["ratings"]["n_obs"] == 0


def test_asymmetric_config_ignores_up_moves_but_acts_on_down_moves(init_state, cfg):
    m, h = init_state
    x, y = two_teams_not_facing(m, [6, 7])
    cfg.strength.symmetric = False
    cfg.strength.threshold_up, cfg.strength.threshold_down = 5.0, 0.12
    res = refresh(m, h, cfg, implied_lines(m, h, [6, 7], LATE, {x: 1.0, y: -1.0}), week=6, as_of=AS_OF_LATE)
    ev = res.master[res.master["home_rating_move"].notna()]
    assert ev["home_strength_shift"].max() <= 1e-12 and ev["away_strength_shift"].max() <= 1e-12
    assert ((ev.home_rating_move > 0.3) & (ev.home_strength_shift == 0)).any()      # X's rise ignored
    assert (ev.home_strength_shift < 0).any() or (ev.away_strength_shift < 0).any()  # Y's decline acted on


def test_live_skill_check_scores_adjusted_vs_unadjusted_against_the_lines_that_arrive(init_state, cfg):
    m, h = init_state
    x = team_playing_in(m, [6, 7, 8])
    a1 = refresh(m, h, cfg, implied_lines(m, h, [6, 7], LATE, {x: 1.0}), week=6, as_of=AS_OF_LATE)
    g8 = a1.master[(a1.master.week == 8) & ((a1.master.home_team == x) | (a1.master.away_team == x))]
    assert g8.adj_active.all()                                                # X's week-8 line was adjusted
    truth = implied_lines(a1.master, a1.hist, [8], "2026-10-20 13:00:00", {x: 1.0}, use_anchor=True)
    xrows = truth[(truth["Home Team"] == x) | (truth["Away Team"] == x)]     # the market confirms +1.0 for X
    a2 = refresh(a1.master, a1.hist, cfg, xrows, week=7, as_of=T("2026-10-20T15:00:00Z"), version=3)
    sk = a2.report.details["adjustment_skill"]
    assert sk["n"] >= 1 and sk["mean_abs_err_adjusted_logit"] < sk["mean_abs_err_stale_logit"]


# ============================================================ persistence, versions, integrity
def _seed(cfg, demo):
    d = demo["dir"]
    run_init(cfg, as_of=INIT_AS_OF, current_week=1, lines_source=str(d / "initial_lines.csv"),
             win_totals_source=str(d / "win_totals.csv"))
    return d


def _run(cfg, d, w, **kw):
    return run_refresh(cfg, as_of=run_as_of(w), current_week=w, lines_source=str(d / f"refresh_w{w:02d}.csv"), **kw)


def test_versioned_state_progression_hash_chain_and_outputs(cfg, demo):
    d = _seed(cfg, demo)
    for w in (2, 3):
        _run(cfg, d, w)
    st = StateStore(cfg.paths.state_dir)
    man = st.read_manifest()
    assert man["current_version"] == 3 and man["current_week"] == 3
    assert [v["version"] for v in man["versions"]] == [1, 2, 3]
    assert man["versions"][1]["parent_sha256"] == man["versions"][0]["sha256"]
    assert all((st.versions_dir / v["file"]).exists() for v in man["versions"])
    master, hist, _ = st.load()
    assert master["is_frozen"].sum() == 32 and hist.latest_id() == 2 and len(hist.snapshots()) == 3
    v1, cur = (pd.read_csv(p, dtype=str) for p in (st.versions_dir / "master_v0001.csv", st.current_path))
    a, b = (x[x.week == "1"].set_index("game_id")["home_win_prob"] for x in (v1, cur))
    assert len(a) == 16 and (a == b.loc[a.index]).all()                       # frozen text-identical to v1
    out = Path(cfg.paths.output_dir)
    assert pd.read_csv(out / "prob_grid_latest.csv").shape == (32, 19)
    tr = pd.read_csv(out / "team_ratings_latest.csv")
    assert len(tr) == 32 and {"CurrentRating", "BaselineRating", "WinTotalRating", "ChangeSinceBaseline"} <= set(tr.columns)
    assert pd.read_csv(out / "team_ratings_history.csv").shape == (32, 4)


def test_next_run_starts_from_the_refreshed_master_not_the_preseason_table(cfg, demo):
    d = _seed(cfg, demo)
    _run(cfg, d, 2)
    after2 = pd.read_csv(StateStore(cfg.paths.state_dir).current_path).set_index("game_id")
    fresh_ids = after2.index[after2.prob_source == "VEGAS_FRESH"]
    assert len(fresh_ids) > 0
    _run(cfg, d, 3)
    after3 = pd.read_csv(StateStore(cfg.paths.state_dir).current_path).set_index("game_id")
    kept = [g for g in fresh_ids if after3.at[g, "last_refresh_version"] == 2]
    assert kept and (after3.loc[kept, "latest_vegas_home_prob"] == after2.loc[kept, "latest_vegas_home_prob"]).all()
    assert (after3.loc[kept, "anchor_rating_snapshot_id"] == 1).all()          # anchored on the run-2 snapshot


def test_tampered_master_or_rating_state_is_refused(cfg, demo):
    _seed(cfg, demo)
    st = StateStore(cfg.paths.state_dir)
    p = st.current_path
    orig = p.read_text()
    p.write_text(orig.replace("0.5", "0.6", 1))
    with pytest.raises(StateError, match="does not match manifest"):
        run_refresh(cfg, as_of=run_as_of(2), current_week=2, use_lines=False)
    p.write_text(orig)
    js = json.loads(st.rating_state_path.read_text())
    js["hfa"] += 0.1
    st.rating_state_path.write_text(json.dumps(js))
    with pytest.raises(StateError, match="rating_state.json"):
        run_refresh(cfg, as_of=run_as_of(2), current_week=2, use_lines=False)


def test_failed_run_writes_nothing_and_dry_run_writes_nothing(cfg, demo, tmp_path):
    d = _seed(cfg, demo)
    st = StateStore(cfg.paths.state_dir)
    before = st.read_manifest()
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Week": [5] * 6, "Away Team": ["Nope"] * 6, "Away Odds": [2.0] * 6,
                  "Home Team": ["Zip"] * 6, "Home Odds": [1.9] * 6, "Timestamp": [D2] * 6}).to_csv(bad, index=False)
    with pytest.raises(ValidationError):
        run_refresh(cfg, as_of=run_as_of(2), current_week=2, lines_source=str(bad))
    _run(cfg, d, 2, dry_run=True)
    assert st.read_manifest() == before and not (st.versions_dir / "master_v0002.csv").exists()


def test_missing_column_and_incomplete_win_totals_are_clear_errors(cfg, demo, tmp_path):
    d = _seed(cfg, demo)
    p = tmp_path / "x.csv"
    pd.DataFrame({"Week": [5], "Away Team": ["KC"]}).to_csv(p, index=False)
    with pytest.raises(InputError, match="missing column"):
        run_refresh(cfg, as_of=run_as_of(2), current_week=2, lines_source=str(p))
    short = pd.read_csv(d / "win_totals.csv").iloc[:-3]
    short.to_csv(tmp_path / "wt.csv", index=False)
    cfg.paths.state_dir = str(tmp_path / "state2")
    with pytest.raises(InputError, match="missing teams"):
        run_init(cfg, as_of=INIT_AS_OF, current_week=1, lines_source=str(d / "initial_lines.csv"),
                 win_totals_source=str(tmp_path / "wt.csv"))
    assert not StateStore(cfg.paths.state_dir).exists()                         # nothing half-built


def test_regenerate_grid_from_current_master(cfg, demo):
    _seed(cfg, demo)
    assert regenerate_grid(cfg).shape == (32, 18)


def test_validation_catches_corruption(init_state, cfg):
    from nfl_prob_grid.validate import validate_master
    m, _ = init_state
    bad = m.copy()
    bad.loc[0, "home_win_prob"] = 1.2
    bad.loc[1, "away_win_prob"] = 0.9
    bad.loc[2, "home_win_prob"] = float("nan")
    dup = pd.concat([m, m.iloc[[3]]], ignore_index=True)
    codes = {i.code for i in validate_master(bad, cfg) if i.level == "error"}
    assert {"impossible_probability", "probabilities_not_complementary"} <= codes
    assert {"duplicate_game_id", "duplicate_game"} <= {i.code for i in validate_master(dup, cfg)}
    tam = m.copy()
    tam.loc[m.week == 1, "home_win_prob"] += 0.01
    tam.loc[m.week == 1, "away_win_prob"] -= 0.01
    assert "frozen_week_modified" in {i.code for i in validate_master(tam, cfg, prev=m, current_week=3)}
