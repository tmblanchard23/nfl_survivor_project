import pytest
from survivor.vegas import Game, WeekSlate
from survivor.workflow import SurvivorPool


def make_pool_with_week1():
    pool = SurvivorPool()
    pool.add_opponent("opp1")
    pool.add_opponent("opp2")
    slate = WeekSlate(1, [Game(1, "A", "X", -300, 250), Game(1, "B", "Y", -150, 130)])
    pool.set_week_slate(slate)
    return pool


def test_get_recommendations_does_not_mutate_state():
    pool = make_pool_with_week1()
    used_mask_before = pool.my_used_mask
    week_before = pool.current_week
    rec = pool.get_recommendations()
    assert pool.my_used_mask == used_mask_before
    assert pool.current_week == week_before
    assert len(rec.pure_survival_top) > 0
    assert len(rec.pool_equity_top) > 0

    # Calling it again should be idempotent -- still no mutation.
    rec2 = pool.get_recommendations()
    assert pool.my_used_mask == used_mask_before
    assert pool.current_week == week_before


def test_advance_week_requires_a_recorded_pick():
    pool = make_pool_with_week1()
    pool.get_recommendations()  # generating a recommendation is not a pick
    with pytest.raises(ValueError):
        pool.advance_week()


def test_recommendation_is_not_auto_applied_as_state_change():
    pool = make_pool_with_week1()
    rec = pool.get_recommendations()
    top_pick = rec.pool_equity_top[0].team
    # Even though we have a "recommended" team, nothing should be marked used.
    assert pool.my_used_mask == 0
    # Only explicit action commits it.
    pool.record_my_pick(top_pick)
    assert pool.my_used_mask == 0  # still not committed until advance_week()
    pool.advance_week()
    assert pool.registry.is_used(pool.my_used_mask, top_pick)
    assert pool.current_week == 2


def test_get_recommendations_refuses_when_no_lines_loaded_at_all():
    pool = make_pool_with_week1()
    pool.record_my_pick("A")
    pool.advance_week()
    assert pool.current_week == 2
    # No week-2 slate loaded yet.
    with pytest.raises(ValueError):
        pool.get_recommendations()

    # Only after loading week-2 data does it work.
    pool.set_week_slate(WeekSlate(2, [Game(2, "C", "Z", -200, 170)]))
    rec = pool.get_recommendations()
    assert rec.week == 2


def test_opponent_updates_only_happen_after_explicit_call():
    pool = make_pool_with_week1()
    opp = pool.opponents["opp1"]
    n_before = len(opp.history)
    pool.get_recommendations()
    assert len(opp.history) == n_before  # recommending must not touch opponent history

    pool.record_opponent_picks({"opp1": "A"})
    assert len(opp.history) == n_before + 1


def test_full_weekly_cycle_and_probability_bookkeeping():
    pool = make_pool_with_week1()
    rec = pool.get_recommendations()
    for cand in rec.pool_equity_top:
        assert 0.0 <= cand.p_survive_this_week <= 1.0
        assert 0.0 <= cand.p_win_pool_approx <= 1.0
        total_prob = sum(cand.remaining_pool_distribution.values())
        assert abs(total_prob - 1.0) < 1e-6

    pool.record_my_pick("A")
    pool.record_opponent_picks({"opp1": "B", "opp2": "A"})
    pool.record_results({"A": True, "B": False})
    assert pool.opponents["opp2"].eliminated is False
    assert pool.opponents["opp1"].eliminated is True
    pool.advance_week()
    assert pool.current_week == 2


def test_confidence_is_high_when_current_week_has_data():
    pool = make_pool_with_week1()
    rec = pool.get_recommendations()
    assert rec.confidence["vegas_information"] == "HIGH"


def test_confidence_is_low_when_current_week_has_no_data():
    pool = SurvivorPool()
    # No week-1 slate loaded at all -- get_recommendations() raises before
    # confidence is ever computed, so check the underlying logic directly.
    with pytest.raises(ValueError):
        pool.get_recommendations()
    assert pool._confidence_report()["vegas_information"] == "LOW"
