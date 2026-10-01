import math
from survivor.vegas import Game, WeekSlate
from survivor.team_registry import TeamRegistry
from survivor import dp


def make_two_week_slates():
    # Week 1: A is the bigger favorite this week, B is a smaller favorite.
    w1 = WeekSlate(1, [Game(1, "A", "X", -500, 400), Game(1, "B", "Y", -120, 100)])
    # Week 2: A becomes a monster favorite. B isn't playing this week (bye),
    # so the only alternative to A next week is a much weaker team Z.
    w2 = WeekSlate(2, [Game(2, "A", "Z", -1000, 700)])
    return {1: w1, 2: w2}


def test_one_team_one_week():
    reg = TeamRegistry()
    slates = {1: WeekSlate(1, [Game(1, "A", "X", -500, 400)])}
    val, policy = dp.pure_survival_value(reg, 0, 1, slates)
    assert math.isclose(val, slates[1].win_probs()["A"], rel_tol=1e-9)
    assert policy == {1: "A"}


def test_two_teams_two_weeks_picks_best_combo():
    reg = TeamRegistry()
    slates = make_two_week_slates()
    val, policy = dp.pure_survival_value(reg, 0, 1, slates)
    # Optimal: use A in week 2 (bigger favorite, -1000) and B in week 1,
    # since A is available both weeks but is a much bigger favorite in week 2.
    assert policy[1] == "B"
    assert policy[2] == "A"


def test_team_cannot_be_reused():
    reg = TeamRegistry()
    slates = make_two_week_slates()
    _, policy = dp.pure_survival_value(reg, 0, 1, slates)
    teams_used = list(policy.values())
    assert len(teams_used) == len(set(teams_used))


def test_used_teams_removed_from_consideration():
    reg = TeamRegistry()
    slates = make_two_week_slates()
    # Pre-mark A as used.
    used_mask = reg.bit("A")
    val, policy = dp.pure_survival_value(reg, used_mask, 1, slates)
    assert "A" not in policy.values()
    assert policy[1] == "B"
    # With A already used, week 2's only legal option is Z.
    assert policy[2] == "Z"


def test_week_advancement_stops_at_horizon():
    reg = TeamRegistry()
    slates = {1: WeekSlate(1, [Game(1, "A", "X", -500, 400)])}
    val, policy = dp.pure_survival_value(reg, 0, 1, slates)
    # Week 2 isn't in `slates` -> treated as beyond horizon, contributes 1.0
    assert math.isclose(val, slates[1].win_probs()["A"], rel_tol=1e-9)
    assert 2 not in policy
