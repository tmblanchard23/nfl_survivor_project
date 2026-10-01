from survivor.vegas import Game, WeekSlate
from survivor.team_registry import TeamRegistry
from survivor import dp


def test_optimizer_prefers_lower_prob_team_today_for_better_future():
    reg = TeamRegistry()
    # Week 1: A is the bigger favorite (~0.78), B is a smaller favorite (~0.72).
    w1 = WeekSlate(1, [Game(1, "A", "X", -400, 350), Game(1, "B", "Y", -300, 250)])
    # Week 2: A becomes a monster favorite (~0.98). B isn't playing (bye) --
    # only a much weaker team C is available alongside A.
    w2 = WeekSlate(2, [Game(2, "A", "Z", -10000, 5000), Game(2, "C", "D", -110, -110)])
    slates = {1: w1, 2: w2}

    val, policy = dp.pure_survival_value(reg, 0, 1, slates)

    w1_probs = w1.win_probs()
    assert w1_probs["A"] > w1_probs["B"]  # A is indeed the better team THIS week

    # Yet the optimal policy should use B this week, preserving A for its
    # much bigger week-2 edge.
    assert policy[1] == "B"
    assert policy[2] == "A"


def test_future_opportunity_cost_is_positive_when_preservation_matters():
    reg = TeamRegistry()
    w1 = WeekSlate(1, [Game(1, "A", "X", -400, 350), Game(1, "B", "Y", -300, 250)])
    w2 = WeekSlate(2, [Game(2, "A", "Z", -10000, 5000), Game(2, "C", "D", -110, -110)])
    slates = {1: w1, 2: w2}
    cost = dp.future_opportunity_cost("A", reg, 0, 1, slates)
    assert cost > 0  # using A now is costly relative to saving it


def test_future_opportunity_cost_near_zero_when_no_better_future_use():
    reg = TeamRegistry()
    # Team A is equally strong both weeks -- no real preservation incentive.
    w1 = WeekSlate(1, [Game(1, "A", "X", -400, 350), Game(1, "B", "Y", -110, -110)])
    w2 = WeekSlate(2, [Game(2, "A", "Z", -400, 350), Game(2, "C", "D", -110, -110)])
    slates = {1: w1, 2: w2}
    cost = dp.future_opportunity_cost("A", reg, 0, 1, slates)
    assert cost <= 0.02  # using it now isn't meaningfully worse than saving it
