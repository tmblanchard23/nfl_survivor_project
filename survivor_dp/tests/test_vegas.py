import math
from survivor.vegas import (
    moneyline_to_implied_prob, no_vig_probs, Game, WeekSlate,
    decimal_to_implied_prob,
)


def test_favorite_implied_prob():
    assert math.isclose(moneyline_to_implied_prob(-200), 2/3, rel_tol=1e-6)


def test_underdog_implied_prob():
    assert math.isclose(moneyline_to_implied_prob(170), 100/270, rel_tol=1e-6)


def test_no_vig_matches_spec_example():
    fav, dog = no_vig_probs(-200, 170)
    assert math.isclose(fav, 0.6428, abs_tol=1e-3)
    assert math.isclose(dog, 1 - 0.6428, abs_tol=1e-3)


def test_no_vig_sums_to_one():
    fav, dog = no_vig_probs(-150, 130)
    assert math.isclose(fav + dog, 1.0, abs_tol=1e-9)


def test_week_slate_win_probs():
    g1 = Game(1, "Chiefs", "Raiders", -300, 250)
    g2 = Game(1, "Eagles", "Giants", -180, 155)
    slate = WeekSlate(1, games=[g1, g2])
    probs = slate.win_probs()
    assert set(probs.keys()) == {"Chiefs", "Raiders", "Eagles", "Giants"}
    assert math.isclose(probs["Chiefs"] + probs["Raiders"], 1.0, abs_tol=1e-9)


def test_decimal_implied_prob():
    assert math.isclose(decimal_to_implied_prob(1.50), 1 / 1.50, rel_tol=1e-9)
    assert math.isclose(decimal_to_implied_prob(2.70), 1 / 2.70, rel_tol=1e-9)


def test_decimal_odds_rejects_non_positive_edge():
    import pytest
    with pytest.raises(ValueError):
        decimal_to_implied_prob(1.0)
    with pytest.raises(ValueError):
        decimal_to_implied_prob(0.8)


def test_decimal_and_american_agree_on_equivalent_market():
    # -200/+170 (American) is the same market as 1.50/2.70 (decimal), give or
    # take rounding.
    fav_a, dog_a = no_vig_probs(-200, 170, odds_format="american")
    fav_d, dog_d = no_vig_probs(1.50, 2.70, odds_format="decimal")
    assert math.isclose(fav_a, fav_d, abs_tol=5e-3)
    assert math.isclose(dog_a, dog_d, abs_tol=5e-3)


def test_game_from_decimal_matches_manual_no_vig():
    g = Game.from_decimal(2, "Buffalo Bills", "Detroit Lions", 1.51, 2.64)
    probs = g.no_vig_win_probs()
    expected_bills, expected_lions = no_vig_probs(1.51, 2.64, odds_format="decimal")
    assert math.isclose(probs["Buffalo Bills"], expected_bills, abs_tol=1e-9)
    assert math.isclose(probs["Detroit Lions"], expected_lions, abs_tol=1e-9)


def test_decimal_week_slate_from_sheet_style_row():
    # Mirrors the actual sheet columns: Away Team, Home Team, Moneyline Away
    # (decimal), Moneyline Home (decimal).
    g = Game.from_decimal(2, "Detroit Lions", "Buffalo Bills", 2.64, 1.51)
    slate = WeekSlate(2, games=[g])
    probs = slate.win_probs()
    assert probs["Buffalo Bills"] > probs["Detroit Lions"]
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)


def test_game_from_probability_uses_value_directly_no_devig():
    # Mirrors "Master Game Table": Away Win % / Home Win % are ALREADY
    # finished probabilities from the grid engine -- no conversion needed.
    g = Game.from_probability(3, "Atlanta Falcons", "Green Bay Packers",
                               prob_a=0.320, prob_b=0.680)
    probs = g.no_vig_win_probs()
    assert math.isclose(probs["Atlanta Falcons"], 0.320, abs_tol=1e-9)
    assert math.isclose(probs["Green Bay Packers"], 0.680, abs_tol=1e-9)


def test_game_from_probability_infers_complement_when_only_one_side_given():
    g = Game.from_probability(3, "Team A", "Team B", prob_a=0.417)
    probs = g.no_vig_win_probs()
    assert math.isclose(probs["Team A"], 0.417, abs_tol=1e-9)
    assert math.isclose(probs["Team B"], 0.583, abs_tol=1e-9)


def test_game_from_probability_normalizes_small_rounding_drift():
    # Sheet shows "39.7%" / "60.3%" -- sums to exactly 1.0 here, but a grid
    # engine's raw output could plausibly be 0.397/0.604 (rounding noise).
    # Should still come out normalized to sum to 1, not literally 1.001.
    g = Game.from_probability(1, "A", "B", prob_a=0.397, prob_b=0.604)
    probs = g.no_vig_win_probs()
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)


def test_probability_format_rejects_out_of_range_values():
    import pytest
    with pytest.raises(ValueError):
        Game.from_probability(1, "A", "B", prob_a=1.2, prob_b=-0.2).no_vig_win_probs()
    with pytest.raises(ValueError):
        Game.from_probability(1, "A", "B", prob_a=0.0, prob_b=1.0).no_vig_win_probs()
