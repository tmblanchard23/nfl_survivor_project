import math
from survivor.vegas import Game, WeekSlate
from survivor import pool_math


def two_game_slate_with_round_probs():
    # Rig moneylines to land on clean 0.7 / 0.4 no-vig probs (approximately)
    # by using matched vig-free-style lines: pick moneylines then verify.
    g1 = Game(1, "A", "X", -233, 190)   # ~0.70 / 0.30 no-vig (approx)
    g2 = Game(1, "B", "Y", -67, -67)    # even game -> ~0.50 each... adjust below
    return WeekSlate(1, [g1, g2])


def test_correlated_elimination_matches_hand_computation():
    g1 = Game(1, "A", "X", -100, -100)  # pa = 0.5 exactly (both -100, symmetric)
    g2 = Game(1, "B", "Y", -100, -100)  # pb = 0.5 exactly
    slate = WeekSlate(1, [g1, g2])
    count_vector = {"A": 10.0, "X": 0.0, "B": 5.0, "Y": 0.0}

    dist = pool_math.remaining_pool_distribution(slate, count_vector)
    assert pool_math.check_probabilities_conserved(dist)

    # Hand-computed: outcomes are (A or X wins) x (B or Y wins), each combo 0.25
    # A&B win -> 15, A&Y(B loses) -> 10, X(A loses)&B -> 5, X&Y -> 0
    expected = {15.0: 0.25, 10.0: 0.25, 5.0: 0.25, 0.0: 0.25}
    for k, v in expected.items():
        assert math.isclose(dist.get(k, 0.0), v, abs_tol=1e-9)


def test_seven_opponents_eliminated_together_not_independently():
    """If 7 opponents all picked the same team and it loses, all 7 go at
    once -- the distribution should never show a partial (e.g. 3-of-7)
    elimination for that single shared game, because it's one random event,
    not seven independent ones."""
    g = Game(1, "Chiefs", "Raiders", -300, 250)
    slate = WeekSlate(1, [g])
    count_vector = {"Chiefs": 7.0, "Raiders": 0.0}
    dist = pool_math.remaining_pool_distribution(slate, count_vector)
    # Only two possible outcomes: all 7 survive, or all 7 are gone.
    assert set(round(k) for k in dist.keys()) <= {0, 7}
    assert pool_math.check_probabilities_conserved(dist)


def test_conditional_on_my_team_folds_in_correlated_survivors():
    g1 = Game(1, "Chiefs", "Raiders", -300, 250)
    g2 = Game(1, "Ravens", "Browns", -150, 130)
    slate = WeekSlate(1, [g1, g2])
    count_vector = {"Chiefs": 40.0, "Raiders": 0.0, "Ravens": 5.0, "Browns": 0.0}

    # If I pick Chiefs (heavily owned), conditioning on my survival guarantees
    # those 40 Chiefs backers survive too -- the minimum possible remaining
    # pool size in this conditional distribution must be at least 40.
    dist_chiefs = pool_math.remaining_pool_distribution_given_my_team_won(
        slate, count_vector, "Chiefs")
    assert pool_math.check_probabilities_conserved(dist_chiefs)
    assert min(dist_chiefs.keys()) >= 40.0 - 1e-9

    # If I pick Ravens (lightly owned), conditioning on my survival only
    # guarantees the 5 Ravens backers -- the Chiefs game's uncertainty (and
    # its 40 backers' fate) is untouched, so the minimum can be as low as 5.
    dist_ravens = pool_math.remaining_pool_distribution_given_my_team_won(
        slate, count_vector, "Ravens")
    assert pool_math.check_probabilities_conserved(dist_ravens)
    assert min(dist_ravens.keys()) <= 5.0 + 1e-9
