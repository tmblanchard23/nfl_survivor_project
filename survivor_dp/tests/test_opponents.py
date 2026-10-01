import math
from survivor.opponents import OpponentModel, aggregate_count_vector
from survivor.vegas import Game, WeekSlate


def week_slate(week, pairs):
    return WeekSlate(week, [Game(week, a, b, ma, mb) for (a, b, ma, mb) in pairs])


def test_single_opponent_predicts_over_available_teams_only():
    opp = OpponentModel("opp1")
    slate = week_slate(1, [("A", "X", -300, 250), ("B", "Y", -150, 130)])
    wp = slate.win_probs()
    dist = opp.predict(["A", "B"], wp)
    assert set(dist.keys()) == {"A", "B"}
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-6)


def test_multiple_opponents_same_team_count_vector():
    slate = week_slate(1, [("A", "X", -300, 250), ("B", "Y", -150, 130)])
    wp = slate.win_probs()
    preds = {}
    for i in range(7):
        opp = OpponentModel(f"opp{i}")
        preds[f"opp{i}"] = {"A": 1.0, "B": 0.0}  # forced deterministic for test clarity
    cv = aggregate_count_vector(preds)
    assert math.isclose(cv["A"], 7.0, abs_tol=1e-9)
    assert cv.get("B", 0.0) == 0.0


def test_chalk_opponent_prediction_favors_higher_win_prob():
    opp = OpponentModel("chalky")
    slate = week_slate(1, [("A", "X", -400, 320), ("B", "Y", -110, -110)])
    wp = slate.win_probs()
    # Feed it a long history of always picking the highest win-prob team.
    for wk in range(1, 10):
        s = week_slate(wk, [(f"T{wk}a", f"T{wk}b", -400, 320)])
        w = s.win_probs()
        opp.observe_pick(wk, f"T{wk}a", [f"T{wk}a", f"T{wk}b"], w)
    dist = opp.predict(["A", "B"], wp)
    assert dist["A"] > dist["B"]
    assert opp.persona_label() in ("Chalk Player", "Favorite-First Player")


def test_small_sample_stays_close_to_population_prior():
    """With only 1 observation, an opponent's fitted beta should be heavily
    shrunk toward the population prior -- one weird pick shouldn't flip the
    model's behavior."""
    opp_low_n = OpponentModel("low_n")
    s = week_slate(1, [("Dog", "Fav", 400, -500)])
    w = s.win_probs()
    opp_low_n.observe_pick(1, "Dog", ["Dog", "Fav"], w)  # one big-underdog pick

    # A population-default (never observed) opponent, for comparison.
    opp_default = OpponentModel("default")

    slate2 = week_slate(2, [("Dog2", "Fav2", 400, -500)])
    w2 = slate2.win_probs()
    dist_low_n = opp_low_n.predict(["Dog2", "Fav2"], w2)
    dist_default = opp_default.predict(["Dog2", "Fav2"], w2)

    # The single contrarian observation should nudge the prediction, but
    # shrinkage should keep it from flipping all the way to favoring the dog.
    assert dist_low_n["Dog2"] > dist_default["Dog2"]        # some movement
    assert dist_low_n["Fav2"] > dist_low_n["Dog2"]           # but favorite still favored


def test_large_sample_converges_further_from_prior_than_small_sample():
    def build_opponent(n_obs):
        opp = OpponentModel(f"n{n_obs}")
        for wk in range(1, n_obs + 1):
            s = week_slate(wk, [("Dog", "Fav", 400, -500)])
            w = s.win_probs()
            opp.observe_pick(wk, "Dog", ["Dog", "Fav"], w)
        return opp

    opp_1 = build_opponent(1)
    opp_15 = build_opponent(15)

    slate = week_slate(100, [("Dog2", "Fav2", 400, -500)])
    w = slate.win_probs()
    p1 = opp_1.predict(["Dog2", "Fav2"], w)["Dog2"]
    p15 = opp_15.predict(["Dog2", "Fav2"], w)["Dog2"]
    assert p15 > p1  # more consistent evidence -> more confident deviation from prior


def test_persona_updates_after_observing_unexpected_pick():
    opp = OpponentModel("switchy")
    # Establish as chalk-like first.
    for wk in range(1, 6):
        s = week_slate(wk, [(f"F{wk}", f"D{wk}", -400, 320)])
        w = s.win_probs()
        opp.observe_pick(wk, f"F{wk}", [f"F{wk}", f"D{wk}"], w)
    slate_before = week_slate(50, [("Fav", "Dog", -400, 320)])
    wb = slate_before.win_probs()
    pred_before = opp.predict(["Fav", "Dog"], wb)

    # Now feed several underdog picks and confirm the prediction shifts.
    for wk in range(6, 12):
        s = week_slate(wk, [(f"F{wk}", f"D{wk}", -400, 320)])
        w = s.win_probs()
        opp.observe_pick(wk, f"D{wk}", [f"F{wk}", f"D{wk}"], w)

    pred_after = opp.predict(["Fav", "Dog"], wb)
    assert pred_after["Dog"] > pred_before["Dog"]
