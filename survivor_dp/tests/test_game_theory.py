from survivor.vegas import Game, WeekSlate
from survivor.team_registry import TeamRegistry
from survivor import dp


def build_single_week_scenario():
    reg = TeamRegistry()
    # Two good, comparable favorites (Chiefs/Ravens), with genuinely bad
    # longshots on the other side of each game so a "wipe out the chalk"
    # lottery ticket can't mathematically hijack the ranking -- this isolates
    # the actual game-theoretic question (which GOOD team to differentiate
    # into) from the separate, more extreme long-shot-correlation dynamic.
    slate = WeekSlate(1, [
        Game(1, "Chiefs", "Raiders", -900, 650),
        Game(1, "Ravens", "Browns", -700, 500),
    ])
    slates = {1: slate}
    return reg, slate, slates


def evaluate(reg, slates, chiefs_ownership_frac, pool_size=100, total_remaining_weeks=6):
    """Build a synthetic opponent pool split between Chiefs and Ravens in the
    given proportion, then evaluate all candidates."""
    opponent_predictions = {
        f"opp{i}": {"Chiefs": chiefs_ownership_frac, "Ravens": 1 - chiefs_ownership_frac}
        for i in range(pool_size)
    }
    opponent_features = {oid: {"avg_pick_win_prob": None} for oid in opponent_predictions}

    return dp.evaluate_candidates(
        reg, 0, 1, slates, opponent_predictions, opponent_features,
        pool_size_before=pool_size, total_remaining_weeks=total_remaining_weeks,
    )


def test_pool_equity_preference_can_flip_with_ownership_concentration():
    reg, slate, slates = build_single_week_scenario()

    evals_low_chiefs_ownership = evaluate(reg, slates, chiefs_ownership_frac=0.15,
                                           total_remaining_weeks=6)
    evals_high_chiefs_ownership = evaluate(reg, slates, chiefs_ownership_frac=0.85,
                                            total_remaining_weeks=6)

    top_low = dp.rank_pool_equity(evals_low_chiefs_ownership)[0].team
    top_high = dp.rank_pool_equity(evals_high_chiefs_ownership)[0].team

    # Same Vegas lines, same two good candidates, different opponent
    # concentration -> the pool-equity-optimal pick is allowed to (and here,
    # does) differ.
    assert top_low != top_high


def test_pure_survival_ranking_ignores_ownership_but_pool_equity_does_not():
    reg, slate, slates = build_single_week_scenario()
    evals = evaluate(reg, slates, chiefs_ownership_frac=0.85, total_remaining_weeks=6)

    survival_ranking = [e.team for e in dp.rank_pure_survival(evals)]
    equity_ranking = [e.team for e in dp.rank_pool_equity(evals)]

    # Pure survival should always prefer the higher win-prob team (Chiefs),
    # regardless of ownership.
    assert survival_ranking[0] == "Chiefs"
    # Pool equity, under heavy Chiefs ownership, should prefer differentiating
    # into the other strong-but-less-owned team (Ravens) -- not into one of
    # the bad long-shot underdogs, which should stay dominated throughout.
    assert equity_ranking[0] == "Ravens"
    assert equity_ranking[0] != "Raiders" and equity_ranking[0] != "Browns"
    # The two objectives disagree here -- exactly the point of section 11.
    assert survival_ranking[0] != equity_ranking[0]
