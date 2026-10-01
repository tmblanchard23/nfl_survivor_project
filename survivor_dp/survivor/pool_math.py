"""
Opponent elimination math (spec sections 7, 8, 12).

The critical modeling point: if 7 opponents all picked the Chiefs and the
Chiefs lose, that's ONE random event eliminating 7 people at once — not 7
independent coin flips. This module treats it that way.

Method: group the (fractional, expected) opponent count vector by team, then
walk the week's games one at a time. Each game is an independent Bernoulli
event (team A wins w.p. p, team B wins w.p. 1-p), and it moves a fixed lump
of "count" from "alive" to "eliminated" depending on which side wins. The
joint distribution over total remaining pool size is then the convolution of
these per-game lumps — computed exactly via a dictionary-based polynomial
convolution (support size stays small in practice: at most 2^n_games, but
usually far smaller because many teams have ~0 predicted opponents on them).

This is exact given the model's own assumptions (opponents' picks are
independent across opponents conditional on their fitted persona, and game
outcomes are independent of each other). It is *not* claiming to know reality
with certainty — it is the correct joint distribution implied by the upstream
probabilistic inputs.
"""
from __future__ import annotations
from .vegas import WeekSlate

_ROUND = 6  # collapse floating point noise in the convolution support


def _r(x: float) -> float:
    return round(x, _ROUND)


def remaining_pool_distribution(slate: WeekSlate, count_vector: dict[str, float]
                                 ) -> dict[float, float]:
    """
    Returns {remaining_pool_size: probability}, exact convolution over this
    week's games given the predicted opponent count vector.

    Teams in `count_vector` that aren't playing this week (bye, or a team
    already eliminated/unused) are dropped with a note-worthy floor of 0 —
    callers should ensure the count vector only reflects teams actually in
    this week's slate.
    """
    win_probs = slate.win_probs()
    dist: dict[float, float] = {0.0: 1.0}

    for g in slate.games:
        ca = count_vector.get(g.team_a, 0.0)
        cb = count_vector.get(g.team_b, 0.0)
        if ca == 0.0 and cb == 0.0:
            continue
        pa = win_probs[g.team_a]
        pb = win_probs[g.team_b]

        new_dist: dict[float, float] = {}
        for total, prob in dist.items():
            # team_a wins -> its backers survive (add ca), team_b's backers eliminated (add 0)
            key_a = _r(total + ca)
            new_dist[key_a] = new_dist.get(key_a, 0.0) + prob * pa
            # team_b wins -> its backers survive (add cb)
            key_b = _r(total + cb)
            new_dist[key_b] = new_dist.get(key_b, 0.0) + prob * pb
        dist = new_dist

    return dist


def summarize(dist: dict[float, float], pool_size_before: float
              ) -> dict[str, float]:
    expected_remaining = sum(size * p for size, p in dist.items())
    expected_eliminated = pool_size_before - expected_remaining
    prob_no_eliminations = dist.get(_r(pool_size_before), 0.0)
    prob_at_least_one_eliminated = 1.0 - prob_no_eliminations
    return {
        "expected_remaining_pool": expected_remaining,
        "expected_opponents_eliminated": expected_eliminated,
        "prob_at_least_one_opponent_eliminated": prob_at_least_one_eliminated,
    }


def check_probabilities_conserved(dist: dict[float, float], tol: float = 1e-6) -> bool:
    return abs(sum(dist.values()) - 1.0) < tol


def remaining_pool_distribution_given_my_team_won(slate: WeekSlate,
                                                    count_vector: dict[str, float],
                                                    my_team: str) -> dict[float, float]:
    """
    Distribution of opponent pool size CONDITIONAL on `my_team` having won
    this week (i.e. conditional on my own survival).

    This is the piece that actually produces the game-theoretic effect the
    spec describes in section 13: if I pick the same team as a big chunk of
    the field, my survival is perfectly correlated with theirs (they survive
    right alongside me in this branch, by construction). If I pick a
    lightly-owned team instead, my survival is a near-independent event, so
    conditioning on it barely touches the distribution over what happens to
    everyone else. Computing eliminations the same way regardless of my own
    pick would silently erase this effect, so it's handled by removing my
    own game from the convolution and folding its (now-certain) survivors in
    as a fixed offset.
    """
    my_game = slate.game_for_team(my_team)
    fixed_survivors = count_vector.get(my_team, 0.0)
    other_games_slate = WeekSlate(week=slate.week,
                                   games=[g for g in slate.games if g is not my_game])
    base_dist = remaining_pool_distribution(other_games_slate, count_vector)
    shifted: dict[float, float] = {}
    for size, prob in base_dist.items():
        key = _r(size + fixed_survivors)
        shifted[key] = shifted.get(key, 0.0) + prob
    return shifted
