"""
The DP core (spec sections 3, 10-14, 17-19, 21).

Two exact/approximate layers, deliberately kept separate and both surfaced
to the caller so nothing is silently blended:

1. PURE SURVIVAL (Objective A) — exact.
   Value = probability of surviving from `week` through the end of the
   supplied slates, playing optimally (one new team per week, never
   reused). Solved as an assignment problem on log-probabilities (see the
   Objective A section below for why that's exactly equivalent), memoized
   per (used_mask, week), and reused for the "future opportunity cost"
   diagnostic in section 14.

2. POOL EQUITY (Objective B) — exact for the *current* decision week,
   approximate beyond it.
   For the current week, opponent elimination is computed exactly via
   pool_math's convolution (section 7/8's count-vector correlation).
   The "probability I ultimately win the pool" figure combines that exact
   one-week mechanic with:
     - my own best achievable future survival probability (exact, from
       layer 1), and
     - a labeled APPROXIMATION for opponents' aggregate future survival,
       built from their historical average win-probability-of-pick,
       compounded over remaining weeks (a proportional-hazard-style
       approximation, not a full multi-week joint opponent simulation,
       because that is combinatorially explosive — section 21 explicitly
       sanctions exactly this kind of labeled fallback).

   P(I win pool) is approximated as:
       P(win) ~= P(I survive this week) * [ my_continuation_value /
                  (my_continuation_value + E[competing survivors after this
                   week | I survived] * avg_opponent_future_factor) ]
   i.e. survive-this-week, THEN take a proportional share of the future
   value still in play, conditional on having survived. Both halves of the
   share are conditioned on the same event (my survival), which matters: an
   earlier version of this mixed a marginal probability into one half and a
   conditional one into the other, which let extreme long-shots on the
   opposite side of a chalk block look artificially good. This is a
   standard, honestly-approximate heuristic used in real survivor-pool
   equity calculators. It is NOT an ad hoc weighted score over today's pick
   features (section 10's prohibition) — it's a terminal-value estimate
   plugged into an otherwise-exact recursive one-week evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

from .vegas import WeekSlate
from .team_registry import TeamRegistry
from . import pool_math


# ---------------------------------------------------------------------
# Objective A: exact pure-survival optimization (assignment formulation)
# ---------------------------------------------------------------------
#
# "Pick one team per remaining week, never reuse a team, maximize the
# product of win probabilities" is exactly equivalent to maximizing the SUM
# of log win probabilities under the same constraints -- which is a
# rectangular assignment problem (weeks x teams). The Hungarian algorithm
# (scipy.optimize.linear_sum_assignment) solves that EXACTLY in polynomial
# time. An earlier version of this module enumerated pick sequences
# recursively; that was also exact, but its state space grows combinatorially
# (roughly C(32, weeks_remaining)), so it never finished on a real 18-week
# season. Same answer, different algorithm -- nothing here is approximate.

import math
import numpy as np
from scipy.optimize import linear_sum_assignment

_INFEASIBLE_COST = 1e9  # team not playing that week (bye) or already used


def _horizon_weeks(week: int, future_slates: dict[int, WeekSlate]) -> list[int]:
    """Consecutive weeks from `week` onward that have data. The first missing
    week ends the modeled horizon (same semantics as before)."""
    weeks = []
    w = week
    while w in future_slates:
        weeks.append(w)
        w += 1
    return weeks


def pure_survival_value(team_registry: TeamRegistry, used_mask: int, week: int,
                         future_slates: dict[int, WeekSlate],
                         memo: Optional[dict] = None
                         ) -> tuple[float, dict[int, str]]:
    """
    Exact. Returns (best survival probability from `week` onward, policy),
    where policy maps week -> team. Any week not present in `future_slates`
    ends the modeled horizon. Returns (0.0, {}) if there is no legal way to
    fill every remaining week (e.g. more weeks left than unused teams).
    """
    if memo is None:
        memo = {}
    key = (used_mask, week)
    if key in memo:
        return memo[key]

    weeks = _horizon_weeks(week, future_slates)
    if not weeks:
        memo[key] = (1.0, {})
        return memo[key]

    week_probs = [future_slates[w].win_probs() for w in weeks]
    teams = sorted({t for probs in week_probs for t in probs
                    if not team_registry.is_used(used_mask, t)})
    if len(teams) < len(weeks):
        memo[key] = (0.0, {})
        return memo[key]

    cost = np.full((len(weeks), len(teams)), _INFEASIBLE_COST)
    for i, probs in enumerate(week_probs):
        for j, t in enumerate(teams):
            p = probs.get(t)
            if p is not None and p > 0.0:
                cost[i, j] = -math.log(p)

    rows, cols = linear_sum_assignment(cost)
    if cost[rows, cols].max() >= _INFEASIBLE_COST:
        memo[key] = (0.0, {})
        return memo[key]

    value = math.exp(-float(cost[rows, cols].sum()))
    policy = {weeks[r]: teams[c] for r, c in zip(rows, cols)}
    memo[key] = (value, policy)
    return memo[key]


def future_opportunity_cost(team: str, team_registry: TeamRegistry, used_mask: int,
                             week: int, future_slates: dict[int, WeekSlate],
                             memo: Optional[dict] = None) -> float:
    """
    Exact, given the solver above.
    cost = V(preserve `team`, use best alternative this week)
         - V(use `team` this week)
    Positive => preserving is worth more than spending it now (it's "expensive"
    to use). Negative/zero => there is no real cost to using it now.
    """
    if memo is None:
        memo = {}
    slate = future_slates[week]
    win_probs = slate.win_probs()
    bit = team_registry.bit(team)

    cont_with, _ = pure_survival_value(team_registry, used_mask | bit, week + 1,
                                        future_slates, memo)
    value_with = win_probs[team] * cont_with

    best_alt = 0.0
    for other_team in slate.teams():
        if other_team == team:
            continue
        other_bit = team_registry.bit(other_team)
        if used_mask & other_bit:
            continue
        cont_other, _ = pure_survival_value(team_registry, used_mask | other_bit,
                                             week + 1, future_slates, memo)
        best_alt = max(best_alt, win_probs[other_team] * cont_other)

    return best_alt - value_with


# ---------------------------------------------------------------------
# Objective B: pool equity for the current decision
# ---------------------------------------------------------------------

@dataclass
class OpponentFutureEstimate:
    """Labeled-approximate per-opponent future survival estimate."""
    opponent_id: str
    remaining_weeks: int
    avg_pick_win_prob: float  # historical, or population default
    future_survival_prob: float  # avg_pick_win_prob ** remaining_weeks (approx)


DEFAULT_POPULATION_AVG_PICK_WIN_PROB = 0.65


def estimate_opponent_future_survival(opponent_descriptive_features: dict,
                                       remaining_weeks: int) -> OpponentFutureEstimate:
    """
    APPROXIMATE (labeled). Assumes an opponent's future per-week survival
    probability equals their historical average win-probability-of-pick
    (or a population default if they have no history), compounded
    independently across remaining weeks. This ignores their own team-reuse
    constraints and any skill trend — it is a coarse aggregate proxy for
    "how good is this opponent's future survival odds", used only to build
    the proportional pool-equity share, not the current week's exact
    mechanics.
    """
    avg = opponent_descriptive_features.get("avg_pick_win_prob")
    if avg is None:
        avg = DEFAULT_POPULATION_AVG_PICK_WIN_PROB
    future_surv = avg ** max(remaining_weeks, 0)
    return OpponentFutureEstimate(
        opponent_id="?", remaining_weeks=remaining_weeks,
        avg_pick_win_prob=avg, future_survival_prob=future_surv,
    )


@dataclass
class CandidateEvaluation:
    team: str
    vegas_win_prob: float
    p_survive_this_week: float
    my_future_survival_value: float          # exact, from pure_survival_value using this pick now
    future_opportunity_cost: float           # exact
    expected_opponents_eliminated: float     # exact, this week
    expected_remaining_pool: float           # exact, this week
    remaining_pool_distribution: dict        # exact, this week
    p_win_pool_approx: float                 # APPROXIMATE (see module docstring)


def evaluate_candidates(
    team_registry: TeamRegistry,
    my_used_mask: int,
    week: int,
    future_slates: dict[int, WeekSlate],
    opponent_predictions: dict[str, dict[str, float]],  # {opp_id: {team: prob}} for THIS week only
    opponent_descriptive_features: dict[str, dict],      # {opp_id: features} for future-survival proxy
    pool_size_before: int,
    total_remaining_weeks: int,
) -> list[CandidateEvaluation]:
    """
    Evaluate every legal team I could pick this week under both objectives.
    This is the function the weekly workflow calls to build the Top-5 (spec
    section 17). Nothing here mutates any state — pure computation.
    """
    from .opponents import aggregate_count_vector

    slate = future_slates[week]
    win_probs = slate.win_probs()
    count_vector = aggregate_count_vector(opponent_predictions)

    memo: dict = {}
    results: list[CandidateEvaluation] = []

    for team in slate.teams():
        bit = team_registry.bit(team)
        if my_used_mask & bit:
            continue

        p_survive = win_probs[team]

        cont_val, _ = pure_survival_value(team_registry, my_used_mask | bit,
                                           week + 1, future_slates, memo)
        my_future_value = p_survive * cont_val

        opp_cost = future_opportunity_cost(team, team_registry, my_used_mask,
                                            week, future_slates, memo)

        # This week's exact opponent-elimination mechanics, CONDITIONAL on
        # my own team winning (i.e. conditional on me surviving). This is
        # what makes the ranking sensitive to how crowded my pick is:
        # sharing a team with a large chunk of the field means their fate
        # is tied to mine, while an independent, lightly-owned pick leaves
        # the rest of the field's randomness fully intact.
        dist = pool_math.remaining_pool_distribution_given_my_team_won(
            slate, count_vector, team)
        summary = pool_math.summarize(dist, pool_size_before)

        # Approximate terminal pool-win share.
        #
        # IMPORTANT: this has to be P(I survive this week) * P(I eventually
        # win | I survived this week) -- both halves conditioned on the SAME
        # event. An earlier version of this put the marginal p_survive only
        # into the numerator while leaving the competitor-pool term computed
        # some other way, which let extreme long-shots on the *opposite*
        # side of a chalk block look artificially attractive: conditional on
        # such a long shot hitting, the correlated chalk block is wiped out
        # too, making the conditional competitor pool tiny -- but nothing
        # was discounting for how rarely that branch happens in the first
        # place. Multiplying p_survive_this_week on the OUTSIDE of the ratio
        # (rather than baking it only into the numerator) fixes that.
        remaining_weeks_after = max(total_remaining_weeks - 1, 0)
        opp_future_factors = [
            estimate_opponent_future_survival(feat, remaining_weeks_after).future_survival_prob
            for feat in opponent_descriptive_features.values()
        ]
        avg_opp_future_factor = (
            sum(opp_future_factors) / len(opp_future_factors) if opp_future_factors else 0.0
        )
        # expected_remaining_pool is already conditional on THIS candidate
        # team having won this week (pool_math.remaining_pool_distribution_given_my_team_won).
        expected_competitor_future_given_survival = (
            summary["expected_remaining_pool"] * avg_opp_future_factor
        )
        denom = cont_val + expected_competitor_future_given_survival
        p_win_given_survival = (cont_val / denom) if denom > 0 else 0.0
        p_win_pool = p_survive * p_win_given_survival

        results.append(CandidateEvaluation(
            team=team,
            vegas_win_prob=p_survive,
            p_survive_this_week=p_survive,
            my_future_survival_value=my_future_value,
            future_opportunity_cost=opp_cost,
            expected_opponents_eliminated=summary["expected_opponents_eliminated"],
            expected_remaining_pool=summary["expected_remaining_pool"],
            remaining_pool_distribution=dist,
            p_win_pool_approx=p_win_pool,
        ))

    return results


def rank_pure_survival(evals: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    return sorted(evals, key=lambda e: e.my_future_survival_value, reverse=True)


def rank_pool_equity(evals: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    return sorted(evals, key=lambda e: e.p_win_pool_approx, reverse=True)
