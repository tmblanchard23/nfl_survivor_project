"""
Formats a WeeklyRecommendation into the report structure spec section 17-19
asks for: two ranked tables, a qualitative explanation per pool-equity
candidate, scenario analysis, and a confidence readout.

All numbers here come straight from CandidateEvaluation / WeeklyRecommendation
-- this module only formats, it does not compute anything new, so there's
nothing here that could silently reintroduce an ad hoc scoring formula.
"""
from __future__ import annotations
from .workflow import WeeklyRecommendation
from .dp import CandidateEvaluation


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_signed(x: float) -> str:
    return f"{x:+.4f}"


def ranking_a_table(rec: WeeklyRecommendation) -> str:
    lines = ["| Rank | Team | Vegas W% | P(Survive) | Future Opportunity Cost |",
             "|---|---|---:|---:|---:|"]
    for i, c in enumerate(rec.pure_survival_top, 1):
        lines.append(f"| {i} | {c.team} | {_pct(c.vegas_win_prob)} | "
                      f"{_pct(c.p_survive_this_week)} | {_fmt_signed(c.future_opportunity_cost)} |")
    return "\n".join(lines)


def ranking_b_table(rec: WeeklyRecommendation) -> str:
    lines = ["| Rank | Team | Vegas W% | P(Survive) | P(Win Pool)* | Exp. Opp. Eliminated | Exp. Remaining Pool |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for i, c in enumerate(rec.pool_equity_top, 1):
        lines.append(
            f"| {i} | {c.team} | {_pct(c.vegas_win_prob)} | {_pct(c.p_survive_this_week)} | "
            f"{_pct(c.p_win_pool_approx)} | {c.expected_opponents_eliminated:.2f} | "
            f"{c.expected_remaining_pool:.2f} |"
        )
    return "\n".join(lines)


def _candidate_ownership(rec: WeeklyRecommendation, team: str) -> float:
    return rec.count_vector.get(team, 0.0)


def candidate_explanation(rec: WeeklyRecommendation, c: CandidateEvaluation,
                           pool_size_before: int) -> str:
    ownership = _candidate_ownership(rec, c.team)
    ownership_pct = ownership / pool_size_before if pool_size_before else 0.0

    why = (f"{_pct(c.p_survive_this_week)} survival probability from Vegas, combined with "
           f"a {_pct(c.p_win_pool_approx)} approximate share of ultimately winning the pool "
           f"from this point.")

    if ownership_pct >= 0.4:
        interaction = (f"Roughly {_pct(ownership_pct)} of the opponent field is predicted to "
                        f"also be on {c.team} this week, so a loss here eliminates a large chunk "
                        f"of the field alongside you -- low differentiation value if it wins.")
    elif ownership_pct <= 0.1:
        interaction = (f"Only about {_pct(ownership_pct)} of the opponent field is predicted to "
                        f"share this pick, so surviving here meaningfully separates you from "
                        f"the crowd riding more popular teams.")
    else:
        interaction = f"Moderate overlap (~{_pct(ownership_pct)}) with the predicted opponent field."

    if c.future_opportunity_cost > 0.02:
        future = (f"This team carries real future value -- the model estimates preserving it "
                  f"would be worth {c.future_opportunity_cost:.4f} more in survival-probability "
                  f"terms than spending it now.")
    elif c.future_opportunity_cost < -0.02:
        future = "No real cost to using this team now; it's not clearly more valuable saved for later."
    else:
        future = "Using it now vs. saving it appears roughly a wash."

    risk = (f"Vegas gives this a {_pct(1 - c.vegas_win_prob)} chance of losing outright, which "
            f"ends your season immediately regardless of any pool-equity upside.")

    return (f"**{c.team}**\n"
            f"- Why DP likes this pick: {why}\n"
            f"- Opponent interaction: {interaction}\n"
            f"- Future opportunity cost: {future}\n"
            f"- What could make this pick fail strategically: {risk}")


def scenario_analysis(c: CandidateEvaluation) -> str:
    win_line = (f"If {c.team} wins: your survival continues, "
                f"~{c.expected_opponents_eliminated:.1f} opponents are expected to be "
                f"eliminated this week, leaving an expected pool of "
                f"~{c.expected_remaining_pool:.1f} (conditional on your own survival).")
    lose_line = f"If {c.team} loses: you are eliminated."
    dist_note = (f"Full conditional distribution over remaining pool size has "
                 f"{len(c.remaining_pool_distribution)} distinct outcomes; "
                 f"probabilities sum to "
                 f"{sum(c.remaining_pool_distribution.values()):.4f}.")
    return f"### {c.team}\n{win_line}\n\n{lose_line}\n\n{dist_note}"


def confidence_block(rec: WeeklyRecommendation) -> str:
    conf = rec.confidence
    lines = [f"- Vegas information: **{conf['vegas_information']}**"]
    lines.append(f"- Opponent model: **{conf['opponent_model']}**")
    lines.append(f"- Overall strategic confidence: **{conf['overall_strategic_confidence']}**")
    return "\n".join(lines)


def full_report(rec: WeeklyRecommendation, pool_size_before: int) -> str:
    parts = [
        f"# Week {rec.week} Recommendation\n",
        "## Ranking A -- Pure Survival\n",
        ranking_a_table(rec),
        "\n\n## Ranking B -- Pool Equity\n",
        "*P(Win Pool) is an approximation (proportional survival share) -- see "
        "dp.py module docstring. Everything else in this table is exact given "
        "the modeled inputs.*\n",
        ranking_b_table(rec),
        "\n\n## Why each pool-equity candidate is here\n",
    ]
    for c in rec.pool_equity_top:
        parts.append(candidate_explanation(rec, c, pool_size_before))
        parts.append("")
    parts.append("## Scenario Analysis\n")
    for c in rec.pool_equity_top:
        parts.append(scenario_analysis(c))
        parts.append("")
    parts.append("## Confidence\n")
    parts.append(confidence_block(rec))
    parts.append(
        "\n\n**Reminder:** this is a recommendation, not an action. Nothing has "
        "been marked as used -- the optimizer only learns what you actually "
        "picked once it's logged in the Pools Picks tab."
    )
    return "\n".join(parts)
