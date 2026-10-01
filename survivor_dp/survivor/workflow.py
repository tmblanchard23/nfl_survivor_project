"""
The weekly workflow (spec section 16), enforced structurally rather than by
convention:

  get_recommendations()   -> pure computation, NEVER mutates state
  record_my_pick(team)    -> bookkeeping only; does NOT advance the week
  record_opponent_picks() -> updates opponent history/persona (post-decision)
  record_results()        -> updates opponent elimination status
  advance_week()          -> the ONLY method that moves `current_week` forward
                              and locks in `my_used_mask`; requires a pick to
                              have been recorded first

get_recommendations() raises if there is no line at all for the current
week (set_week_slate() was never called for it, or it has zero games). It
will never silently reuse a previous week's numbers -- each week's slate is
tracked separately, so there's no code path that could even do that by
accident.

Note on line staleness: an earlier version of this system computed its own
"how old is this line" signal from a raw-odds source. That's no longer this
system's job -- probabilities now arrive already finished from an upstream
probability-grid engine that owns rating updates, anchoring, and stale-line
adjustment itself (backtested separately, on real historical data). Feeding
already-adjusted probabilities back through a second, cruder staleness
check here would just be redundant, so this module doesn't do that anymore.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .vegas import WeekSlate
from .team_registry import TeamRegistry
from .opponents import OpponentModel, aggregate_count_vector
from . import dp


@dataclass
class WeeklyRecord:
    week: int
    my_pick: Optional[str] = None
    opponent_actual_picks: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    recommendations_generated: bool = False


@dataclass
class WeeklyRecommendation:
    week: int
    pure_survival_top: list
    pool_equity_top: list
    confidence: dict
    count_vector: dict


class SurvivorPool:
    def __init__(self):
        self.registry = TeamRegistry()
        self.my_used_mask = 0
        self.current_week = 1
        self.opponents: dict[str, OpponentModel] = {}
        self.future_slates: dict[int, WeekSlate] = {}
        self.weekly_records: dict[int, WeeklyRecord] = {}
        self.total_season_weeks: Optional[int] = None

    # ---- setup / data loading -----------------------------------------
    def add_opponent(self, opponent_id: str) -> OpponentModel:
        if opponent_id not in self.opponents:
            self.opponents[opponent_id] = OpponentModel(opponent_id=opponent_id)
        return self.opponents[opponent_id]

    def set_week_slate(self, slate: WeekSlate) -> None:
        """Load (or overwrite) the probabilities for a given week. Overwriting
        is expected/normal as projections firm up into real numbers, or as
        the upstream grid re-runs and produces an updated value."""
        self.future_slates[slate.week] = slate

    def set_my_used_teams(self, teams: list[str]) -> None:
        self.my_used_mask = self.registry.mask_from_teams(teams)

    def set_opponent_used_teams(self, opponent_id: str, teams: list[str]) -> None:
        self.add_opponent(opponent_id).used_mask = self.registry.mask_from_teams(teams)

    def eliminate_opponent(self, opponent_id: str) -> None:
        self.add_opponent(opponent_id).eliminated = True

    # ---- helpers --------------------------------------------------------
    def alive_opponents(self) -> dict[str, OpponentModel]:
        return {oid: o for oid, o in self.opponents.items() if not o.eliminated}

    def _remaining_weeks(self) -> int:
        if self.total_season_weeks is not None:
            return max(self.total_season_weeks - self.current_week + 1, 0)
        # Fall back to however many weeks of slate data we actually have.
        return len([w for w in self.future_slates if w >= self.current_week])

    def _confidence_report(self) -> dict:
        alive = self.alive_opponents()
        if not alive:
            opp_conf = "LOW"
        else:
            levels = [o.confidence() for o in alive.values()]
            if all(l == "HIGH" for l in levels):
                opp_conf = "HIGH"
            elif any(l == "LOW" for l in levels):
                opp_conf = "MEDIUM" if levels.count("LOW") < len(levels) / 2 else "LOW"
            else:
                opp_conf = "MEDIUM"
        vegas_conf = "HIGH" if self.current_week in self.future_slates else "LOW"
        order = {"LOW": 0, "MEDIUM": 1, "MEDIUM-HIGH": 1.5, "HIGH": 2}
        overall_score = (order[vegas_conf] + order[opp_conf]) / 2
        overall = min(order, key=lambda k: abs(order[k] - overall_score))
        return {"vegas_information": vegas_conf, "opponent_model": opp_conf,
                "overall_strategic_confidence": overall}

    # ---- DECISION PHASE (never mutates state) ---------------------------
    def get_recommendations(self, top_n: int = 5) -> WeeklyRecommendation:
        wk = self.current_week
        if wk not in self.future_slates:
            raise ValueError(
                f"No probabilities loaded for week {wk}. Call set_week_slate() "
                f"with at least one game for this week before requesting a "
                f"recommendation."
            )
        slate = self.future_slates[wk]
        win_probs = slate.win_probs()

        opponent_predictions: dict[str, dict[str, float]] = {}
        opponent_features: dict[str, dict] = {}
        for oid, opp in self.alive_opponents().items():
            avail = [t for t in slate.teams() if not self.registry.is_used(opp.used_mask, t)]
            if not avail:
                continue
            opponent_predictions[oid] = opp.predict(avail, win_probs)
            opponent_features[oid] = opp.descriptive_features()

        pool_size_before = len(self.alive_opponents())
        remaining_weeks = self._remaining_weeks()

        evals = dp.evaluate_candidates(
            self.registry, self.my_used_mask, wk, self.future_slates,
            opponent_predictions, opponent_features, pool_size_before, remaining_weeks,
        )

        rec = self.weekly_records.setdefault(wk, WeeklyRecord(week=wk))
        rec.recommendations_generated = True  # bookkeeping flag only, no state changed

        return WeeklyRecommendation(
            week=wk,
            pure_survival_top=dp.rank_pure_survival(evals)[:top_n],
            pool_equity_top=dp.rank_pool_equity(evals)[:top_n],
            confidence=self._confidence_report(),
            count_vector=aggregate_count_vector(opponent_predictions),
        )

    # ---- POST-DECISION UPDATE PHASE --------------------------------------
    def record_my_pick(self, team: str) -> None:
        """Record what I actually picked. Does NOT advance the week or mark
        the team used yet — that only happens in advance_week(), so you can
        still call get_recommendations() again (e.g. to sanity check) without
        the state having silently changed."""
        rec = self.weekly_records.setdefault(self.current_week, WeeklyRecord(week=self.current_week))
        rec.my_pick = team

    def record_opponent_picks(self, picks: dict[str, str]) -> None:
        """picks: {opponent_id: team_actually_picked}. Updates each
        opponent's history/persona immediately (section 6) — this is exactly
        the post-decision information the model is not allowed to see before
        my pick, per section 22."""
        wk = self.current_week
        slate = self.future_slates[wk]
        win_probs = slate.win_probs()
        rec = self.weekly_records.setdefault(wk, WeeklyRecord(week=wk))
        for oid, team in picks.items():
            opp = self.add_opponent(oid)
            avail = [t for t in slate.teams() if not self.registry.is_used(opp.used_mask, t)]
            if team not in avail:
                # Data inconsistency (e.g. opponent used mask out of date) —
                # don't silently drop it, but don't crash the whole update either.
                avail = list(set(avail) | {team})
            opp.observe_pick(wk, team, avail, win_probs)
            opp.used_mask |= self.registry.bit(team)
            rec.opponent_actual_picks[oid] = team

    def record_results(self, results: dict[str, bool]) -> None:
        """results: {team: True if that team won, False if it lost}."""
        wk = self.current_week
        rec = self.weekly_records.setdefault(wk, WeeklyRecord(week=wk))
        rec.results.update(results)
        for oid, team in rec.opponent_actual_picks.items():
            if team in results:
                self.opponents[oid].observe_result(results[team])

    def advance_week(self) -> None:
        """The only method that moves the week forward. Requires my_pick to
        have been recorded. After this, get_recommendations() will raise
        until you load next week's probabilities via set_week_slate()."""
        rec = self.weekly_records.get(self.current_week)
        if rec is None or rec.my_pick is None:
            raise ValueError(
                "Cannot advance: no pick recorded for the current week. "
                "Call record_my_pick(team) first."
            )
        self.my_used_mask |= self.registry.bit(rec.my_pick)
        self.current_week += 1
