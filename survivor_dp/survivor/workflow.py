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
    market_comparison: Optional[dict] = None  # team -> {"model": x, "market": y} when available


class SurvivorPool:
    def __init__(self):
        self.registry = TeamRegistry()
        self.my_used_mask = 0
        self.current_week = 1
        self.opponents: dict = {}
        self.future_slates: dict = {}
        self.weekly_records: dict = {}
        self.total_season_weeks: Optional[int] = None
        self.load_report = None

    def add_opponent(self, opponent_id: str) -> OpponentModel:
        if opponent_id not in self.opponents:
            self.opponents[opponent_id] = OpponentModel(opponent_id=opponent_id)
        return self.opponents[opponent_id]

    def set_week_slate(self, slate: WeekSlate) -> None:
        self.future_slates[slate.week] = slate

    def set_my_used_teams(self, teams: list) -> None:
        self.my_used_mask = self.registry.mask_from_teams(teams)

    def set_opponent_used_teams(self, opponent_id: str, teams: list) -> None:
        self.add_opponent(opponent_id).used_mask = self.registry.mask_from_teams(teams)

    def eliminate_opponent(self, opponent_id: str) -> None:
        self.add_opponent(opponent_id).eliminated = True

    def alive_opponents(self) -> dict:
        return {oid: o for oid, o in self.opponents.items() if not o.eliminated}

    def _remaining_weeks(self) -> int:
        if self.total_season_weeks is not None:
            return max(self.total_season_weeks - self.current_week + 1, 0)
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

    def get_recommendations(self, top_n: int = 5,
                            market_pick_pct: Optional[dict] = None,
                            market_blend_weight: Optional[float] = None
                            ) -> WeeklyRecommendation:
        """
        market_pick_pct: optional {team: fraction_of_field_in_0_to_1}, e.g.
            from survivorgrid.fetch_week(). When given, the aggregated
            opponent count vector is blended toward this real market
            distribution (see _blend_count_vector for exactly how).
        market_blend_weight: 0 = ignore market data entirely, 1 = use ONLY
            the market distribution, ignoring the fitted opponent models.
            None (default) auto-picks a weight based on how much pick
            history has actually been logged for this pool (see
            _auto_blend_weight) -- more logged history, less reliance on
            the broader market, since your specific small pool can and will
            diverge from it.
        """
        wk = self.current_week
        if wk not in self.future_slates:
            raise ValueError(
                f"No probabilities loaded for week {wk}. Call set_week_slate() "
                f"with at least one game for this week before requesting a "
                f"recommendation."
            )
        slate = self.future_slates[wk]
        win_probs = slate.win_probs()

        opponent_predictions: dict = {}
        opponent_features: dict = {}
        for oid, opp in self.alive_opponents().items():
            avail = [t for t in slate.teams() if not self.registry.is_used(opp.used_mask, t)]
            if not avail:
                continue
            opponent_predictions[oid] = opp.predict(avail, win_probs)
            opponent_features[oid] = opp.descriptive_features()

        pool_size_before = len(self.alive_opponents())
        remaining_weeks = self._remaining_weeks()

        market_comparison = None
        if market_pick_pct:
            model_count_vector = aggregate_count_vector(opponent_predictions)
            weight = (market_blend_weight if market_blend_weight is not None
                     else _auto_blend_weight(self.alive_opponents()))
            blended = _blend_count_vector(model_count_vector, market_pick_pct,
                                          pool_size_before, weight)
            market_comparison = {
                t: {"model": model_count_vector.get(t, 0.0) / pool_size_before if pool_size_before else 0.0,
                    "market": market_pick_pct.get(t, 0.0),
                    "blended": blended.get(t, 0.0) / pool_size_before if pool_size_before else 0.0}
                for t in set(model_count_vector) | set(market_pick_pct)
                if t in win_probs
            }
            # aggregate_count_vector (called again inside evaluate_candidates)
            # just sums per-entry dicts, so one synthetic "opponent" holding
            # the already-blended counts reproduces `blended` exactly.
            opponent_predictions = {"__blended__": dict(blended)}

        evals = dp.evaluate_candidates(
            self.registry, self.my_used_mask, wk, self.future_slates,
            opponent_predictions, opponent_features, pool_size_before, remaining_weeks,
        )

        rec = self.weekly_records.setdefault(wk, WeeklyRecord(week=wk))
        rec.recommendations_generated = True

        return WeeklyRecommendation(
            week=wk,
            pure_survival_top=dp.rank_pure_survival(evals)[:top_n],
            pool_equity_top=dp.rank_pool_equity(evals)[:top_n],
            confidence=self._confidence_report(),
            count_vector=aggregate_count_vector(opponent_predictions),
            market_comparison=market_comparison,
        )

    def record_my_pick(self, team: str) -> None:
        rec = self.weekly_records.setdefault(self.current_week, WeeklyRecord(week=self.current_week))
        rec.my_pick = team

    def record_opponent_picks(self, picks: dict) -> None:
        wk = self.current_week
        slate = self.future_slates[wk]
        win_probs = slate.win_probs()
        rec = self.weekly_records.setdefault(wk, WeeklyRecord(week=wk))
        for oid, team in picks.items():
            opp = self.add_opponent(oid)
            avail = [t for t in slate.teams() if not self.registry.is_used(opp.used_mask, t)]
            if team not in avail:
                avail = list(set(avail) | {team})
            opp.observe_pick(wk, team, avail, win_probs)
            opp.used_mask |= self.registry.bit(team)
            rec.opponent_actual_picks[oid] = team

    def record_results(self, results: dict) -> None:
        wk = self.current_week
        rec = self.weekly_records.setdefault(wk, WeeklyRecord(week=wk))
        rec.results.update(results)
        for oid, team in rec.opponent_actual_picks.items():
            if team in results:
                self.opponents[oid].observe_result(results[team])

    def advance_week(self) -> None:
        rec = self.weekly_records.get(self.current_week)
        if rec is None or rec.my_pick is None:
            raise ValueError(
                "Cannot advance: no pick recorded for the current week. "
                "Call record_my_pick(team) first."
            )
        self.my_used_mask |= self.registry.bit(rec.my_pick)
        self.current_week += 1


def _auto_blend_weight(alive_opponents: dict) -> float:
    """More logged pick history for THIS pool -> trust the fitted per-
    opponent models more; little/no history -> lean on the broader market
    distribution instead of an unvalidated prior. Averages each alive
    opponent's observation count, maps 0 obs -> weight 0.15 (mostly market),
    8+ obs -> weight 0.85 (mostly the fitted models), linear between."""
    if not alive_opponents:
        return 0.15
    avg_n = sum(len(o.history) for o in alive_opponents.values()) / len(alive_opponents)
    lo, hi = 0.15, 0.85
    return lo + (hi - lo) * min(avg_n, 8) / 8


def _blend_count_vector(model_count_vector: dict, market_pick_pct: dict,
                        pool_size: int, weight: float) -> dict:
    """Blend the fitted-model aggregate count vector with a real market
    percentage-per-team distribution, both rescaled to the same pool size
    and renormalized so counts still sum to pool_size exactly.
    weight = how much to trust the fitted models (1-weight = market)."""
    teams = set(model_count_vector) | set(market_pick_pct)
    model_total = sum(model_count_vector.values()) or 1.0
    market_total = sum(market_pick_pct.values()) or 1.0
    blended = {}
    for t in teams:
        model_share = model_count_vector.get(t, 0.0) / model_total
        market_share = market_pick_pct.get(t, 0.0) / market_total
        blended[t] = (weight * model_share + (1 - weight) * market_share) * pool_size
    return blended
