"""
Opponent modeling (spec sections 4-9, 22).

Core idea: predicting what an opponent will pick this week is a *discrete
choice* problem — the set of alternatives (available teams) changes every
week, and the alternative someone doesn't pick still shaped their decision.
That's exactly what a conditional logit model is for, so that's what this
uses, rather than a hand-built weighted formula.

    P(opponent picks team i | available set S) = exp(beta . x_i) / sum_{j in S} exp(beta . x_j)

Features x_i are computed per (team, week-context) and must *vary across
alternatives* within a choice set for a conditional logit — a feature that's
the same for every team that week (like "week number") would cancel out of
the softmax, so it's deliberately left out.

Each opponent gets their own weight vector beta, fit by maximum likelihood on
their pick history, but *shrunk toward a population-level beta* via an L2
penalty whose strength decays with the opponent's sample size. With ~18 picks
a season this is essential — the spec is explicit that the system should not
draw strong conclusions from a handful of observations (section 6).

Persona labels (Chalk Player, Contrarian, etc.) are computed only for human
-readable reporting, as a nearest-archetype description of the *fitted
feature weights* — they never feed back into the model itself, so the system
never hard-codes a label and then reasons from it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .vegas import WeekSlate

FEATURE_NAMES = ["win_prob", "rank_pct", "is_big_favorite", "future_utility"]
N_FEATURES = len(FEATURE_NAMES)

# Population-level prior weights: mild chalk bias (people lean favorite),
# and a mild aversion to "spending" a team with high future utility, all
# rescaled to be modest until data says otherwise. This is a *prior*, not a
# hard rule — with enough opponent-specific data it gets overridden.
POPULATION_PRIOR_BETA = np.array([2.5, -1.0, 0.3, -0.5])

# How much to trust the population prior vs. this opponent's own data.
# Effective L2 strength = SHRINKAGE_LAMBDA0 / (1 + n_observations)
SHRINKAGE_LAMBDA0 = 8.0


def _features_for_team(team: str, win_prob: float, rank_pct: float,
                        future_utility: float) -> np.ndarray:
    is_big_fav = 1.0 if win_prob >= 0.75 else 0.0
    return np.array([win_prob, rank_pct, is_big_fav, future_utility])


def build_choice_features(available_teams: list[str],
                           win_probs: dict[str, float],
                           future_utility: dict[str, float] | None = None
                           ) -> dict[str, np.ndarray]:
    """
    Build the feature vector for every alternative in a choice set.

    rank_pct: 0.0 = highest win-prob option available that week, 1.0 = lowest.
    future_utility: optional team -> expected value of that team in future
        weeks (e.g. average of its win probs in upcoming known/projected
        games). Defaults to the team's *current* win prob if not supplied,
        i.e. "no real future-value signal available yet".
    """
    future_utility = future_utility or {}
    ranked = sorted(available_teams, key=lambda t: win_probs[t], reverse=True)
    n = len(ranked)
    rank_pct_of = {t: (i / (n - 1) if n > 1 else 0.0) for i, t in enumerate(ranked)}

    feats = {}
    for t in available_teams:
        wp = win_probs[t]
        fu = future_utility.get(t, wp)
        feats[t] = _features_for_team(t, wp, rank_pct_of[t], fu)
    return feats


@dataclass
class PickObservation:
    week: int
    picked_team: str
    available_teams: list[str]
    win_probs: dict[str, float]
    future_utility: dict[str, float] = field(default_factory=dict)


@dataclass
class OpponentModel:
    opponent_id: str
    history: list[PickObservation] = field(default_factory=list)
    used_mask: int = 0
    eliminated: bool = False
    beta: np.ndarray = field(default_factory=lambda: POPULATION_PRIOR_BETA.copy())

    # ---- fitting -----------------------------------------------------
    def fit(self, n_iters: int = 300, lr: float = 0.15) -> None:
        """
        Refit this opponent's beta via gradient ascent on log-likelihood of
        their observed picks, MAP-regularized (L2) toward the population
        prior. Shrinkage strength decays with sample size (section 6:
        "shrinkage toward a population-level/default opponent model when an
        individual opponent has insufficient data").
        """
        n = len(self.history)
        if n == 0:
            self.beta = POPULATION_PRIOR_BETA.copy()
            return

        lam = SHRINKAGE_LAMBDA0 / (1.0 + n)
        beta = self.beta.copy()

        # Feature matrices don't depend on beta, so build them once rather
        # than on every iteration (same math, much faster late in a season).
        prepared = []
        for obs in self.history:
            feats = build_choice_features(obs.available_teams, obs.win_probs,
                                           obs.future_utility)
            xs = np.stack([feats[t] for t in obs.available_teams])
            prepared.append((xs, obs.available_teams.index(obs.picked_team)))

        for _ in range(n_iters):
            grad = -lam * (beta - POPULATION_PRIOR_BETA)  # regularization term
            for xs, chosen_idx in prepared:
                scores = xs @ beta
                scores = scores - scores.max()
                probs = np.exp(scores)
                probs /= probs.sum()
                # d(log P(chosen))/d(beta) = x_chosen - E[x]
                grad += xs[chosen_idx] - (probs[:, None] * xs).sum(axis=0)
            beta += lr * grad / max(1, n)

        self.beta = beta

    # ---- prediction ----------------------------------------------------
    def predict(self, available_teams: list[str], win_probs: dict[str, float],
                future_utility: dict[str, float] | None = None) -> dict[str, float]:
        """P(pick = team) for each available team this week."""
        if not available_teams:
            return {}
        feats = build_choice_features(available_teams, win_probs, future_utility)
        xs = np.stack([feats[t] for t in available_teams])
        scores = xs @ self.beta
        scores = scores - scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        return {t: float(p) for t, p in zip(available_teams, probs)}

    # ---- updating (post-decision only, section 22) ----------------------
    def observe_pick(self, week: int, picked_team: str, available_teams: list[str],
                      win_probs: dict[str, float],
                      future_utility: dict[str, float] | None = None,
                      refit: bool = True) -> None:
        """Record one observed pick. `refit=False` defers fitting so a caller
        replaying a whole season of history can fit once at the end."""
        self.history.append(PickObservation(
            week=week, picked_team=picked_team, available_teams=list(available_teams),
            win_probs=dict(win_probs), future_utility=dict(future_utility or {}),
        ))
        if refit:
            self.fit()

    def observe_result(self, team_won: bool) -> None:
        if not team_won:
            self.eliminated = True

    # ---- descriptive persona (reporting only) ---------------------------
    def descriptive_features(self) -> dict[str, float]:
        if not self.history:
            return {"chalk_rate": None, "avg_pick_win_prob": None,
                     "avg_rank_pct": None, "contrarian_rate": None, "n_obs": 0}
        chalk = 0
        contrarian = 0
        win_probs_of_picks = []
        rank_pcts = []
        for obs in self.history:
            feats = build_choice_features(obs.available_teams, obs.win_probs, obs.future_utility)
            ranked = sorted(obs.available_teams, key=lambda t: obs.win_probs[t], reverse=True)
            is_chalk = (obs.picked_team == ranked[0])
            chalk += int(is_chalk)
            win_probs_of_picks.append(obs.win_probs[obs.picked_team])
            rp = feats[obs.picked_team][1]  # rank_pct feature
            rank_pcts.append(rp)
            contrarian += int(rp > 0.5)
        n = len(self.history)
        return {
            "chalk_rate": chalk / n,
            "avg_pick_win_prob": float(np.mean(win_probs_of_picks)),
            "avg_rank_pct": float(np.mean(rank_pcts)),
            "contrarian_rate": contrarian / n,
            "n_obs": n,
        }

    def persona_label(self) -> str:
        """Nearest-archetype label for reporting only — not used by the model."""
        f = self.descriptive_features()
        if f["n_obs"] < 3:
            return "Insufficient data"
        if f["chalk_rate"] >= 0.75:
            return "Chalk Player"
        if f["contrarian_rate"] >= 0.5:
            return "Contrarian Player"
        if f["avg_pick_win_prob"] >= 0.72:
            return "Favorite-First Player"
        return "Mixed / Balanced Player"

    def confidence(self) -> str:
        n = len(self.history)
        if n >= 8:
            return "HIGH"
        if n >= 3:
            return "MEDIUM"
        return "LOW"


def aggregate_count_vector(opponent_predictions: dict[str, dict[str, float]]
                            ) -> dict[str, float]:
    """
    Turn {opponent_id: {team: prob}} into the pool-level expected count
    vector {team: expected number of opponents picking that team}
    (spec section 7).
    """
    counts: dict[str, float] = {}
    for _, dist in opponent_predictions.items():
        for team, p in dist.items():
            counts[team] = counts.get(team, 0.0) + p
    return counts
