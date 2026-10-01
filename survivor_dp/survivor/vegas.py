"""
Vegas is the source of truth (spec section 1).

This module represents a week's slate of games and their win probabilities.
It can compute a no-vig probability from raw sportsbook odds (American
moneylines or decimal odds) when you have those -- but the primary path now
is `Game.from_probability()`, which takes an ALREADY-COMPUTED win
probability directly. That's what your probability-grid engine's "Master
Game Table" hands over: a finished, de-vigged, stale-line-adjusted number
per game. This module doesn't re-derive or second-guess it -- the grid
engine already owns that problem (rating updates, anchoring, stale-line
adjustment, all backtested separately). Re-deriving anything here would be
duplicating work that's already done, and done better, upstream.

No subjective power ratings live anywhere in this file, or anywhere in this
package.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

OddsFormat = Literal["american", "decimal", "probability"]


def moneyline_to_implied_prob(moneyline: float) -> float:
    """Raw (vig-included) implied probability from an American moneyline."""
    if moneyline == 0:
        raise ValueError("Moneyline cannot be 0")
    if moneyline < 0:
        return (-moneyline) / ((-moneyline) + 100.0)
    return 100.0 / (moneyline + 100.0)


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Raw (vig-included) implied probability from decimal ("European") odds.

    e.g. 1.51 -> 1/1.51 = 66.2%, 2.64 -> 1/2.64 = 37.9%
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def implied_prob(odds: float, odds_format: OddsFormat = "american") -> float:
    if odds_format == "decimal":
        return decimal_to_implied_prob(odds)
    if odds_format == "american":
        return moneyline_to_implied_prob(odds)
    if odds_format == "probability":
        if not (0.0 < odds < 1.0):
            raise ValueError(f"Probability must be in (0, 1), got {odds}")
        return odds
    raise ValueError(f"Unknown odds_format: {odds_format!r}")


def no_vig_probs(odds_a: float, odds_b: float,
                  odds_format: OddsFormat = "american") -> tuple[float, float]:
    """
    Normalize two implied probabilities so they sum to 1.

    For odds_format="american"/"decimal", this is the actual vig-removal
    step: raw implied probabilities from each side, normalized.

    For odds_format="probability" (the grid-engine path), odds_a/odds_b are
    ALREADY de-vigged win probabilities -- this just guards against the tiny
    rounding drift you'd expect from a sheet showing "39.7%"/"60.3%" (sums to
    100.0 here, but wouldn't always to the last decimal) rather than doing
    any real vig removal.

    Example from the spec (American):
        favorite -200 -> 66.67% raw
        underdog +170 -> 37.04% raw
        no-vig favorite = 66.67 / (66.67 + 37.04) = 64.28%
    """
    pa = implied_prob(odds_a, odds_format)
    pb = implied_prob(odds_b, odds_format)
    total = pa + pb
    if total <= 0:
        raise ValueError("Degenerate market: probabilities sum to 0")
    return pa / total, pb / total


@dataclass(frozen=True)
class Game:
    """One game in a week's slate.

    `odds_format` controls how `moneyline_a`/`moneyline_b` are interpreted
    (names kept for backward compatibility across all three formats):
      - "american": moneyline_a/b are American moneylines (e.g. -200, +170)
      - "decimal":   moneyline_a/b are decimal odds (e.g. 1.50, 2.70)
      - "probability": moneyline_a/b are ALREADY win probabilities in (0, 1)
        -- the format used when reading a probability-grid engine's output
        directly (see Game.from_probability / data_io's Master Game Table
        loader). No de-vig math is performed in this case, just the
        rounding-drift normalization described in no_vig_probs().

    `line_last_updated` is an optional raw timestamp string, kept purely for
    display/reference if your data source provides one (e.g. a "Last Line
    Update" column). This system does not compute its own staleness signal
    from it -- if your probability source already accounts for line age
    (e.g. a grid engine's own stale-line adjustment), re-deriving that here
    would just be duplicating work that's already done upstream, probably
    less accurately.
    """
    week: int
    team_a: str
    team_b: str
    moneyline_a: float
    moneyline_b: float
    odds_format: OddsFormat = "american"
    line_last_updated: str | None = None

    def no_vig_win_probs(self) -> dict[str, float]:
        pa, pb = no_vig_probs(self.moneyline_a, self.moneyline_b, self.odds_format)
        return {self.team_a: pa, self.team_b: pb}

    def opponent_of(self, team: str) -> str:
        if team == self.team_a:
            return self.team_b
        if team == self.team_b:
            return self.team_a
        raise KeyError(f"{team} is not in this game")

    @classmethod
    def from_decimal(cls, week: int, team_a: str, team_b: str,
                      decimal_a: float, decimal_b: float,
                      line_last_updated: str | None = None) -> "Game":
        return cls(week, team_a, team_b, decimal_a, decimal_b,
                   odds_format="decimal", line_last_updated=line_last_updated)

    @classmethod
    def from_american(cls, week: int, team_a: str, team_b: str,
                       moneyline_a: float, moneyline_b: float,
                       line_last_updated: str | None = None) -> "Game":
        return cls(week, team_a, team_b, moneyline_a, moneyline_b,
                   odds_format="american", line_last_updated=line_last_updated)

    @classmethod
    def from_probability(cls, week: int, team_a: str, team_b: str,
                          prob_a: float, prob_b: float | None = None,
                          line_last_updated: str | None = None) -> "Game":
        """Build a Game from an already-computed win probability -- the
        primary path for a probability-grid engine's output. `prob_b`
        defaults to `1 - prob_a` if not given (e.g. if your source only
        reports one side)."""
        if prob_b is None:
            prob_b = 1.0 - prob_a
        return cls(week, team_a, team_b, prob_a, prob_b,
                   odds_format="probability", line_last_updated=line_last_updated)


@dataclass
class WeekSlate:
    """All games for a given week."""
    week: int
    games: list[Game] = field(default_factory=list)

    def win_probs(self) -> dict[str, float]:
        """team -> no-vig win probability, for every team playing this week."""
        probs: dict[str, float] = {}
        for g in self.games:
            probs.update(g.no_vig_win_probs())
        return probs

    def teams(self) -> list[str]:
        return list(self.win_probs().keys())

    def game_for_team(self, team: str) -> Game:
        for g in self.games:
            if team in (g.team_a, g.team_b):
                return g
        raise KeyError(f"{team} not found in week {self.week} slate")
