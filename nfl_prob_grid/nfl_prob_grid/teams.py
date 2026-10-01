"""Team-name normalisation.

Canonical team ids are the standard 2-3 letter abbreviations.  Any raw name (full name,
nickname, unambiguous city, common abbreviation variants) is resolved to a canonical id.
Extra aliases can be supplied via ``[team_aliases]`` in config.toml.  Ambiguous names
("New York", "Los Angeles", "LA") are deliberately NOT mapped: they are rejected and
reported rather than guessed.
"""
from __future__ import annotations

import re

from .errors import ConfigError

# (abbr, city, nickname)
TEAM_TABLE: list[tuple[str, str, str]] = [
    ("ARI", "Arizona", "Cardinals"), ("ATL", "Atlanta", "Falcons"),
    ("BAL", "Baltimore", "Ravens"), ("BUF", "Buffalo", "Bills"),
    ("CAR", "Carolina", "Panthers"), ("CHI", "Chicago", "Bears"),
    ("CIN", "Cincinnati", "Bengals"), ("CLE", "Cleveland", "Browns"),
    ("DAL", "Dallas", "Cowboys"), ("DEN", "Denver", "Broncos"),
    ("DET", "Detroit", "Lions"), ("GB", "Green Bay", "Packers"),
    ("HOU", "Houston", "Texans"), ("IND", "Indianapolis", "Colts"),
    ("JAX", "Jacksonville", "Jaguars"), ("KC", "Kansas City", "Chiefs"),
    ("LV", "Las Vegas", "Raiders"), ("LAC", "Los Angeles", "Chargers"),
    ("LAR", "Los Angeles", "Rams"), ("MIA", "Miami", "Dolphins"),
    ("MIN", "Minnesota", "Vikings"), ("NE", "New England", "Patriots"),
    ("NO", "New Orleans", "Saints"), ("NYG", "New York", "Giants"),
    ("NYJ", "New York", "Jets"), ("PHI", "Philadelphia", "Eagles"),
    ("PIT", "Pittsburgh", "Steelers"), ("SF", "San Francisco", "49ers"),
    ("SEA", "Seattle", "Seahawks"), ("TB", "Tampa Bay", "Buccaneers"),
    ("TEN", "Tennessee", "Titans"), ("WAS", "Washington", "Commanders"),
]
CANONICAL_TEAMS: list[str] = sorted(t[0] for t in TEAM_TABLE)
_AMBIGUOUS_CITIES = {"new york", "los angeles"}

_EXTRA = {
    "jac": "JAX", "lvr": "LV", "oak": "LV", "oakland raiders": "LV", "wsh": "WAS",
    "was": "WAS", "washington football team": "WAS", "sd": "LAC",
    "san diego chargers": "LAC", "stl": "LAR", "st louis rams": "LAR",
    "ny giants": "NYG", "ny jets": "NYJ", "la rams": "LAR", "la chargers": "LAC",
    "gnb": "GB", "kan": "KC", "nwe": "NE", "nor": "NO", "sfo": "SF", "tam": "TB",
    "n y giants": "NYG", "n y jets": "NYJ",
}


def _key(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[.'’]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class TeamNormalizer:
    """Callable: raw name -> canonical id, or ``None`` if unknown/ambiguous."""

    def __init__(self, extra_aliases: dict | None = None):
        m: dict[str, str] = {}
        for abbr, city, nick in TEAM_TABLE:
            for alias in (abbr, f"{city} {nick}", nick):
                m[_key(alias)] = abbr
            if _key(city) not in _AMBIGUOUS_CITIES:
                m[_key(city)] = abbr
        m.update(_EXTRA)
        for raw, canon in (extra_aliases or {}).items():
            if canon not in CANONICAL_TEAMS:
                raise ConfigError(f"[team_aliases] {raw!r} -> {canon!r}: not a canonical team id")
            m[_key(raw)] = canon
        self._map = m

    def __call__(self, name) -> str | None:
        if name is None:
            return None
        return self._map.get(_key(name))

    @property
    def teams(self) -> list[str]:
        return list(CANONICAL_TEAMS)
