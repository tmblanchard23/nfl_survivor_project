"""Generate the Team x Week win-probability grid from the master game table.

The master table remains the source of truth; the grid is a pure derived view.  Bye weeks
are left blank (NaN).  Each team appears at most once per week (enforced by validation)."""
from __future__ import annotations

import pandas as pd

from .config import Config
from .errors import Issue, ValidationError
from .teams import CANONICAL_TEAMS


def build_grid(master: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    home = master[["home_team", "week", "home_win_prob"]].set_axis(["team", "week", "p"], axis=1)
    away = master[["away_team", "week", "away_win_prob"]].set_axis(["team", "week", "p"], axis=1)
    long = pd.concat([home, away], ignore_index=True)
    if long.duplicated(["team", "week"]).any():
        raise ValidationError([Issue("error", "team_plays_twice",
                                     "a team has two games in one week; cannot build grid")])
    grid = long.pivot(index="team", columns="week", values="p")
    grid = grid.reindex(index=CANONICAL_TEAMS, columns=list(range(1, cfg.season.n_weeks + 1)))
    if cfg.output.grid_decimals >= 0:
        grid = grid.round(cfg.output.grid_decimals)
    grid.columns = [f"{cfg.output.grid_week_prefix}{w}" for w in grid.columns]
    grid.index.name = cfg.output.grid_team_column
    return grid
